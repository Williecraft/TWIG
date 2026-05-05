#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Query-Aware GNN 評估腳本 v2

核心改進（相較原始 evaluate_retrieval_query_aware.py）：
1. 使用單一訓練完成的 QueryAwareModel（不會每查詢重建模型）
2. 模型的 query edge 權重是訓練過的（不是隨機初始化）
3. 使用 canonical metadata 確保模型結構一致
4. 支援可配置的粗排大小 (coarse_k)

Pipeline:
  1. 全圖 forward → 固定 table embedding（粗排用）
  2. 對每個查詢：粗排 top-k → 建子圖(含 query 節點) → 子圖 forward → 重排序
  3. 計算 Recall@1/5/10, MRR, nDCG@10 等指標
"""

import sys
import json
import math
from typing import List, Set

import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer

# 從 v2 訓練腳本匯入共用元件
from train_query_aware_v2 import (
    QueryAwareModel,
    get_canonical_metadata,
    build_subgraph,
    filter_edges,
    get_key_fields,
    make_key,
    rebuild_id_to_idx,
    BEST_EDGE_CONFIGS,
    QUERY_EDGE_MODE,
)

# ===========================
# 設定
# ===========================

MODEL_NAME = 'BAAI/bge-m3'
COARSE_K = 100            # 粗排取多少候選（增大以提高 recall ceiling）
TOP_K = 10                # 最終取 top-k 結果
ALPHA = 0.3               # 分數插值: final = alpha * coarse + (1-alpha) * rerank

QUERY_FILE = "/user_data/TabGNN/data/table/test/{source}/query.jsonl"
GRAPH_PATH = "/user_data/TabGNN/data/processed/test/{source}/graph.pt"
MODEL_PATH = "/user_data/TabGNN/checkpoints/{dataset}/model_query_aware_v2.pt"
TWIG_MODEL_PATH = "/user_data/TabGNN/checkpoints/{dataset}/model.pt"
RESULT_DIR = "/user_data/TabGNN/results/query_aware_v2"

EVAL_PAIRS = [
    ("feta", "feta"),
    ("ottqa", "ottqa"),
    ("mimo_en", "mimo_en"),
    ("mimo_ch", "mimo_ch"),
    ("e2ewtq", "e2ewtq"),
    ("mmqa", "mmqa"),
]


# ===========================
# 指標計算
# ===========================

def full_recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_set = set(retrieved[:k])
    return len(retrieved_set & relevant) / len(relevant)

def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    for i, rid in enumerate(retrieved, 1):
        if rid in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, rid in enumerate(retrieved[:k], 1) if rid in relevant)
    num_rel = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


# ===========================
# 查詢解析
# ===========================

def parse_queries(query_file, mapping_keys, key_fields):
    """解析查詢檔案，回傳 (question, gt_keys) 列表"""
    queries = []
    total = 0
    mappable = 0

    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            obj = json.loads(line)

            question = None
            if 'questions' in obj:
                question = obj['questions'][0].strip()
            elif 'question' in obj:
                question = obj['question'].strip()
            if not question:
                continue

            gt_list = obj.get('ground_truth_list', []) or []
            gt_keys = set()
            for gt in gt_list:
                if all(gt.get(field) is not None for field in key_fields):
                    key = make_key(gt, key_fields)
                    if key in mapping_keys:
                        gt_keys.add(key)

            if gt_keys:
                mappable += 1
            queries.append((question, gt_keys))

    print(f"  載入查詢 {total} 條，可對齊 {mappable} 條")
    return queries


# ===========================
# 評估
# ===========================

def evaluate(source, dataset, coarse_k=COARSE_K, edge_mode=QUERY_EDGE_MODE):
    key_fields = get_key_fields(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph_path = GRAPH_PATH.format(source=source)
    model_path = MODEL_PATH.format(dataset=dataset)
    query_file = QUERY_FILE.format(source=source)

    if not Path(graph_path).exists():
        print(f"  跳過：找不到 {graph_path}")
        return None
    if not Path(model_path).exists():
        print(f"  跳過：找不到 {model_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Query-Aware v2 評估: {source} (model={dataset})")
    print(f"{'='*60}")

    # 載入原始圖（不過濾，用於 TWIG 粗排）
    data_full = torch.load(graph_path, map_location=device, weights_only=False)

    # 建立 idx → id 映射（在過濾前）
    id_to_idx = rebuild_id_to_idx(data_full, key_fields)
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    mapping_keys = set(idx_to_id.values())

    embed_dim = data_full['table'].x.size(1)

    # 過濾後的圖（用於 QA 子圖）
    best_edges = BEST_EDGE_CONFIGS.get(source, [])
    if best_edges:
        data_filtered = filter_edges(data_full, best_edges)
        print(f"  邊過濾(QA子圖用): {best_edges}")
    else:
        data_filtered = data_full

    # 載入 QA 模型（使用 canonical metadata）
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    hps = ckpt.get('hps', {})
    ckpt_edge_mode = ckpt.get('edge_mode', edge_mode)
    metadata = get_canonical_metadata(ckpt_edge_mode)

    model = QueryAwareModel(
        embed_dim=embed_dim,
        hidden_channels=hps.get('HIDDEN_CHANNELS', 768),
        metadata=metadata,
        dropout=hps.get('DROPOUT', 0.1),
        sage_aggr=hps.get('SAGE_AGGR', 'min'),
        hetero_aggr=hps.get('HETERO_AGGR', 'max'),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    print(f"  QA 模型載入完成 (edge_mode={ckpt_edge_mode})")

    # 載入原始 TWIG 模型（用於粗排，在完整圖上運行）
    twig_model_path = TWIG_MODEL_PATH.format(dataset=dataset)
    twig_model = None
    if Path(twig_model_path).exists():
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from train_model import DiffusionModel
        twig_ckpt = torch.load(twig_model_path, map_location=device, weights_only=False)
        twig_hps = twig_ckpt.get('hps', hps)
        # 使用完整圖的 metadata（TWIG 模型在完整圖上訓練）
        twig_metadata = data_full.metadata()
        twig_model = DiffusionModel(
            embed_dim=embed_dim,
            hidden_channels=twig_hps.get('HIDDEN_CHANNELS', 768),
            metadata=twig_metadata,
            dropout=twig_hps.get('DROPOUT', 0.1),
            sage_aggr=twig_hps.get('SAGE_AGGR', 'min'),
            hetero_aggr=twig_hps.get('HETERO_AGGR', 'max'),
        ).to(device)
        twig_model.load_state_dict(twig_ckpt['model_state_dict'], strict=False)
        twig_model.eval()
        print(f"  TWIG 模型載入完成（完整圖，用於粗排）")

    # 解析查詢
    queries = parse_queries(query_file, mapping_keys, key_fields)
    questions = [q for q, _ in queries]
    relevants = [gt for _, gt in queries]
    total = len(queries)
    eval_count = sum(1 for gt in relevants if len(gt) > 0)

    # 嵌入查詢
    embedder = SentenceTransformer(MODEL_NAME, device=str(device))
    query_vecs = embedder.encode(questions, show_progress_bar=True, convert_to_tensor=True).to(device)
    query_vecs = F.normalize(query_vecs, p=2, dim=1)
    del embedder
    torch.cuda.empty_cache()

    # 全圖 forward（粗排用 — 使用 TWIG 模型在完整圖上）
    print("  全圖 GNN forward (粗排)...")
    data_full = data_full.to(device)
    data_filtered = data_filtered.to(device)
    with torch.no_grad():
        if twig_model is not None:
            table_emb_fixed = twig_model.forward(data_full.x_dict, data_full.edge_index_dict)
        else:
            table_emb_fixed = model.forward(data_full.x_dict, data_full.edge_index_dict)
        coarse_scores = torch.matmul(query_vecs, table_emb_fixed.T)

    # 也計算純粗排的 baseline 結果（E0 equivalent）
    print("  計算粗排 baseline...")
    e0_recall1 = e0_recall5 = e0_recall10 = e0_mrr = 0.0
    for qi in range(total):
        gt = relevants[qi]
        if not gt:
            continue
        _, top_indices = torch.topk(coarse_scores[qi], k=min(TOP_K, coarse_scores.size(1)))
        retrieved = [idx_to_id.get(idx.item(), "") for idx in top_indices]
        e0_recall1 += full_recall_at_k(retrieved, gt, 1)
        e0_recall5 += full_recall_at_k(retrieved, gt, 5)
        e0_recall10 += full_recall_at_k(retrieved, gt, 10)
        e0_mrr += reciprocal_rank(retrieved, gt)

    print(f"  粗排 Baseline: R@1={e0_recall1/eval_count:.4f} R@5={e0_recall5/eval_count:.4f} "
          f"R@10={e0_recall10/eval_count:.4f} MRR={e0_mrr/eval_count:.4f}")

    # Query-Aware 重排序（使用分數插值）
    alpha = ALPHA
    print(f"  Query-Aware 重排序 (coarse_k={coarse_k}, alpha={alpha})...")
    recall1 = recall5 = recall10 = 0.0
    mrr = ndcg5 = ndcg10 = prec5 = full_recall5 = 0.0

    with torch.no_grad():
        for qi in tqdm(range(total), desc="  QA Rerank", leave=False):
            gt = relevants[qi]
            if not gt:
                continue

            q_vec = query_vecs[qi:qi + 1]

            # 粗排候選
            top_k_scores, top_k_idx = torch.topk(coarse_scores[qi], k=min(coarse_k, coarse_scores.size(1)))
            candidate_indices = top_k_idx.cpu().tolist()

            # 建子圖（含 query 節點，使用過濾後的圖）
            subgraph, table_mapping = build_subgraph(
                data_filtered, q_vec, candidate_indices,
                edge_mode=ckpt_edge_mode, device=device)

            # 子圖 forward（使用訓練好的 query edge 權重）
            sub_table_emb = model.forward(subgraph.x_dict, subgraph.edge_index_dict)

            # 分數插值：結合粗排分數和重排序分數
            rerank_scores = torch.matmul(q_vec, sub_table_emb.T).squeeze(0)

            # 將粗排分數映射到子圖索引空間
            coarse_in_sub = torch.zeros(rerank_scores.size(0), device=device)
            for orig_rank, orig_idx in enumerate(candidate_indices):
                new_idx = table_mapping.get(orig_idx, -1)
                if new_idx >= 0 and new_idx < coarse_in_sub.size(0):
                    coarse_in_sub[new_idx] = top_k_scores[orig_rank]

            # 插值：alpha * coarse + (1 - alpha) * rerank
            final_scores = alpha * coarse_in_sub + (1 - alpha) * rerank_scores

            _, reranked = torch.topk(final_scores, k=min(TOP_K, final_scores.size(0)))

            # 映射回原索引
            new_to_old = {v: k for k, v in table_mapping.items()}
            retrieved = [idx_to_id.get(new_to_old.get(idx.item(), -1), "") for idx in reranked]

            # 累積指標
            recall1 += full_recall_at_k(retrieved, gt, 1)
            recall5 += full_recall_at_k(retrieved, gt, 5)
            recall10 += full_recall_at_k(retrieved, gt, 10)
            mrr += reciprocal_rank(retrieved, gt)
            ndcg5 += ndcg_at_k(retrieved, gt, 5)
            ndcg10 += ndcg_at_k(retrieved, gt, 10)
            prec5 += precision_at_k(retrieved, gt, 5)
            full_recall5 += full_recall_at_k(retrieved, gt, 5)

    # 計算指標
    results = {
        'eval_count': eval_count,
        'total': total,
        'coarse_k': coarse_k,
        'alpha': alpha,
        'edge_mode': ckpt_edge_mode,
        'Recall@1': recall1 / eval_count,
        'Recall@5': recall5 / eval_count,
        'Recall@10': recall10 / eval_count,
        'MRR@k': mrr / eval_count,
        'nDCG@5': ndcg5 / eval_count,
        'nDCG@10': ndcg10 / eval_count,
        'Precision@5': prec5 / eval_count,
        'Full Recall@5': full_recall5 / eval_count,
        'E0_Recall@1': e0_recall1 / eval_count,
        'E0_Recall@5': e0_recall5 / eval_count,
        'E0_Recall@10': e0_recall10 / eval_count,
        'E0_MRR': e0_mrr / eval_count,
    }

    print(f"\n  === 結果 ===")
    print(f"  E0 Baseline:   R@1={results['E0_Recall@1']:.4f}  R@5={results['E0_Recall@5']:.4f}  "
          f"R@10={results['E0_Recall@10']:.4f}  MRR={results['E0_MRR']:.4f}")
    print(f"  QA Reranked:   R@1={results['Recall@1']:.4f}  R@5={results['Recall@5']:.4f}  "
          f"R@10={results['Recall@10']:.4f}  MRR={results['MRR@k']:.4f}")

    # 計算改善量
    for metric in ['Recall@1', 'Recall@5', 'Recall@10']:
        e0_key = f'E0_{metric}'
        delta = results[metric] - results[e0_key]
        arrow = '↑' if delta > 0 else '↓'
        print(f"  Δ{metric}: {arrow}{abs(delta):.4f}")

    return results


def main():
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)

    pairs = EVAL_PAIRS
    if len(sys.argv) > 1:
        pairs = [(ds, ds) for ds in sys.argv[1:]]

    all_results = {}
    for source, dataset in pairs:
        results = evaluate(source, dataset)
        if results is not None:
            all_results[source] = results

            # 儲存個別結果
            result_file = Path(RESULT_DIR) / f"{source}.json"
            output = {
                'source': source,
                'dataset': dataset,
                'best_edges': BEST_EDGE_CONFIGS.get(source, []),
                'results': results,
            }
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

    # 匯總
    if all_results:
        print(f"\n\n{'='*80}")
        print("匯總結果")
        print(f"{'='*80}")
        print(f"{'Dataset':>10} | {'E0 R@10':>8} | {'QA R@10':>8} | {'ΔR@10':>8} | {'E0 MRR':>8} | {'QA MRR':>8} | {'ΔMRR':>8}")
        print("-" * 80)
        for ds, r in all_results.items():
            dr10 = r['Recall@10'] - r['E0_Recall@10']
            dmrr = r['MRR@k'] - r['E0_MRR']
            a10 = '↑' if dr10 > 0 else '↓'
            amrr = '↑' if dmrr > 0 else '↓'
            print(f"{ds:>10} | {r['E0_Recall@10']:>8.4f} | {r['Recall@10']:>8.4f} | "
                  f"{a10}{abs(dr10):.4f} | {r['E0_MRR']:>8.4f} | {r['MRR@k']:>8.4f} | {amrr}{abs(dmrr):.4f}")

        # 儲存匯總
        summary_file = Path(RESULT_DIR) / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n結果已儲存至 {RESULT_DIR}/")


if __name__ == '__main__':
    main()
