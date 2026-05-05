import argparse
import json
import os
import sys
import gc
import time
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_model import get_embedder
from run_edge_ablation import filter_edges, build_id_to_idx
from query_aware.evaluate_retrieval_query_aware import (
    EVAL_PAIRS, 
    get_key_fields, 
    parse_queries, 
    load_graph_and_model,
    TOP_K,
    build_query_subgraph,
    build_query_aware_model,
    full_recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    precision_at_k
)
import query_aware.evaluate_retrieval_query_aware as eval_module

# 定義每個資料集的最佳基礎圖連邊配置
BEST_EDGE_CONFIGS = {
    "feta": ["has_column", "same_page"],
    "ottqa": ["has_column", "similar_content"],
    "mimo_en": ["has_column", "similar_content", "shared_column_name"],
    "mimo_ch": ["similar_content"],
    "e2ewtq": ["similar_content"],
    "mmqa": ["similar_table", "has_column", "comes_from", "same_page", "similar_content"]
}

QUERY_FILE = "/user_data/TabGNN/data/table/test/{source}/query.jsonl"
MODEL_PATH = "/user_data/TabGNN/checkpoints/{dataset}/model.pt"
GRAPH_PATH = "/user_data/TabGNN/data/processed/test/{source}/graph.pt"
RESULT_DIR = "/user_data/TabGNN/results/evaluate_query_aware_final"

def evaluate_with_preloaded(
    query_file: str,
    device: torch.device,
    data: torch.Tensor,
    idx_to_id: dict,
    model: torch.nn.Module,
    hps: dict,
    embedder,
    top_k: int,
    subgraph_k: int,
    edge_mode: str,
):
    """
    修改自 evaluate_retrieval_query_aware.py 內的 evaluate 函數。
    接收已經預先載入好且過濾好連邊的圖資料和模型進行評估。
    """
    mapping_keys = set(idx_to_id.values())

    queries = parse_queries(query_file, mapping_keys)
    questions = [q for q, _ in queries]
    relevants = [gt for _, gt in queries]

    total = len(queries)
    eval_count = sum(1 for gt in relevants if len(gt) > 0)

    # 初始化指標
    recall1 = recall5 = recall10 = 0.0
    exact_match5 = 0.0
    mrr = 0.0
    ndcg5 = ndcg10 = 0.0
    precision5 = 0.0

    query_vecs = embedder.encode(questions, show_progress_bar=False, convert_to_tensor=True).to(device)
    query_vecs = F.normalize(query_vecs, p=2, dim=1)

    with torch.no_grad():
        data_on_device = data.to(device)

        if model is None:
            table_emb_fixed = F.normalize(data_on_device['table'].x, p=2, dim=1)
        else:
            table_emb_fixed = model.forward(data_on_device.x_dict, data_on_device.edge_index_dict)

    if edge_mode == "E0":
        with torch.no_grad():
            chunk_size = 256
            for start in tqdm(range(0, total, chunk_size), desc="評估中", leave=False):
                end = min(start + chunk_size, total)
                q_chunk = query_vecs[start:end]
                scores = torch.matmul(q_chunk, table_emb_fixed.T)

                for i in range(end - start):
                    q_idx = start + i
                    relevant_ids = relevants[q_idx]
                    if not relevant_ids:
                        continue

                    _, top_indices = torch.topk(scores[i], k=min(top_k, scores.size(1)))
                    retrieved_ids = [idx_to_id.get(idx.item(), "") for idx in top_indices]

                    recall1 += full_recall_at_k(retrieved_ids, relevant_ids, 1)
                    recall5 += full_recall_at_k(retrieved_ids, relevant_ids, 5)
                    recall10 += full_recall_at_k(retrieved_ids, relevant_ids, 10)

                    relevant_set = set(relevant_ids)
                    retrieved_set_5 = set(retrieved_ids[:5])
                    if relevant_set.issubset(retrieved_set_5):
                        exact_match5 += 1.0

                    mrr += reciprocal_rank(retrieved_ids, relevant_ids)
                    ndcg5 += ndcg_at_k(retrieved_ids, relevant_ids, 5)
                    ndcg10 += ndcg_at_k(retrieved_ids, relevant_ids, 10)
                    precision5 += precision_at_k(retrieved_ids, relevant_ids, 5)
    else:
        with torch.no_grad():
            coarse_scores = torch.matmul(query_vecs, table_emb_fixed.T)

        for q_idx in tqdm(range(total), desc="Query-Aware 評估", leave=False):
            relevant_ids = relevants[q_idx]
            if not relevant_ids:
                continue

            q_vec = query_vecs[q_idx:q_idx+1]

            _, coarse_top_indices = torch.topk(
                coarse_scores[q_idx],
                k=min(subgraph_k, coarse_scores.size(1))
            )
            candidate_indices = coarse_top_indices.cpu().tolist()

            with torch.no_grad():
                subgraph, table_mapping = build_query_subgraph(
                    data_on_device, q_vec, candidate_indices,
                    edge_mode=edge_mode, device=device
                )

                qa_model = build_query_aware_model(model, subgraph, hps, device)
                x_dict_out = qa_model.hetero_sage(subgraph.x_dict, subgraph.edge_index_dict)

                if 'table' in x_dict_out:
                    x_table = x_dict_out['table']
                    x_table = qa_model.norm(x_table)
                    updated_table_emb = qa_model.proj_head(x_table)
                    updated_table_emb = F.normalize(updated_table_emb, p=2, dim=1)
                else:
                    updated_table_emb = table_emb_fixed[candidate_indices]

                rerank_scores = torch.matmul(q_vec, updated_table_emb.T).squeeze(0)
                _, reranked_indices = torch.topk(rerank_scores, k=min(top_k, rerank_scores.size(0)))

                new_to_old = {v: k for k, v in table_mapping.items()}
                retrieved_orig_indices = [new_to_old[idx.item()] for idx in reranked_indices]
                retrieved_ids = [idx_to_id.get(orig_idx, "") for orig_idx in retrieved_orig_indices]

            recall1 += full_recall_at_k(retrieved_ids, relevant_ids, 1)
            recall5 += full_recall_at_k(retrieved_ids, relevant_ids, 5)
            recall10 += full_recall_at_k(retrieved_ids, relevant_ids, 10)

            relevant_set = set(relevant_ids)
            retrieved_set_5 = set(retrieved_ids[:5])
            if relevant_set.issubset(retrieved_set_5):
                exact_match5 += 1.0

            mrr += reciprocal_rank(retrieved_ids, relevant_ids)
            ndcg5 += ndcg_at_k(retrieved_ids, relevant_ids, 5)
            ndcg10 += ndcg_at_k(retrieved_ids, relevant_ids, 10)
            precision5 += precision_at_k(retrieved_ids, relevant_ids, 5)

    if eval_count == 0:
        return {
            "eval_count": 0, "total": total,
            "edge_mode": edge_mode, "subgraph_k": subgraph_k,
            "Recall@1": None, "Recall@5": None, "Recall@10": None,
            "MRR@k": None, "nDCG@5": None, "nDCG@10": None,
            "Precision@5": None, "Full Recall@5": None,
        }

    return {
        "eval_count": eval_count,
        "total": total,
        "edge_mode": edge_mode,
        "subgraph_k": subgraph_k,
        "Recall@1": recall1 / eval_count,
        "Recall@5": recall5 / eval_count,
        "Recall@10": recall10 / eval_count,
        "MRR@k": mrr / eval_count,
        "nDCG@5": ndcg5 / eval_count,
        "nDCG@10": ndcg10 / eval_count,
        "Precision@5": precision5 / eval_count,
        "Full Recall@5": exact_match5 / eval_count,
    }

def main():
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedder = get_embedder(device=('cuda' if torch.cuda.is_available() else 'cpu'))
    
    edge_modes = ["E0", "E1", "E2", "E3", "E4"]
    subgraph_ks = [5, 10, 20, 50]

    for source, dataset in EVAL_PAIRS:
        KEY_FIELDS = get_key_fields(dataset)
        eval_module.KEY_FIELDS = KEY_FIELDS
        
        query_file = QUERY_FILE.format(source=source)
        model_path = MODEL_PATH.format(dataset=dataset)
        graph_path = GRAPH_PATH.format(source=source)

        if not Path(graph_path).exists() or not Path(model_path).exists():
            print(f"Skipping ({source}, {dataset}): Files not found.")
            continue

        print(f"\n============================================================")
        print(f"Processing Dataset: {source.upper()}")
        print(f"============================================================")

        # 1. 載入完整的圖與模型
        _, full_data, idx_to_id, model, hps = load_graph_and_model(graph_path, model_path)
        
        # 2. 獲取該資料集的最佳連邊配置，並過濾圖
        best_edges = BEST_EDGE_CONFIGS.get(source, [])
        if not best_edges:
            print(f"警告：未找到 {source} 的最佳連邊配置，將使用完整圖！")
            filtered_data = full_data
        else:
            print(f"Filtering base graph edges to: {best_edges}")
            filtered_data = filter_edges(full_data, best_edges)

        # Fix dimension error by ensuring all expected edge types exist as empty tensors
        expected_edge_types = full_data.metadata()[1]
        for et in expected_edge_types:
            if et not in filtered_data.edge_types:
                filtered_data[et].edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # 清理記憶體
        del full_data
        gc.collect()

        # 3. 針對不同的 Query-Aware 配置進行測試
        dataset_results = []
        
        for k in subgraph_ks:
            for emode in edge_modes:
                # E0 模式 (Baseline) 不依賴 subgraph_k，因此 k=20 跑 E0 是重複的，我們只在 k=10 跑一次 E0 即可。
                if emode == "E0" and k != 10:
                    continue

                print(f"  --> Running edge_mode={emode}, subgraph_k={k}")
                t0 = time.time()
                
                results = evaluate_with_preloaded(
                    query_file=query_file,
                    device=device,
                    data=filtered_data,
                    idx_to_id=idx_to_id,
                    model=model,
                    hps=hps,
                    embedder=embedder,
                    top_k=TOP_K,
                    subgraph_k=k,
                    edge_mode=emode,
                )
                
                dt = time.time() - t0
                print(f"      R@10: {results.get('Recall@10', 0):.4f} | MRR: {results.get('MRR@k', 0):.4f} ({dt:.1f}s)")

                # 寫入 JSON (為每個配置儲存一個獨立檔案)
                result_file = Path(RESULT_DIR) / f"{source}_{emode}_k{k}.json"
                output = {
                    "source": source,
                    "dataset": dataset,
                    "base_edges": best_edges,
                    "edge_mode": emode,
                    "subgraph_k": k,
                    "results": results,
                }
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)

        # 清除資料集相關的模型與資料
        del filtered_data, model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
