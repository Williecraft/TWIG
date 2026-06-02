"""評估 GNN 表格檢索模型的 Recall 和 MRR 指標"""
import json
from typing import List, Tuple, Dict, Set

import torch
import torch.nn.functional as F
from tqdm import tqdm

from train_model import DiffusionModel, get_embedder
from pathlib import Path

# ========= 可調參數 =========
EVAL_PAIRS = [
    # (source, dataset)  source=評估資料集, dataset=模型來源
    ("feta", "feta"),
    ("ottqa", "ottqa"),
    ("mimo_en", "mimo_en"),
    ("mimo_ch", "mimo_ch"),
    ("e2ewtq", "e2ewtq"),
    ("mmqa", "mmqa"),
]


def get_key_fields(dataset_name: str) -> tuple:
    """根據資料集名稱返回對應的 KEY_FIELDS"""
    if dataset_name in ["ottqa", "feta", "e2ewtq"]:
        return ("sheet_name", "file_name")
    elif dataset_name in ["mimo_en", "mimo_ch", "mmqa"]:
        return ("id",)
    else:
        return ("id",)


QUERY_FILE = "/user_data/TabGNN/data/table/test/{source}/query.jsonl"
MODEL_PATH = "/user_data/TabGNN/checkpoints/{dataset}/model_best_edges.pt"
GRAPH_PATH = "/user_data/TabGNN/data/processed/test/{source}/graph.pt"
RESULT_DIR = "/user_data/TabGNN/results/evaluate"
TOP_K = 50

# ===========================

def make_key(item: dict, key_fields: tuple) -> str:
    """根據 key_fields 組合產生唯一鍵"""
    return "|".join(str(item.get(f, "")) for f in key_fields)

def filter_graph_edges(data, keep_relations: list):
    """過濾圖中的邊，只保留訓練時使用的邊類型"""
    from torch_geometric.data import HeteroData
    filtered = HeteroData()
    for node_type in data.node_types:
        for attr_name in data[node_type].keys():
            filtered[node_type][attr_name] = data[node_type][attr_name]

    forward_to_reverse = {
        'has_column': 'rev_has_column',
        'comes_from': 'rev_comes_from',
        'similar_table': 'similar_table',
        'same_page': 'same_page',
        'similar_content': 'similar_content',
        'shared_column_name': 'shared_column_name',
    }
    edges_to_keep = set()
    for rel in keep_relations:
        edges_to_keep.add(rel)
        if rel in forward_to_reverse:
            edges_to_keep.add(forward_to_reverse[rel])

    reverse_to_forward = {'rev_has_column': 'has_column', 'rev_comes_from': 'comes_from'}
    for edge_type, edge_index in data.edge_index_dict.items():
        _, relation, _ = edge_type
        fwd = reverse_to_forward.get(relation, relation)
        if relation in edges_to_keep or fwd in keep_relations:
            filtered[edge_type].edge_index = edge_index

    # 確保每個 node type 都有邊可以接收訊息，否則 to_hetero 會報錯
    dest_types = {dst for _, _, dst in filtered.edge_types} if filtered.edge_types else set()
    for node_type in filtered.node_types:
        if node_type not in dest_types:
            filtered[node_type, f'_self_loop_{node_type}', node_type].edge_index = \
                torch.tensor([[0], [0]], dtype=torch.long)

    if hasattr(data, 'metadata_maps'):
        filtered.metadata_maps = data.metadata_maps
    return filtered


def load_graph_and_model(graph_path: str, model_path: str):
    """載入圖結構和模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的設備: {device}")

    # 先載入 checkpoint 以取得 best_edges
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    best_edges = checkpoint.get('best_edges', None)

    data = torch.load(graph_path, map_location=device, weights_only=False)

    # 若 checkpoint 有 best_edges，過濾 test graph 以匹配訓練時的邊配置
    if best_edges:
        print(f"  過濾邊: {best_edges}")
        data = filter_graph_edges(data, best_edges)

    # 取得映射表
    if hasattr(data, 'metadata_maps') and 'table_meta' in data.metadata_maps:
        print(f"根據 KEY_FIELDS={KEY_FIELDS} 重建 table_id_to_idx...")
        table_meta = data.metadata_maps['table_meta']
        table_id_to_idx = {}
        for idx, meta in enumerate(table_meta):
            key = make_key(meta, KEY_FIELDS)
            if key not in table_id_to_idx:
                table_id_to_idx[key] = idx
    else:
        try:
            table_id_to_idx = data.metadata_maps['table_id_to_idx']
        except Exception:
            try:
                table_ids = getattr(data['table'], 'id')
                table_id_to_idx = {tid: idx for idx, tid in enumerate(table_ids)}
            except Exception:
                table_id_to_idx = {idx: idx for idx in range(data['table'].x.size(0))}

    embed_dim = data['table'].x.size(1)
    hps = checkpoint.get('hps', {'HIDDEN_CHANNELS': 128, 'DROPOUT': 0.2, 'AGGR': 'sum'})

    model = DiffusionModel(
        embed_dim=embed_dim,
        hidden_channels=hps.get('HIDDEN_CHANNELS', 128),
        metadata=data.metadata(),
        dropout=hps.get('DROPOUT', 0.2),
        sage_aggr=hps.get('SAGE_AGGR', 'sum'),
        hetero_aggr=hps.get('HETERO_AGGR', 'sum'),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    idx_to_id = {v: str(k) for k, v in table_id_to_idx.items()}

    return device, data, idx_to_id, model


def candidates_from_ground_truth(gt: Dict) -> List[str]:
    """從 ground truth 提取表格識別鍵 (根據 KEY_FIELDS)"""
    # 檢查所有 key fields 是否存在（允許空字串，但不允許 None）
    if all(gt.get(f) is not None for f in KEY_FIELDS):
        return [make_key(gt, KEY_FIELDS)]
    return []


def parse_queries(query_file: str, mapping_keys: Set[str]) -> List[Tuple[str, Set[str]]]:
    """解析查詢檔案，返回 (問題, 正確答案ID集合) 列表，支援多表格 ground_truth_list"""
    queries = []
    total_lines = 0
    mappable = 0

    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            obj = json.loads(line)

            if "questions" in obj:
                question = obj.get("questions")[0].strip()
            elif "question" in obj:
                question = obj.get("question").strip()
            else:
                continue

            # 從 ground_truth_list 提取所有正確表格 ID
            gt_list = obj.get("ground_truth_list", []) or []
            gt_keys: Set[str] = set()
            for gt in gt_list:
                for c in candidates_from_ground_truth(gt):
                    if c in mapping_keys:
                        gt_keys.add(c)
                        break  # 每個 gt 只取第一個匹配的候選

            if gt_keys:
                mappable += 1
            if question:
                queries.append((question, gt_keys))

    print(f"載入查詢 {total_lines} 條，其中可對齊到圖的樣本 {mappable} 條。")
    return queries


def hits_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> int:
    """檢查前 k 個檢索結果中是否有任一正確表格"""
    return int(any(rid in relevant_ids for rid in retrieved_ids[:k]))


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """計算 MRR，找出所有正確表格中最早出現的排名"""
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


import math

def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """計算 nDCG@k，支援多表格 ground truth
    
    對於每個檢索結果，如果是正確表格則 relevance=1，否則=0
    DCG = sum(rel_i / log2(i+1)) for i in 1..k
    IDCG = 理想情況下所有正確表格都排在前面
    """
    # 計算 DCG
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
    
    # 計算 IDCG（理想情況：所有正確表格都排在最前面）
    num_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_relevant + 1))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """計算 Precision@k"""
    if k == 0: return 0.0
    retrieved_set = set(retrieved_ids[:k])
    intersection = retrieved_set.intersection(relevant_ids)
    return len(intersection) / k


def full_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """計算真正的 Recall@k (找回多少比例的相關文檔)"""
    if not relevant_ids: return 0.0
    retrieved_set = set(retrieved_ids[:k])
    intersection = retrieved_set.intersection(relevant_ids)
    return len(intersection) / len(relevant_ids)


def evaluate(
    query_file: str = None,
    model_path: str = None,
    graph_path: str = None,
    top_k: int = TOP_K
):
    """執行 GNN 評估"""
    device, data, idx_to_id, model = load_graph_and_model(graph_path, model_path)
    mapping_keys = set(idx_to_id.values())
    embedder = get_embedder(device=('cuda' if torch.cuda.is_available() else 'cpu'))

    queries = parse_queries(query_file, mapping_keys)
    questions = [q for q, _ in queries]
    relevants = [gt for _, gt in queries]

    total = len(queries)
    eval_count = sum(1 for gt in relevants if len(gt) > 0)

    # 初始化指標
    recall1 = recall5 = recall10 = recall50 = 0.0  # Standard Recall
    exact_match5 = 0.0                  # Real Full Recall (全對才算)

    mrr = 0.0
    ndcg5 = ndcg10 = 0.0
    precision5 = 0.0

    print("嵌入所有查詢向量...")
    query_vecs = embedder.encode(questions, show_progress_bar=True, convert_to_tensor=True).to(device)
    query_vecs = F.normalize(query_vecs, p=2, dim=1)

    print("計算 GNN 表格嵌入...")
    with torch.no_grad():
        data_on_device = data.to(device)
        
        if model is None:
            # G0: 純嵌入，不使用 GNN
            print("  使用純嵌入評估 (No GNN)...")
            table_emb = F.normalize(data_on_device['table'].x, p=2, dim=1)
        else:
            table_emb = model.forward(data_on_device.x_dict, data_on_device.edge_index_dict)

    print("計算相似度並評估...")
    with torch.no_grad():
        chunk_size = 256
        for start in tqdm(range(0, total, chunk_size), desc="評估中"):
            end = min(start + chunk_size, total)
            q_chunk = query_vecs[start:end]
            scores = torch.matmul(q_chunk, table_emb.T)

            for i in range(end - start):
                q_idx = start + i
                relevant_ids = relevants[q_idx]

                if not relevant_ids:
                    continue

                _, top_indices = torch.topk(scores[i], k=min(top_k, scores.size(1)))
                retrieved_ids = [idx_to_id.get(idx.item(), "") for idx in top_indices]

                # 1. Standard Recall (找回比例) - 這是學術界的 Recall
                recall1 += full_recall_at_k(retrieved_ids, relevant_ids, 1)
                recall5 += full_recall_at_k(retrieved_ids, relevant_ids, 5)
                recall10 += full_recall_at_k(retrieved_ids, relevant_ids, 10)
                recall50 += full_recall_at_k(retrieved_ids, relevant_ids, 50)

                # 2. Exact Match (真正的 Full Recall) - 必須 100% 找齊才給分
                relevant_set = set(relevant_ids)
                retrieved_set_5 = set(retrieved_ids[:5])
                if relevant_set.issubset(retrieved_set_5):
                    exact_match5 += 1.0

                mrr += reciprocal_rank(retrieved_ids, relevant_ids)
                ndcg5 += ndcg_at_k(retrieved_ids, relevant_ids, 5)
                ndcg10 += ndcg_at_k(retrieved_ids, relevant_ids, 10)
                precision5 += precision_at_k(retrieved_ids, relevant_ids, 5)

    print("\n===== 評估結果 =====")
    if eval_count == 0:
        print("無可對齊的 ground truth，無法計算指標。")
        return {
            "eval_count": 0,
            "total": total,
            "Recall@1": None,
            "Recall@5": None,
            "Recall@10": None,
            "Recall@50": None,
            "MRR@k": None,
            "nDCG@5": None,
            "nDCG@10": None,
            "Precision@5": None,
            "Full Recall@5": None,
        }

    results = {
        "eval_count": eval_count,
        "total": total,
        "Recall@1": recall1 / eval_count,
        "Recall@5": recall5 / eval_count,
        "Recall@10": recall10 / eval_count,
        "Recall@50": recall50 / eval_count,
        "MRR@k": mrr / eval_count,
        "nDCG@5": ndcg5 / eval_count,
        "nDCG@10": ndcg10 / eval_count,
        "Precision@5": precision5 / eval_count,
        "Full Recall@5": exact_match5 / eval_count,
    }

    for k, v in results.items():
        if k in ("eval_count", "total"):
            continue
        print(f"{k}：{v:.4f}")
    print("====================")

    return results



def main():
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)

    for source, dataset in EVAL_PAIRS:
        # 根據資料集設定 KEY_FIELDS
        global KEY_FIELDS
        KEY_FIELDS = get_key_fields(dataset)
        
        query_file = QUERY_FILE.format(source=source)
        model_path = MODEL_PATH.format(dataset=dataset)
        graph_path = GRAPH_PATH.format(source=source)

        # 檢查檔案是否存在
        if not Path(graph_path).exists():
            print(f"跳過 ({source}, {dataset})：找不到 {graph_path}")
            continue
        if not Path(model_path).exists():
            print(f"跳過 ({source}, {dataset})：找不到 {model_path}")
            continue

        print(f"\n{'='*60}")
        print(f"評估 SOURCE={source}  MODEL={dataset}")
        print(f"KEY_FIELDS: {KEY_FIELDS}")
        print(f"{'='*60}")

        results = evaluate(
            query_file=query_file,
            model_path=model_path,
            graph_path=graph_path,
            top_k=TOP_K
        )

        # 儲存結果
        result_file = Path(RESULT_DIR) / f"{source}.json"
        output = {
            "source": source,
            "dataset": dataset,
            "results": results,
        }
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"結果已儲存至 {result_file}")


if __name__ == "__main__":
    main()
