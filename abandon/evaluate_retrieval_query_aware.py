"""
查詢導向之圖式表格檢索評估腳本 (Query-Aware Evaluation)

改進原有 evaluate_retrieval.py：
在上線階段將查詢節點 (query node) 加入動態子圖中，
透過 GNN 前向傳播讓查詢資訊融入表格表示，然後重新排序。

消融實驗支援：
- E0: 不加入 Query 節點（等同原始 baseline）
- E1: Query–Table 連邊
- E2: Query–Table + Query–Page 連邊
- E3: Query–Table + Query–Column 連邊
- E4: Query–Table + Query–Page + Query–Column（Full Query Edges）

子圖大小消融 (k):
- S1: k=5, S2: k=10, S3: k=20, S4: k=50
"""

import json
import math
import copy
from typing import List, Tuple, Dict, Set

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from tqdm import tqdm

from train_model import DiffusionModel, get_embedder
from pathlib import Path

# ========= 可調參數 =========
EVAL_PAIRS = [
    # (source, dataset)  source=評估資料集, dataset=模型來源
    ("mimo_en", "mimo_en"),
    ("mimo_ch", "mimo_ch"),
    ("ottqa", "ottqa"),
    ("feta", "feta"),
    ("e2ewtq", "e2ewtq"),
    ("mmqa", "mmqa"),
]

# Query-Aware 設定
SUBGRAPH_K = 10           # 動態子圖大小（粗排取 top-k 候選）
QUERY_EDGE_MODE = "E4"    # E0/E1/E2/E3/E4
# E0: 不加入 Query 節點
# E1: Query–Table
# E2: Query–Table + Query–Page
# E3: Query–Table + Query–Column
# E4: Query–Table + Query–Page + Query–Column

def get_key_fields(dataset_name: str) -> tuple:
    """根據資料集名稱返回對應的 KEY_FIELDS"""
    if dataset_name in ["ottqa", "feta", "e2ewtq"]:
        return ("sheet_name", "file_name")
    elif dataset_name in ["mimo_en", "mimo_ch", "mmqa"]:
        return ("id",)
    else:
        return ("id",)


QUERY_FILE = "/user_data/TabGNN/data/table/test/{source}/query.jsonl"
MODEL_PATH = "/user_data/TabGNN/checkpoints/{dataset}/model.pt"
GRAPH_PATH = "/user_data/TabGNN/data/processed/test/{source}/graph.pt"
RESULT_DIR = "/user_data/TabGNN/results/evaluate_query_aware"
TOP_K = 10

# ===========================

def make_key(item: dict, key_fields: tuple) -> str:
    """根據 key_fields 組合產生唯一鍵"""
    return "|".join(str(item.get(f, "")) for f in key_fields)


def load_graph_and_model(graph_path: str, model_path: str):
    """載入圖結構和模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的設備: {device}")

    data = torch.load(graph_path, map_location=device, weights_only=False)

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

    # 載入模型和超參數
    checkpoint = torch.load(model_path, map_location=device)
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

    return device, data, idx_to_id, model, hps


def candidates_from_ground_truth(gt: Dict) -> List[str]:
    """從 ground truth 提取表格識別鍵"""
    if all(gt.get(f) is not None for f in KEY_FIELDS):
        return [make_key(gt, KEY_FIELDS)]
    return []


def parse_queries(query_file: str, mapping_keys: Set[str]) -> List[Tuple[str, Set[str]]]:
    """解析查詢檔案"""
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

            gt_list = obj.get("ground_truth_list", []) or []
            gt_keys: Set[str] = set()
            for gt in gt_list:
                for c in candidates_from_ground_truth(gt):
                    if c in mapping_keys:
                        gt_keys.add(c)
                        break

            if gt_keys:
                mappable += 1
            if question:
                queries.append((question, gt_keys))

    print(f"載入查詢 {total_lines} 條，其中可對齊到圖的樣本 {mappable} 條。")
    return queries


def hits_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> int:
    return int(any(rid in relevant_ids for rid in retrieved_ids[:k]))

def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
    num_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_relevant + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg

def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if k == 0: return 0.0
    retrieved_set = set(retrieved_ids[:k])
    intersection = retrieved_set.intersection(relevant_ids)
    return len(intersection) / k

def full_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids: return 0.0
    retrieved_set = set(retrieved_ids[:k])
    intersection = retrieved_set.intersection(relevant_ids)
    return len(intersection) / len(relevant_ids)


def build_query_subgraph(
    data: HeteroData,
    query_vec: torch.Tensor,
    candidate_table_indices: List[int],
    edge_mode: str = "E4",
    device: torch.device = None,
) -> HeteroData:
    """
    建立包含 query 節點的動態子圖。

    步驟：
    1. 從候選 table indices 出發，找出相關的 column 和 page 節點
    2. 收集這些節點間的所有邊
    3. 加入 query 節點及對應連邊
    4. 重新編碼索引（讓子圖節點從 0 開始編號）

    Args:
        data: 完整的 HeteroData 圖
        query_vec: 查詢嵌入向量 (1, embed_dim)
        candidate_table_indices: 候選表格在原圖中的索引列表
        edge_mode: E0/E1/E2/E3/E4
        device: 運算裝置
    """
    if device is None:
        device = query_vec.device

    sub = HeteroData()
    candidate_set = set(candidate_table_indices)
    num_candidates = len(candidate_table_indices)

    # ==========================================
    # 1. 找出子圖中涉及的所有節點
    # ==========================================

    # 表格節點：候選表格
    sub_table_indices = list(candidate_table_indices)
    table_old_to_new = {old: new for new, old in enumerate(sub_table_indices)}

    # 從邊資訊找出相關的 column 和 page 節點
    sub_column_set = set()
    sub_page_set = set()

    # table → has_column → column
    if ('table', 'has_column', 'column') in data.edge_types:
        edge_idx = data['table', 'has_column', 'column'].edge_index
        for i in range(edge_idx.size(1)):
            src = edge_idx[0, i].item()
            dst = edge_idx[1, i].item()
            if src in candidate_set:
                sub_column_set.add(dst)

    # table → comes_from → page
    if ('table', 'comes_from', 'page') in data.edge_types:
        edge_idx = data['table', 'comes_from', 'page'].edge_index
        for i in range(edge_idx.size(1)):
            src = edge_idx[0, i].item()
            dst = edge_idx[1, i].item()
            if src in candidate_set:
                sub_page_set.add(dst)

    sub_column_indices = sorted(sub_column_set)
    sub_page_indices = sorted(sub_page_set)
    column_old_to_new = {old: new for new, old in enumerate(sub_column_indices)}
    page_old_to_new = {old: new for new, old in enumerate(sub_page_indices)}

    # ==========================================
    # 2. 子圖節點特徵
    # ==========================================
    if len(sub_table_indices) > 0:
        sub['table'].x = data['table'].x[sub_table_indices].to(device)
    if len(sub_column_indices) > 0:
        sub['column'].x = data['column'].x[sub_column_indices].to(device)
    if len(sub_page_indices) > 0:
        sub['page'].x = data['page'].x[sub_page_indices].to(device)

    # ==========================================
    # 3. 子圖邊（只保留子圖內的邊）
    # ==========================================
    def filter_and_remap_edges(edge_type, src_map, dst_map, src_set=None, dst_set=None):
        """過濾並重新映射邊索引"""
        if edge_type not in data.edge_types:
            return None
        edge_idx = data[edge_type].edge_index
        new_src, new_dst = [], []
        src_type, rel, dst_type = edge_type

        for i in range(edge_idx.size(1)):
            s = edge_idx[0, i].item()
            d = edge_idx[1, i].item()
            if s in src_map and d in dst_map:
                new_src.append(src_map[s])
                new_dst.append(dst_map[d])

        if new_src:
            sub[edge_type].edge_index = torch.tensor([new_src, new_dst], dtype=torch.long, device=device)
            return True
        return False

    # 結構邊
    filter_and_remap_edges(('table', 'has_column', 'column'), table_old_to_new, column_old_to_new)
    filter_and_remap_edges(('column', 'rev_has_column', 'table'), column_old_to_new, table_old_to_new)
    filter_and_remap_edges(('table', 'comes_from', 'page'), table_old_to_new, page_old_to_new)
    filter_and_remap_edges(('page', 'rev_comes_from', 'table'), page_old_to_new, table_old_to_new)

    # 表格間邊（same_page, similar_table, shared_column_name）
    for rel in ['same_page', 'similar_table', 'shared_column_name']:
        edge_type = ('table', rel, 'table')
        filter_and_remap_edges(edge_type, table_old_to_new, table_old_to_new)

    # 欄位間邊
    filter_and_remap_edges(('column', 'similar_content', 'column'), column_old_to_new, column_old_to_new)

    # ==========================================
    # 4. 加入 Query 節點及連邊
    # ==========================================
    if edge_mode == "E0":
        # 不加入 query 節點，直接返回子圖
        return sub, table_old_to_new

    # 加入 query 節點（只有 1 個）
    sub['query'].x = query_vec.to(device)  # (1, embed_dim)

    query_idx = 0  # query 節點在 query 類型中的索引為 0

    # E1: Query–Table 連邊
    if edge_mode in ["E1", "E2", "E3", "E4"]:
        q_to_t_src = [query_idx] * num_candidates
        q_to_t_dst = list(range(num_candidates))
        sub['query', 'queries', 'table'].edge_index = torch.tensor(
            [q_to_t_src, q_to_t_dst], dtype=torch.long, device=device
        )
        # 反向邊：table → rev_queries → query
        sub['table', 'rev_queries', 'query'].edge_index = torch.tensor(
            [q_to_t_dst, q_to_t_src], dtype=torch.long, device=device
        )

    # E2: + Query–Page 連邊
    if edge_mode in ["E2", "E4"]:
        num_pages = len(sub_page_indices)
        if num_pages > 0:
            q_to_p_src = [query_idx] * num_pages
            q_to_p_dst = list(range(num_pages))
            sub['query', 'queries_page', 'page'].edge_index = torch.tensor(
                [q_to_p_src, q_to_p_dst], dtype=torch.long, device=device
            )
            sub['page', 'rev_queries_page', 'query'].edge_index = torch.tensor(
                [q_to_p_dst, q_to_p_src], dtype=torch.long, device=device
            )

    # E3: + Query–Column 連邊
    if edge_mode in ["E3", "E4"]:
        num_cols = len(sub_column_indices)
        if num_cols > 0:
            q_to_c_src = [query_idx] * num_cols
            q_to_c_dst = list(range(num_cols))
            sub['query', 'queries_column', 'column'].edge_index = torch.tensor(
                [q_to_c_src, q_to_c_dst], dtype=torch.long, device=device
            )
            sub['column', 'rev_queries_column', 'query'].edge_index = torch.tensor(
                [q_to_c_dst, q_to_c_src], dtype=torch.long, device=device
            )

    return sub, table_old_to_new


def build_query_aware_model(base_model: DiffusionModel, subgraph: HeteroData, hps: dict, device: torch.device):
    """
    根據子圖的 metadata 建立一個新的 query-aware 模型。
    
    因為子圖引入了新的節點類型 (query) 和新的邊類型，
    需要重新建立 to_hetero 映射。但為了保留離線訓練好的權重，
    我們使用 strict=False 載入原模型權重。
    """
    from torch_geometric.nn import GraphSAGE, to_hetero, GraphNorm
    from torch import nn

    embed_dim = base_model.proj_head[-1].out_features
    hidden_channels = hps.get('HIDDEN_CHANNELS', 128)
    dropout = hps.get('DROPOUT', 0.2)
    sage_aggr = hps.get('SAGE_AGGR', 'sum')
    hetero_aggr = hps.get('HETERO_AGGR', 'sum')

    new_model = DiffusionModel(
        embed_dim=embed_dim,
        hidden_channels=hidden_channels,
        metadata=subgraph.metadata(),
        dropout=dropout,
        sage_aggr=sage_aggr,
        hetero_aggr=hetero_aggr,
    ).to(device)

    # 載入原模型的權重（strict=False 允許缺失的 key）
    old_state = base_model.state_dict()
    new_model.load_state_dict(old_state, strict=False)
    new_model.eval()

    return new_model


def evaluate(
    query_file: str = None,
    model_path: str = None,
    graph_path: str = None,
    top_k: int = TOP_K,
    subgraph_k: int = SUBGRAPH_K,
    edge_mode: str = QUERY_EDGE_MODE,
):
    """
    執行 Query-Aware GNN 評估
    
    流程：
    1. 用固定 table embedding 進行粗排，取 top-subgraph_k 候選
    2. 為每個查詢建立動態子圖，加入 query 節點
    3. 在子圖上重新 GNN forward
    4. 用更新後的 table embedding 重新排序
    """
    device, data, idx_to_id, model, hps = load_graph_and_model(graph_path, model_path)
    mapping_keys = set(idx_to_id.values())
    embedder = get_embedder(device=('cuda' if torch.cuda.is_available() else 'cpu'))

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

    print(f"\n{'='*60}")
    print(f"Query-Aware 評估設定:")
    print(f"  Edge Mode: {edge_mode}")
    print(f"  Subgraph K: {subgraph_k}")
    print(f"{'='*60}")

    print("嵌入所有查詢向量...")
    query_vecs = embedder.encode(questions, show_progress_bar=True, convert_to_tensor=True).to(device)
    query_vecs = F.normalize(query_vecs, p=2, dim=1)

    print("計算 GNN 表格嵌入（用於粗排）...")
    with torch.no_grad():
        data_on_device = data.to(device)

        # 首先用原始模型得到固定 table embedding（用於粗排）
        if model is None:
            table_emb_fixed = F.normalize(data_on_device['table'].x, p=2, dim=1)
        else:
            table_emb_fixed = model.forward(data_on_device.x_dict, data_on_device.edge_index_dict)

    if edge_mode == "E0":
        # E0 模式：不加入 query 節點，等同原始 baseline
        print("E0 模式：純固定表示，不使用 query 節點")
        print("計算相似度並評估...")
        with torch.no_grad():
            chunk_size = 256
            for start in tqdm(range(0, total, chunk_size), desc="評估中"):
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
        # Query-Aware 模式：加入 query 節點進行子圖推論
        print(f"Query-Aware 模式 ({edge_mode})：建立動態子圖並加入 query 節點")
        print("逐查詢建立子圖並推論...")

        # 預先計算粗排分數
        with torch.no_grad():
            coarse_scores = torch.matmul(query_vecs, table_emb_fixed.T)

        for q_idx in tqdm(range(total), desc="Query-Aware 評估"):
            relevant_ids = relevants[q_idx]
            if not relevant_ids:
                continue

            q_vec = query_vecs[q_idx:q_idx+1]  # (1, embed_dim)

            # Step 1: 粗排取 top-subgraph_k 候選
            _, coarse_top_indices = torch.topk(
                coarse_scores[q_idx],
                k=min(subgraph_k, coarse_scores.size(1))
            )
            candidate_indices = coarse_top_indices.cpu().tolist()

            # Step 2: 建立動態子圖
            with torch.no_grad():
                subgraph, table_mapping = build_query_subgraph(
                    data_on_device, q_vec, candidate_indices,
                    edge_mode=edge_mode, device=device
                )

                # Step 3: 建立 query-aware 模型並推論
                qa_model = build_query_aware_model(model, subgraph, hps, device)

                # 在子圖上 forward
                x_dict_out = qa_model.hetero_sage(subgraph.x_dict, subgraph.edge_index_dict)

                # 取得更新後的 table embedding
                if 'table' in x_dict_out:
                    x_table = x_dict_out['table']
                    x_table = qa_model.norm(x_table)
                    updated_table_emb = qa_model.proj_head(x_table)
                    updated_table_emb = F.normalize(updated_table_emb, p=2, dim=1)
                else:
                    # 如果子圖沒有 table 節點（理論上不會發生），fallback
                    updated_table_emb = table_emb_fixed[candidate_indices]

                # Step 4: 用更新後的 embedding 重新排序
                rerank_scores = torch.matmul(q_vec, updated_table_emb.T).squeeze(0)
                _, reranked_indices = torch.topk(rerank_scores, k=min(top_k, rerank_scores.size(0)))

                # 將子圖中的索引映射回原圖索引
                new_to_old = {v: k for k, v in table_mapping.items()}
                retrieved_orig_indices = [new_to_old[idx.item()] for idx in reranked_indices]
                retrieved_ids = [idx_to_id.get(orig_idx, "") for orig_idx in retrieved_orig_indices]

            # 計算指標
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

    print("\n===== 評估結果 =====")
    if eval_count == 0:
        print("無可對齊的 ground truth，無法計算指標。")
        return {
            "eval_count": 0, "total": total,
            "edge_mode": edge_mode, "subgraph_k": subgraph_k,
            "Recall@1": None, "Recall@5": None, "Recall@10": None,
            "MRR@k": None, "nDCG@5": None, "nDCG@10": None,
            "Precision@5": None, "Full Recall@5": None,
        }

    results = {
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

    print(f"Edge Mode: {edge_mode} | Subgraph K: {subgraph_k}")
    for k_name, v in results.items():
        if k_name in ("eval_count", "total", "edge_mode", "subgraph_k"):
            continue
        print(f"{k_name}：{v:.4f}")
    print("====================")

    return results


def main():
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)

    for source, dataset in EVAL_PAIRS:
        global KEY_FIELDS
        KEY_FIELDS = get_key_fields(dataset)

        query_file = QUERY_FILE.format(source=source)
        model_path = MODEL_PATH.format(dataset=dataset)
        graph_path = GRAPH_PATH.format(source=source)

        if not Path(graph_path).exists():
            print(f"跳過 ({source}, {dataset})：找不到 {graph_path}")
            continue
        if not Path(model_path).exists():
            print(f"跳過 ({source}, {dataset})：找不到 {model_path}")
            continue

        print(f"\n{'='*60}")
        print(f"評估 SOURCE={source}  MODEL={dataset}")
        print(f"KEY_FIELDS: {KEY_FIELDS}")
        print(f"Edge Mode: {QUERY_EDGE_MODE}")
        print(f"Subgraph K: {SUBGRAPH_K}")
        print(f"{'='*60}")

        results = evaluate(
            query_file=query_file,
            model_path=model_path,
            graph_path=graph_path,
            top_k=TOP_K,
            subgraph_k=SUBGRAPH_K,
            edge_mode=QUERY_EDGE_MODE,
        )

        # 儲存結果
        result_file = Path(RESULT_DIR) / f"{source}_{QUERY_EDGE_MODE}_k{SUBGRAPH_K}.json"
        output = {
            "source": source,
            "dataset": dataset,
            "edge_mode": QUERY_EDGE_MODE,
            "subgraph_k": SUBGRAPH_K,
            "results": results,
        }
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"結果已儲存至 {result_file}")


if __name__ == "__main__":
    main()
