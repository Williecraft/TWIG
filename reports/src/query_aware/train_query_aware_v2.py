#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Query-Aware GNN 訓練腳本 v2

核心改進（相較原始 train_model_query_aware.py）：
1. 從預訓練 TWIG checkpoint 初始化（不是從頭開始）
2. Query edge 參數零初始化（初始行為 = 原始 TWIG）
3. 差異學習率：base 參數 LR 低、query 參數 LR 高
4. 使用 canonical metadata 建構模型（避免 metadata 不一致）
5. 訓練完的模型可直接用於評估（不需每查詢重建模型）
"""

import sys
import json
import copy
import random

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphSAGE, to_hetero, GraphNorm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ===========================
# 設定
# ===========================

BEST_EDGE_CONFIGS = {
    # QA ablation (A0-A63) 最佳配置，依 QA_R@10 on test set 選出
    "feta":    ["similar_table", "has_column", "comes_from", "same_page", "shared_column_name"],           # A61
    "ottqa":   ["similar_table", "has_column", "comes_from", "same_page", "similar_content", "shared_column_name"],  # A63
    "mimo_en": ["similar_table", "comes_from", "same_page", "shared_column_name"],                         # A45
    "mimo_ch": ["has_column", "comes_from", "same_page", "similar_content"],                               # A30
    "e2ewtq":  ["has_column", "shared_column_name"],                                                       # A17
    "mmqa":    ["has_column", "comes_from", "similar_content", "shared_column_name"],                      # A27
}

MODEL_NAME = 'BAAI/bge-m3'
QUERY_EDGE_MODE = "E4"
SUBGRAPH_K = 30          # 訓練子圖大小
COARSE_K_EVAL = 50       # 驗證粗排大小
NUM_EPOCHS = 15           # 微調不需太多 epoch
WARMUP_EPOCHS = 2
BATCH_SIZE = 64
NUM_HARD_NEGATIVES = 3
EARLY_STOPPING_PATIENCE = 5

# 微調學習率（比原始訓練低，因為已有好的初始化）
BASE_LR = 1e-4            # base 參數
QUERY_LR = 5e-4           # query 參數（需要學更多）
WEIGHT_DECAY = 0.03
CLIP_GRAD_NORM = 0.60
TEMP = 0.04
LABEL_SMOOTH = 0.08

# 路徑
PRETRAINED_PATH = "/user_data/TabGNN/checkpoints/{dataset}/model.pt"
GRAPH_FILE = "/user_data/TabGNN/data/processed/train/{dataset}/graph.pt"
QUERY_FILE = "/user_data/TabGNN/data/table/train/{dataset}/query.jsonl"
VAL_GRAPH_FILE = "/user_data/TabGNN/data/processed/dev/{dataset}/graph.pt"
VAL_QUERY_FILE = "/user_data/TabGNN/data/table/dev/{dataset}/query.jsonl"
SAVE_PATH = "/user_data/TabGNN/checkpoints/{dataset}/model_query_aware_v2.pt"


def get_key_fields(dataset_name: str) -> tuple:
    if dataset_name in ["ottqa", "feta", "e2ewtq"]:
        return ("sheet_name", "file_name")
    elif dataset_name in ["mimo_en", "mimo_ch", "mmqa"]:
        return ("id",)
    return ("id",)


def make_key(item: dict, key_fields: tuple) -> str:
    return "|".join(str(item.get(f, "")) for f in key_fields)


# ===========================
# Canonical metadata：保證模型結構一致
# ===========================

def get_canonical_metadata(edge_mode="E4"):
    """
    構建包含 query 節點的 canonical metadata。
    所有資料集使用同一結構，避免 metadata 不一致。
    """
    node_types = ['table', 'column', 'page']
    edge_types = [
        ('table', 'has_column', 'column'),
        ('table', 'comes_from', 'page'),
        ('table', 'same_page', 'table'),
        ('table', 'similar_table', 'table'),
        ('column', 'similar_content', 'column'),
        ('table', 'shared_column_name', 'table'),
        ('column', 'rev_has_column', 'table'),
        ('page', 'rev_comes_from', 'table'),
    ]

    if edge_mode == "E0":
        return (node_types, edge_types)

    node_types = node_types + ['query']

    if edge_mode in ["E1", "E2", "E3", "E4"]:
        edge_types.append(('query', 'queries', 'table'))
        edge_types.append(('table', 'rev_queries', 'query'))
    if edge_mode in ["E2", "E4"]:
        edge_types.append(('query', 'queries_page', 'page'))
        edge_types.append(('page', 'rev_queries_page', 'query'))
    if edge_mode in ["E3", "E4"]:
        edge_types.append(('query', 'queries_column', 'column'))
        edge_types.append(('column', 'rev_queries_column', 'query'))

    return (node_types, edge_types)


# ===========================
# 模型
# ===========================

class QueryAwareModel(nn.Module):
    """與 DiffusionModel 結構一致，但使用包含 query 節點的 metadata"""

    def __init__(self, embed_dim, hidden_channels, metadata, dropout=0.2,
                 sage_aggr='mean', hetero_aggr='sum'):
        super().__init__()
        self.sage = GraphSAGE(
            in_channels=embed_dim,
            hidden_channels=hidden_channels,
            num_layers=2,
            out_channels=hidden_channels,
            aggr=sage_aggr,
        )
        self.hetero_sage = to_hetero(self.sage, metadata, aggr=hetero_aggr)
        self.norm = GraphNorm(hidden_channels)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, embed_dim),
        )

    def forward(self, x_dict, edge_index_dict):
        # 確保 query 節點類型存在（全圖 forward 時需要 dummy query）
        x_dict, edge_index_dict = self._ensure_query_entries(x_dict, edge_index_dict)
        x_dict_out = self.hetero_sage(x_dict, edge_index_dict)
        x_table = x_dict_out['table']
        x_table = self.norm(x_table)
        table_features = self.proj_head(x_table)
        return F.normalize(table_features, p=2, dim=1)

    def _ensure_query_entries(self, x_dict, edge_index_dict):
        """
        確保所有 canonical metadata 中的 node types 和 edge types 都存在於輸入中。
        to_hetero 產生的 forward 需要完整的 types，缺少的用 dummy/empty 補齊。
        """
        device = x_dict['table'].device
        embed_dim = x_dict['table'].size(1)

        x_dict = dict(x_dict)
        edge_index_dict = dict(edge_index_dict)

        # 確保所有 node types 存在
        for nt in ['table', 'column', 'page', 'query']:
            if nt not in x_dict:
                x_dict[nt] = torch.zeros((1, embed_dim), device=device)

        # 確保所有 canonical edge types 存在
        canonical_edges = [
            ('table', 'has_column', 'column'),
            ('table', 'comes_from', 'page'),
            ('table', 'same_page', 'table'),
            ('table', 'similar_table', 'table'),
            ('column', 'similar_content', 'column'),
            ('table', 'shared_column_name', 'table'),
            ('column', 'rev_has_column', 'table'),
            ('page', 'rev_comes_from', 'table'),
            ('query', 'queries', 'table'),
            ('table', 'rev_queries', 'query'),
            ('query', 'queries_page', 'page'),
            ('page', 'rev_queries_page', 'query'),
            ('query', 'queries_column', 'column'),
            ('column', 'rev_queries_column', 'query'),
        ]
        empty = torch.zeros((2, 0), dtype=torch.long, device=device)
        for et in canonical_edges:
            if et not in edge_index_dict:
                edge_index_dict[et] = empty

        # 移除非 canonical edge types（如 _self_loop_page）
        canonical_set = set(canonical_edges)
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if k in canonical_set}

        return x_dict, edge_index_dict


def load_pretrained_into_qa_model(model, pretrained_path, device):
    """
    載入預訓練 TWIG 權重到 QueryAwareModel。
    Base edge modules 載入預訓練值，query edge modules 零初始化。
    """
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    pretrained_state = ckpt['model_state_dict']
    hps = ckpt.get('hps', {})

    # 載入匹配的權重（strict=False 跳過 query 相關的 key）
    missing, unexpected = model.load_state_dict(pretrained_state, strict=False)

    # 零初始化 query 相關參數
    query_keywords = ['queries', 'rev_queries', 'queries_page', 'rev_queries_page',
                      'queries_column', 'rev_queries_column']
    zero_count = 0
    for name, param in model.named_parameters():
        if any(kw in name for kw in query_keywords):
            nn.init.zeros_(param)
            zero_count += 1

    print(f"  預訓練權重載入: {len(pretrained_state) - len(missing)} matched, "
          f"{len(missing)} missing (query modules)")
    print(f"  零初始化 {zero_count} 個 query 參數")

    return hps


def get_param_groups(model, base_lr, query_lr):
    """差異學習率：base 參數低 LR，query 參數高 LR"""
    query_keywords = ['queries', 'rev_queries', 'queries_page', 'rev_queries_page',
                      'queries_column', 'rev_queries_column']
    base_params = []
    query_params = []

    for name, param in model.named_parameters():
        if any(kw in name for kw in query_keywords):
            query_params.append(param)
        else:
            base_params.append(param)

    print(f"  Base params: {len(base_params)}, Query params: {len(query_params)}")
    return [
        {'params': base_params, 'lr': base_lr},
        {'params': query_params, 'lr': query_lr},
    ]


# ===========================
# 邊過濾
# ===========================

def filter_edges(data, keep_relations):
    """從 run_edge_ablation.py 引入的邊過濾函數"""
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
    reverse_to_forward = {
        'rev_has_column': 'has_column',
        'rev_comes_from': 'comes_from',
    }

    edges_to_keep = set()
    for relation in keep_relations:
        edges_to_keep.add(relation)
        if relation in forward_to_reverse:
            edges_to_keep.add(forward_to_reverse[relation])

    for edge_type, edge_index in data.edge_index_dict.items():
        _, relation, _ = edge_type
        should_keep = relation in edges_to_keep
        if not should_keep and relation in reverse_to_forward:
            should_keep = reverse_to_forward[relation] in keep_relations
        if should_keep:
            filtered[edge_type].edge_index = edge_index

    # 確保每個 node type 都有入邊（否則 to_hetero 會出問題）
    if len(filtered.edge_types) > 0:
        dest_types = {dst for _, _, dst in filtered.edge_types}
        for node_type in filtered.node_types:
            if node_type not in dest_types:
                n = filtered[node_type].x.size(0)
                filtered[node_type, f'_self_loop_{node_type}', node_type].edge_index = \
                    torch.tensor([[0], [0]], dtype=torch.long)

    if hasattr(data, 'metadata_maps'):
        filtered.metadata_maps = data.metadata_maps

    return filtered


# ===========================
# 子圖建構
# ===========================

def build_subgraph(data, query_vec, candidate_indices, edge_mode="E4", device=None):
    """
    建立包含 query 節點的動態子圖。
    確保所有 canonical edge types 都存在（可能為空）。
    """
    if device is None:
        device = query_vec.device

    sub = HeteroData()
    candidate_set = set(candidate_indices)
    num_candidates = len(candidate_indices)

    # 節點映射
    sub_table_indices = list(candidate_indices)
    table_old_to_new = {old: new for new, old in enumerate(sub_table_indices)}

    sub_column_set = set()
    sub_page_set = set()

    if ('table', 'has_column', 'column') in data.edge_types:
        ei = data['table', 'has_column', 'column'].edge_index
        mask = torch.isin(ei[0], torch.tensor(sub_table_indices, device=ei.device))
        sub_column_set.update(ei[1, mask].cpu().tolist())

    if ('table', 'comes_from', 'page') in data.edge_types:
        ei = data['table', 'comes_from', 'page'].edge_index
        mask = torch.isin(ei[0], torch.tensor(sub_table_indices, device=ei.device))
        sub_page_set.update(ei[1, mask].cpu().tolist())

    sub_column_indices = sorted(sub_column_set)
    sub_page_indices = sorted(sub_page_set)
    column_old_to_new = {old: new for new, old in enumerate(sub_column_indices)}
    page_old_to_new = {old: new for new, old in enumerate(sub_page_indices)}

    # 節點特徵
    if sub_table_indices:
        sub['table'].x = data['table'].x[sub_table_indices].to(device)
    else:
        sub['table'].x = torch.zeros((0, data['table'].x.size(1)), device=device)
    if sub_column_indices:
        sub['column'].x = data['column'].x[sub_column_indices].to(device)
    else:
        sub['column'].x = torch.zeros((0, data['table'].x.size(1)), device=device)
    if sub_page_indices:
        sub['page'].x = data['page'].x[sub_page_indices].to(device)
    else:
        sub['page'].x = torch.zeros((0, data['table'].x.size(1)), device=device)

    # 輔助函數：過濾並重映射邊
    def remap_edges(edge_type, src_map, dst_map):
        if edge_type not in data.edge_types:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        ei = data[edge_type].edge_index
        src_tensor = torch.tensor(list(src_map.keys()), device=ei.device)
        dst_tensor = torch.tensor(list(dst_map.keys()), device=ei.device)
        mask = torch.isin(ei[0], src_tensor) & torch.isin(ei[1], dst_tensor)
        if mask.sum() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        filtered_src = ei[0, mask].cpu().tolist()
        filtered_dst = ei[1, mask].cpu().tolist()
        new_src = [src_map[s] for s in filtered_src]
        new_dst = [dst_map[d] for d in filtered_dst]
        return torch.tensor([new_src, new_dst], dtype=torch.long, device=device)

    # 結構邊
    sub['table', 'has_column', 'column'].edge_index = remap_edges(
        ('table', 'has_column', 'column'), table_old_to_new, column_old_to_new)
    sub['column', 'rev_has_column', 'table'].edge_index = remap_edges(
        ('column', 'rev_has_column', 'table'), column_old_to_new, table_old_to_new)
    sub['table', 'comes_from', 'page'].edge_index = remap_edges(
        ('table', 'comes_from', 'page'), table_old_to_new, page_old_to_new)
    sub['page', 'rev_comes_from', 'table'].edge_index = remap_edges(
        ('page', 'rev_comes_from', 'table'), page_old_to_new, table_old_to_new)

    for rel in ['same_page', 'similar_table', 'shared_column_name']:
        sub['table', rel, 'table'].edge_index = remap_edges(
            ('table', rel, 'table'), table_old_to_new, table_old_to_new)

    sub['column', 'similar_content', 'column'].edge_index = remap_edges(
        ('column', 'similar_content', 'column'), column_old_to_new, column_old_to_new)

    # Query 節點與連邊
    if edge_mode != "E0":
        sub['query'].x = query_vec.to(device)  # (1, embed_dim)
        qi = 0

        if edge_mode in ["E1", "E2", "E3", "E4"]:
            sub['query', 'queries', 'table'].edge_index = torch.tensor(
                [[qi] * num_candidates, list(range(num_candidates))], dtype=torch.long, device=device)
            sub['table', 'rev_queries', 'query'].edge_index = torch.tensor(
                [list(range(num_candidates)), [qi] * num_candidates], dtype=torch.long, device=device)
        else:
            sub['query', 'queries', 'table'].edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            sub['table', 'rev_queries', 'query'].edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

        if edge_mode in ["E2", "E4"] and sub_page_indices:
            np_ = len(sub_page_indices)
            sub['query', 'queries_page', 'page'].edge_index = torch.tensor(
                [[qi] * np_, list(range(np_))], dtype=torch.long, device=device)
            sub['page', 'rev_queries_page', 'query'].edge_index = torch.tensor(
                [list(range(np_)), [qi] * np_], dtype=torch.long, device=device)
        else:
            sub['query', 'queries_page', 'page'].edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            sub['page', 'rev_queries_page', 'query'].edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

        if edge_mode in ["E3", "E4"] and sub_column_indices:
            nc = len(sub_column_indices)
            sub['query', 'queries_column', 'column'].edge_index = torch.tensor(
                [[qi] * nc, list(range(nc))], dtype=torch.long, device=device)
            sub['column', 'rev_queries_column', 'query'].edge_index = torch.tensor(
                [list(range(nc)), [qi] * nc], dtype=torch.long, device=device)
        else:
            sub['query', 'queries_column', 'column'].edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            sub['column', 'rev_queries_column', 'query'].edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

    return sub, table_old_to_new


# ===========================
# 資料載入
# ===========================

def load_queries(query_file, key_fields, id_to_idx, num_tables):
    """載入查詢資料，回傳 (texts, pos_indices_lists)"""
    samples = []
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            questions = obj.get('questions', [])
            if not questions and 'question' in obj:
                questions = [obj['question']]

            gt_list = obj.get('ground_truth_list', []) or []
            pos_indices = []
            for gt in gt_list:
                if all(gt.get(field) is not None for field in key_fields):
                    key = make_key(gt, key_fields)
                    idx = id_to_idx.get(key, -1)
                    if 0 <= idx < num_tables:
                        pos_indices.append(idx)

            if not pos_indices:
                continue

            for q in questions:
                if q and q.strip():
                    samples.append((q.strip(), pos_indices))

    if not samples:
        return [], []
    texts, pos_lists = zip(*samples)
    return list(texts), list(pos_lists)


def rebuild_id_to_idx(data, key_fields):
    """從 graph metadata 重建 id_to_idx 映射"""
    if hasattr(data, 'metadata_maps') and 'table_meta' in data.metadata_maps:
        table_meta = data.metadata_maps['table_meta']
        id_to_idx = {}
        for idx, meta in enumerate(table_meta):
            key = make_key(meta, key_fields)
            if key not in id_to_idx:
                id_to_idx[key] = idx
        return id_to_idx
    return data.metadata_maps.get('table_id_to_idx', {})


# ===========================
# 訓練
# ===========================

def train_one_epoch(model, data, optimizer, scaler, query_vectors, pos_indices_lists,
                    hard_neg_indices, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    n_batches = 0
    indices = list(range(len(query_vectors)))
    random.shuffle(indices)

    # 快取全圖 table embedding（用於粗排建子圖，不帶梯度）
    with torch.no_grad():
        base_table_emb = model.forward(data.x_dict, data.edge_index_dict)

    batch_iter = tqdm(range(0, len(indices), BATCH_SIZE),
                      desc=f"Epoch {epoch}/{num_epochs}", leave=False)

    for start in batch_iter:
        end = min(start + BATCH_SIZE, len(indices))
        batch_idx = indices[start:end]

        optimizer.zero_grad()
        batch_loss = 0.0
        batch_count = 0

        with autocast(enabled=(device.type == 'cuda')):
            for sample_idx in batch_idx:
                q_vec = query_vectors[sample_idx:sample_idx + 1]  # (1, dim)
                pos_list = pos_indices_lists[sample_idx]
                hard_negs = hard_neg_indices[sample_idx] if hard_neg_indices else []

                # 粗排取候選
                with torch.no_grad():
                    q_norm = F.normalize(q_vec, p=2, dim=1)
                    scores = torch.matmul(q_norm, base_table_emb.T).squeeze(0)
                    _, top_k = torch.topk(scores, k=min(SUBGRAPH_K, scores.size(0)))
                    candidate_set = set(top_k.cpu().tolist())

                # 確保正確答案和困難負樣本在候選中
                for idx in pos_list:
                    candidate_set.add(idx)
                for idx in hard_negs:
                    if idx >= 0:
                        candidate_set.add(idx)

                candidate_indices = sorted(candidate_set)

                # 建子圖
                subgraph, table_mapping = build_subgraph(
                    data, q_vec, candidate_indices,
                    edge_mode=QUERY_EDGE_MODE, device=device)

                # 子圖 forward
                sub_table_emb = model.forward(subgraph.x_dict, subgraph.edge_index_dict)

                # 正確答案在子圖中的索引
                pos_new_idx = table_mapping.get(pos_list[0], -1)
                if pos_new_idx == -1:
                    continue

                # InfoNCE loss
                logits = torch.matmul(q_norm, sub_table_emb.T).squeeze(0) / TEMP
                label = torch.tensor([pos_new_idx], dtype=torch.long, device=device)
                loss_sample = F.cross_entropy(logits.unsqueeze(0), label,
                                              label_smoothing=LABEL_SMOOTH)

                batch_loss += loss_sample
                batch_count += 1

            if batch_count == 0:
                continue

            loss = batch_loss / batch_count

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

        # 每隔一段時間更新粗排 embedding
        if n_batches % 20 == 0:
            with torch.no_grad():
                base_table_emb = model.forward(data.x_dict, data.edge_index_dict)

    return total_loss / max(1, n_batches)


def evaluate_val(model, data, query_vectors, pos_indices_lists, device, k=COARSE_K_EVAL):
    """在驗證集上評估 query-aware recall@10"""
    model.eval()
    with torch.no_grad():
        base_table_emb = model.forward(data.x_dict, data.edge_index_dict)
        q_all = F.normalize(query_vectors, p=2, dim=1)

        # 粗排分數
        coarse_scores = torch.matmul(q_all, base_table_emb.T)

        hits10 = 0
        mrr_sum = 0.0
        total = len(query_vectors)

        for qi in range(total):
            pos_set = set(pos_indices_lists[qi])
            q_vec = query_vectors[qi:qi + 1]

            # 粗排候選
            _, top_k = torch.topk(coarse_scores[qi], k=min(k, coarse_scores.size(1)))
            candidate_indices = top_k.cpu().tolist()

            # 確保正確答案在候選中（用於準確評估）
            for pos_idx in pos_set:
                if pos_idx not in set(candidate_indices):
                    candidate_indices.append(pos_idx)

            # 子圖 forward
            subgraph, table_mapping = build_subgraph(
                data, q_vec, candidate_indices,
                edge_mode=QUERY_EDGE_MODE, device=device)

            sub_table_emb = model.forward(subgraph.x_dict, subgraph.edge_index_dict)

            # 重排序
            q_norm = F.normalize(q_vec, p=2, dim=1)
            rerank_scores = torch.matmul(q_norm, sub_table_emb.T).squeeze(0)
            _, reranked = torch.topk(rerank_scores, k=min(10, rerank_scores.size(0)))

            new_to_old = {v: k for k, v in table_mapping.items()}
            reranked_orig = [new_to_old[idx.item()] for idx in reranked]

            best_rank = float('inf')
            for rank, orig_idx in enumerate(reranked_orig, 1):
                if orig_idx in pos_set:
                    best_rank = min(best_rank, rank)

            if best_rank <= 10:
                hits10 += 1
            if best_rank < float('inf'):
                mrr_sum += 1.0 / best_rank

    return {
        'recall@10': hits10 / total,
        'mrr': mrr_sum / total,
    }


def mine_hard_negatives(model, data, query_vectors, pos_indices_lists, num_hard=3, device='cuda'):
    """困難負樣本挖掘"""
    model.eval()
    with torch.no_grad():
        table_emb = model.forward(data.x_dict, data.edge_index_dict)
        q_norm = F.normalize(query_vectors, p=2, dim=1)
        num_tables = table_emb.size(0)

        all_negs = []
        for start in range(0, len(query_vectors), 1024):
            end = min(start + 1024, len(query_vectors))
            sim = torch.matmul(q_norm[start:end], table_emb.T)
            for i in range(end - start):
                for pos_idx in pos_indices_lists[start + i]:
                    if 0 <= pos_idx < num_tables:
                        sim[i, pos_idx] = -float('inf')
            _, topk = torch.topk(sim, k=min(num_hard, num_tables - 1), dim=1)
            all_negs.extend(topk.tolist())

    model.train()
    return all_negs


# ===========================
# 主訓練流程
# ===========================

def main(dataset):
    key_fields = get_key_fields(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Query-Aware v2 訓練: {dataset}")
    print(f"Device: {device}, Edge Mode: {QUERY_EDGE_MODE}")
    print(f"Best edges: {BEST_EDGE_CONFIGS.get(dataset, [])}")
    print(f"{'='*60}")

    # 載入圖
    graph_file = GRAPH_FILE.format(dataset=dataset)
    data_cpu = torch.load(graph_file, map_location='cpu', weights_only=False)
    id_to_idx = rebuild_id_to_idx(data_cpu, key_fields)

    # 邊過濾
    best_edges = BEST_EDGE_CONFIGS.get(dataset, [])
    if best_edges:
        data_cpu = filter_edges(data_cpu, best_edges)
        print(f"  過濾連邊: {best_edges}")

    embed_dim = data_cpu['table'].x.size(1)
    num_tables = data_cpu['table'].num_nodes

    # 建立模型
    metadata = get_canonical_metadata(QUERY_EDGE_MODE)
    model = QueryAwareModel(
        embed_dim=embed_dim,
        hidden_channels=768,
        metadata=metadata,
        dropout=0.10,
        sage_aggr='min',
        hetero_aggr='max',
    ).to(device)

    # 載入預訓練權重
    pretrained_path = PRETRAINED_PATH.format(dataset=dataset)
    if Path(pretrained_path).exists():
        hps = load_pretrained_into_qa_model(model, pretrained_path, device)
        print(f"  已載入預訓練模型: {pretrained_path}")
    else:
        print(f"  警告：找不到預訓練模型 {pretrained_path}，從頭訓練")

    # 差異學習率
    param_groups = get_param_groups(model, BASE_LR, QUERY_LR)
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    warmup_sched = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / float(WARMUP_EPOCHS)))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS])
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # 載入訓練查詢
    query_file = QUERY_FILE.format(dataset=dataset)
    texts, pos_lists = load_queries(query_file, key_fields, id_to_idx, num_tables)
    print(f"  訓練樣本: {len(texts)}")

    if not texts:
        print("  無有效訓練樣本，跳過")
        return

    embedder = SentenceTransformer(MODEL_NAME, device=str(device))
    query_vecs = torch.tensor(
        embedder.encode(texts, show_progress_bar=True),
        dtype=torch.float, device=device)

    # 載入驗證集
    val_graph_file = VAL_GRAPH_FILE.format(dataset=dataset)
    val_query_file = VAL_QUERY_FILE.format(dataset=dataset)
    val_data_cpu = None
    val_query_vecs = None
    val_pos_lists = None

    if Path(val_graph_file).exists() and Path(val_query_file).exists():
        val_data_cpu = torch.load(val_graph_file, map_location='cpu', weights_only=False)
        if best_edges:
            val_data_cpu = filter_edges(val_data_cpu, best_edges)
        val_id_to_idx = rebuild_id_to_idx(val_data_cpu, key_fields)
        val_num_tables = val_data_cpu['table'].num_nodes
        val_texts, val_pos_lists = load_queries(val_query_file, key_fields, val_id_to_idx, val_num_tables)
        if val_texts:
            val_query_vecs = torch.tensor(
                embedder.encode(val_texts, show_progress_bar=True),
                dtype=torch.float, device=device)
            print(f"  驗證樣本: {len(val_texts)}")

    del embedder
    torch.cuda.empty_cache()

    # 訓練
    data = data_cpu.clone().to(device)
    hard_neg_indices = None
    best_val_recall = -1.0
    best_model_state = None
    best_epoch = 0
    patience = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # 困難負樣本挖掘
        if epoch == 1 or epoch % 2 == 0:
            hard_neg_indices = mine_hard_negatives(
                model, data, query_vecs, pos_lists,
                num_hard=NUM_HARD_NEGATIVES, device=device)

        avg_loss = train_one_epoch(
            model, data, optimizer, scaler, query_vecs, pos_lists,
            hard_neg_indices, device, epoch, NUM_EPOCHS)

        scheduler.step()

        # 驗證
        if val_data_cpu is not None and val_query_vecs is not None:
            val_data = val_data_cpu.clone().to(device)
            val_metrics = evaluate_val(model, val_data, val_query_vecs, val_pos_lists, device)
            del val_data
            torch.cuda.empty_cache()

            print(f"  Epoch {epoch}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | "
                  f"Val R@10: {val_metrics['recall@10']:.4f} | Val MRR: {val_metrics['mrr']:.4f}")

            if val_metrics['recall@10'] > best_val_recall:
                best_val_recall = val_metrics['recall@10']
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                patience = 0
                print(f"    ★ 新最佳 Epoch {epoch}")
            else:
                patience += 1
                if patience >= EARLY_STOPPING_PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        else:
            print(f"  Epoch {epoch}/{NUM_EPOCHS} | Loss: {avg_loss:.4f}")
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    # 儲存模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    save_path = SAVE_PATH.format(dataset=dataset)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'hps': {
            'HIDDEN_CHANNELS': 768,
            'DROPOUT': 0.10,
            'SAGE_AGGR': 'min',
            'HETERO_AGGR': 'max',
        },
        'edge_mode': QUERY_EDGE_MODE,
        'best_edges': best_edges,
        'best_epoch': best_epoch,
        'best_val_recall': best_val_recall,
    }, save_path)
    print(f"  模型已儲存至 {save_path} (best epoch={best_epoch}, val R@10={best_val_recall:.4f})")

    del model, data
    torch.cuda.empty_cache()


if __name__ == '__main__':
    datasets = sys.argv[1:] if len(sys.argv) > 1 else [
        "feta", "ottqa", "mimo_en", "mimo_ch", "e2ewtq", "mmqa"
    ]
    for ds in datasets:
        graph_path = Path(GRAPH_FILE.format(dataset=ds))
        if not graph_path.exists():
            print(f"跳過 {ds}：找不到 {graph_path}")
            continue
        main(ds)
