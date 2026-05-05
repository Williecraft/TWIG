#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查詢導向之圖式表格檢索訓練腳本 (Query-Aware Training)

改進原有 train_model.py：
在訓練階段就讓模型學習處理包含 query 節點的子圖，
使 query 資訊透過 GNN 聚合機制影響表格的最終表示。

核心改動：
1. 模型架構不變（仍是 GraphSAGE + to_hetero），但 metadata 中包含 query 節點
2. 每個 batch 中，為每個查詢建立包含 query 節點的動態子圖
3. 在子圖上 forward，取得 query 影響後的 table embedding
4. 用更新後的 embedding 計算 contrastive loss

消融設定：
- QUERY_EDGE_MODE: E0/E1/E2/E3/E4
- SUBGRAPH_K: 動態子圖候選數量
"""

import json
import csv
import copy
import random
import itertools

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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
from pathlib import Path

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATASETS = [
    # "mimo_en",
    # "mimo_ch",
    # "ottqa",
    "feta",
    # "e2ewtq",
    # "mmqa",
]

# Query-Aware 設定
QUERY_EDGE_MODE = "E4"    # E0/E1/E2/E3/E4
SUBGRAPH_K = 10           # 動態子圖候選數量


def get_key_fields(dataset_name: str) -> tuple:
    """根據資料集名稱返回對應的 KEY_FIELDS"""
    if dataset_name in ["ottqa", "feta", "e2ewtq"]:
        return ("sheet_name", "file_name")
    elif dataset_name in ["mimo_en", "mimo_ch", "mmqa"]:
        return ("id",)
    else:
        return ("id",)


# 資料路徑
GRAPH_FILE = "/user_data/TabGNN/data/processed/train/{dataset}/graph.pt"
QUERY_FILE = "/user_data/TabGNN/data/table/train/{dataset}/query.jsonl"

# 驗證集路徑
VAL_GRAPH_FILE = "/user_data/TabGNN/data/processed/dev/{dataset}/graph.pt"
VAL_QUERY_FILE = "/user_data/TabGNN/data/table/dev/{dataset}/query.jsonl"

SAVE_PATH = "/user_data/TabGNN/checkpoints/{dataset}/model_query_aware.pt"
RESULTS_FILE = "/user_data/TabGNN/results/grid_search_results_query_aware.csv"

MODEL_NAME = 'BAAI/bge-m3'
NUM_EPOCHS = 30
WARMUP_EPOCHS = 2
BATCH_SIZE = 128

# 訓練超參數
CLIP_GRAD_NORM = 0.60
CHUNK_SIZE = 1024
TEMP_START = 0.05
TEMP_END = 0.03
SMOOTH_START = 0.120
SMOOTH_END = 0.060

# Grid Search / Hyperopt 設定
USE_HYPEROPT = False
MAX_EVALS = 300
BEST_PARAMS = {
    'LEARNING_RATE': 0.0005543418807199451,
    'HIDDEN_CHANNELS': 768,
    'DROPOUT': 0.10028905529000982,
    'WEIGHT_DECAY': 0.03217253330215496,
    'SAGE_AGGR': 'min',
    'HETERO_AGGR': 'max',
    'CLIP_GRAD_NORM': 0.60,
    'TEMP_START': 0.05,
    'TEMP_END': 0.03,
    'SMOOTH_START': 0.120,
    'SMOOTH_END': 0.060,
}

# Hard Negative 設定
NUM_HARD_NEGATIVES = 3
REMINING_INTERVAL = 1

# Early Stopping 設定
EARLY_STOPPING_PATIENCE = 10

# ===========================


def make_key(item: dict, key_fields: tuple) -> str:
    """根據 key_fields 組合產生唯一鍵"""
    return "|".join(str(item.get(f, "")) for f in key_fields)


def get_embedder(model_name: str = MODEL_NAME, device: str = 'cuda') -> SentenceTransformer:
    """載入 SentenceTransformer 嵌入模型"""
    return SentenceTransformer(model_name, device=device)


class QueryAwareDiffusionModel(nn.Module):
    """
    查詢導向異構圖 GNN 模型。
    
    相比原始 DiffusionModel，此模型的 metadata 中包含 'query' 節點類型，
    使得模型在訓練和推論時都能處理包含 query 節點的子圖。
    """

    def __init__(self, embed_dim: int, hidden_channels: int, metadata, dropout: float = 0.2,
                 sage_aggr: str = 'mean', hetero_aggr: str = 'sum'):
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

    def forward(self, x_dict: dict, edge_index_dict: dict):
        """前向傳播，返回 L2 歸一化的表格嵌入"""
        x_dict_out = self.hetero_sage(x_dict, edge_index_dict)
        x_table = x_dict_out['table']
        x_table = self.norm(x_table)
        table_features = self.proj_head(x_table)
        return F.normalize(table_features, p=2, dim=1)

    def forward_full(self, x_dict: dict, edge_index_dict: dict):
        """完整前向傳播，返回所有節點類型的輸出"""
        x_dict_out = self.hetero_sage(x_dict, edge_index_dict)
        return x_dict_out

    def score_tables(self, data: HeteroData, query_vec: torch.Tensor):
        """計算查詢向量與所有表格的相似度分數"""
        table_embeddings = self.forward(data.x_dict, data.edge_index_dict)
        query_vec_norm = F.normalize(query_vec.to(table_embeddings.device), p=2, dim=1)
        scores = torch.matmul(query_vec_norm, table_embeddings.T).squeeze(0)
        return scores


def get_query_aware_metadata(base_data: HeteroData, edge_mode: str = "E4"):
    """
    根據 edge_mode 建構包含 query 節點的 metadata。
    
    返回 (node_types, edge_types) 元組。
    """
    node_types = list(base_data.node_types)
    edge_types = list(base_data.edge_types)

    if edge_mode == "E0":
        return (node_types, edge_types)

    # 加入 query 節點類型
    if 'query' not in node_types:
        node_types.append('query')

    # E1: Query–Table
    if edge_mode in ["E1", "E2", "E3", "E4"]:
        edge_types.append(('query', 'queries', 'table'))
        edge_types.append(('table', 'rev_queries', 'query'))

    # E2: + Query–Page
    if edge_mode in ["E2", "E4"]:
        edge_types.append(('query', 'queries_page', 'page'))
        edge_types.append(('page', 'rev_queries_page', 'query'))

    # E3: + Query–Column
    if edge_mode in ["E3", "E4"]:
        edge_types.append(('query', 'queries_column', 'column'))
        edge_types.append(('column', 'rev_queries_column', 'query'))

    return (node_types, edge_types)


def build_training_subgraph(
    data: HeteroData,
    query_vec: torch.Tensor,
    pos_table_indices: list,
    neg_table_indices: list,
    all_table_emb: torch.Tensor,
    subgraph_k: int,
    edge_mode: str = "E4",
    device: torch.device = None,
):
    """
    為訓練建立包含 query 節點的動態子圖。
    
    與評估時不同，訓練時需要確保正確答案表格在子圖中。
    
    流程：
    1. 用固定 embedding 粗排取 top-k 候選
    2. 確保正確答案和困難負樣本都在候選中
    3. 從候選中收集相關 column 和 page
    4. 加入 query 節點和連邊
    """
    if device is None:
        device = query_vec.device

    # Step 1: 粗排取候選
    with torch.no_grad():
        q_norm = F.normalize(query_vec, p=2, dim=1)
        scores = torch.matmul(q_norm, all_table_emb.T).squeeze(0)
        _, coarse_top = torch.topk(scores, k=min(subgraph_k, scores.size(0)))
        candidate_set = set(coarse_top.cpu().tolist())

    # Step 2: 確保正確答案和困難負樣本在候選中
    for idx in pos_table_indices:
        if idx >= 0:
            candidate_set.add(idx)
    for idx in neg_table_indices:
        if idx >= 0:
            candidate_set.add(idx)

    candidate_indices = sorted(candidate_set)
    num_candidates = len(candidate_indices)

    sub = HeteroData()
    table_old_to_new = {old: new for new, old in enumerate(candidate_indices)}

    # 從邊資訊找出相關的 column 和 page 節點
    sub_column_set = set()
    sub_page_set = set()

    if ('table', 'has_column', 'column') in data.edge_types:
        edge_idx = data['table', 'has_column', 'column'].edge_index
        for i in range(edge_idx.size(1)):
            src = edge_idx[0, i].item()
            dst = edge_idx[1, i].item()
            if src in candidate_set:
                sub_column_set.add(dst)

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

    # 節點特徵
    if candidate_indices:
        sub['table'].x = data['table'].x[candidate_indices].to(device)
    if sub_column_indices:
        sub['column'].x = data['column'].x[sub_column_indices].to(device)
    if sub_page_indices:
        sub['page'].x = data['page'].x[sub_page_indices].to(device)

    # 子圖邊
    def filter_and_remap_edges(edge_type, src_map, dst_map):
        if edge_type not in data.edge_types:
            return
        edge_idx = data[edge_type].edge_index
        new_src, new_dst = [], []
        for i in range(edge_idx.size(1)):
            s = edge_idx[0, i].item()
            d = edge_idx[1, i].item()
            if s in src_map and d in dst_map:
                new_src.append(src_map[s])
                new_dst.append(dst_map[d])
        if new_src:
            sub[edge_type].edge_index = torch.tensor([new_src, new_dst], dtype=torch.long, device=device)

    filter_and_remap_edges(('table', 'has_column', 'column'), table_old_to_new, column_old_to_new)
    filter_and_remap_edges(('column', 'rev_has_column', 'table'), column_old_to_new, table_old_to_new)
    filter_and_remap_edges(('table', 'comes_from', 'page'), table_old_to_new, page_old_to_new)
    filter_and_remap_edges(('page', 'rev_comes_from', 'table'), page_old_to_new, table_old_to_new)

    for rel in ['same_page', 'similar_table', 'shared_column_name']:
        filter_and_remap_edges(('table', rel, 'table'), table_old_to_new, table_old_to_new)

    filter_and_remap_edges(('column', 'similar_content', 'column'), column_old_to_new, column_old_to_new)

    # 加入 Query 節點及連邊
    if edge_mode != "E0":
        sub['query'].x = query_vec.to(device)
        query_idx = 0

        if edge_mode in ["E1", "E2", "E3", "E4"]:
            q_to_t_src = [query_idx] * num_candidates
            q_to_t_dst = list(range(num_candidates))
            sub['query', 'queries', 'table'].edge_index = torch.tensor(
                [q_to_t_src, q_to_t_dst], dtype=torch.long, device=device
            )
            sub['table', 'rev_queries', 'query'].edge_index = torch.tensor(
                [q_to_t_dst, q_to_t_src], dtype=torch.long, device=device
            )

        if edge_mode in ["E2", "E4"] and sub_page_indices:
            num_pages = len(sub_page_indices)
            q_to_p_src = [query_idx] * num_pages
            q_to_p_dst = list(range(num_pages))
            sub['query', 'queries_page', 'page'].edge_index = torch.tensor(
                [q_to_p_src, q_to_p_dst], dtype=torch.long, device=device
            )
            sub['page', 'rev_queries_page', 'query'].edge_index = torch.tensor(
                [q_to_p_dst, q_to_p_src], dtype=torch.long, device=device
            )

        if edge_mode in ["E3", "E4"] and sub_column_indices:
            num_cols = len(sub_column_indices)
            q_to_c_src = [query_idx] * num_cols
            q_to_c_dst = list(range(num_cols))
            sub['query', 'queries_column', 'column'].edge_index = torch.tensor(
                [q_to_c_src, q_to_c_dst], dtype=torch.long, device=device
            )
            sub['column', 'rev_queries_column', 'query'].edge_index = torch.tensor(
                [q_to_c_dst, q_to_c_src], dtype=torch.long, device=device
            )

    return sub, table_old_to_new


def load_training_data(query_file_path: str, id_to_idx: dict, data: HeteroData = None, num_hard_negatives: int = 1):
    """
    載入 training query 資料
    
    回傳:
      1. queries_text: List[str]
      2. pos_indices_lists: List[List[int]]
      3. hard_neg_indices: List[List[int]]
    """
    if data is not None and hasattr(data, 'metadata_maps') and 'table_meta' in data.metadata_maps:
        print(f"根據 KEY_FIELDS={KEY_FIELDS} 重建 id_to_idx...")
        table_meta = data.metadata_maps['table_meta']
        id_to_idx = {}
        for idx, meta in enumerate(table_meta):
            key = make_key(meta, KEY_FIELDS)
            if key not in id_to_idx:
                id_to_idx[key] = idx
    else:
        print("警告：Graph data 中沒有 table_meta，使用預設的 id_to_idx。")

    training_samples = []
    print(f"從 {query_file_path} 載入 queries...")

    # 建立鄰居索引
    table_neighbors = {}
    if data is not None:
        print("正在建立鄰居索引...")
        t2c = data['table', 'has_column', 'column'].edge_index.cpu()
        c2c = data['column', 'similar_content', 'column'].edge_index.cpu()

        c2t_map = {}
        for i in range(t2c.size(1)):
            c2t_map[t2c[1, i].item()] = t2c[0, i].item()

        for i in tqdm(range(c2c.size(1)), desc="Building Graph Index"):
            c_src, c_dst = c2c[0, i].item(), c2c[1, i].item()
            if c_src in c2t_map and c_dst in c2t_map:
                t_src, t_dst = c2t_map[c_src], c2t_map[c_dst]
                if t_src != t_dst:
                    table_neighbors.setdefault(t_src, set()).add(t_dst)
        print(f"Total tables with neighbors: {len(table_neighbors)}")

    with open(query_file_path, "r", encoding='utf-8') as f:
        for line in tqdm(f, desc="載入 queries"):
            temp = json.loads(line)

            ground_truth_list = temp.get('ground_truth_list', []) or []
            pos_indices_list = []
            for gt in ground_truth_list:
                if all(gt.get(f) is not None for f in KEY_FIELDS):
                    key = make_key(gt, KEY_FIELDS)
                    idx = id_to_idx.get(key, -1)
                    if idx != -1:
                        pos_indices_list.append(idx)

            if not pos_indices_list:
                continue

            all_neighbors = set()
            for pos_idx in pos_indices_list:
                all_neighbors.update(table_neighbors.get(pos_idx, set()))
            all_neighbors -= set(pos_indices_list)
            neighbors = list(all_neighbors)

            if len(neighbors) >= num_hard_negatives:
                hard_negs = random.sample(neighbors, num_hard_negatives)
            else:
                hard_negs = neighbors + [-1] * (num_hard_negatives - len(neighbors))

            questions = temp.get('questions', [])
            if not questions and 'question' in temp:
                questions = [temp['question']]

            for question in questions:
                if question and question.strip():
                    training_samples.append((question, pos_indices_list, hard_negs))

    if not training_samples:
        print("警告：沒有載入任何有效的訓練樣本。")
        return [], [], []

    queries_text, pos_indices_lists, hard_neg_indices = [list(t) for t in zip(*training_samples)]

    if data is not None:
        num_tables = data['table'].num_nodes
        valid_samples = []
        for q, p_list, h in zip(queries_text, pos_indices_lists, hard_neg_indices):
            valid_pos = [p for p in p_list if p < num_tables]
            if valid_pos:
                valid_h = [idx if idx < num_tables else -1 for idx in h]
                valid_samples.append((q, valid_pos, valid_h))
        if len(valid_samples) < len(queries_text):
            queries_text, pos_indices_lists, hard_neg_indices = [list(t) for t in zip(*valid_samples)]
            print(f"Filtered to {len(queries_text)} valid samples.")

    return queries_text, pos_indices_lists, hard_neg_indices


def mine_hard_negatives_topk(model, data, query_vectors, pos_indices_lists, num_hard_negatives=5, device='cuda'):
    """Query-Aware 困難負樣本挖掘"""
    model.eval()
    with torch.no_grad():
        table_emb = F.normalize(model.forward(data.x_dict, data.edge_index_dict), p=2, dim=1)
        q_norm = F.normalize(query_vectors, p=2, dim=1)

        num_tables = table_emb.size(0)
        actual_k = min(num_hard_negatives, num_tables - 1)

        all_topk_indices = []
        chunk_size = 1024

        for start in range(0, len(query_vectors), chunk_size):
            end = min(start + chunk_size, len(query_vectors))
            q_chunk = q_norm[start:end]
            sim_chunk = torch.matmul(q_chunk, table_emb.T)

            for i in range(end - start):
                pos_list = pos_indices_lists[start + i]
                for pos_idx in pos_list:
                    if 0 <= pos_idx < num_tables:
                        sim_chunk[i, pos_idx] = -float('inf')

            _, topk_idx = torch.topk(sim_chunk, k=actual_k, dim=1)
            all_topk_indices.extend(topk_idx.tolist())

        if actual_k < num_hard_negatives:
            padding = [-1] * (num_hard_negatives - actual_k)
            all_topk_indices = [negs + padding for negs in all_topk_indices]

    model.train()
    return all_topk_indices


def embed_queries(queries_text: list, embedder, device):
    """嵌入查詢文本"""
    print(f"共 {len(queries_text)} 個訓練樣本。開始嵌入查詢向量...")
    query_vectors_np = embedder.encode(queries_text, show_progress_bar=True)
    return torch.tensor(query_vectors_np, dtype=torch.float, device=device)


def setup_components(embed_dim: int, metadata, hps: dict, device):
    """初始化模型、優化器和排程器"""
    model = QueryAwareDiffusionModel(
        embed_dim=embed_dim,
        hidden_channels=hps['HIDDEN_CHANNELS'],
        metadata=metadata,
        dropout=hps['DROPOUT'],
        sage_aggr=hps.get('SAGE_AGGR', 'mean'),
        hetero_aggr=hps.get('HETERO_AGGR', 'sum'),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=hps['LEARNING_RATE'], weight_decay=hps['WEIGHT_DECAY'])

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / float(hps['WARMUP_EPOCHS'])))
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=hps['NUM_EPOCHS'] - hps['WARMUP_EPOCHS'])
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[hps['WARMUP_EPOCHS']])

    scaler = GradScaler(enabled=(device.type == 'cuda'))
    return model, optimizer, scheduler, scaler


def compute_scores_chunked(query_emb_norm: torch.Tensor, table_emb: torch.Tensor, chunk_size: int = 1024):
    """分塊計算相似度矩陣"""
    scores_all = []
    for i in range(0, table_emb.size(0), chunk_size):
        chunk_scores = torch.matmul(query_emb_norm, table_emb[i:i + chunk_size].T)
        scores_all.append(chunk_scores)
    return torch.cat(scores_all, dim=1)


def train_one_epoch_query_aware(
    model, data, optimizer, scaler, scheduler,
    query_vectors, pos_indices_lists, hard_neg_indices,
    hps, epoch, device, edge_mode, subgraph_k,
):
    """
    執行一個 Query-Aware 訓練週期。
    
    與原始訓練不同，這裡為每個 batch 中的查詢建立動態子圖並進行子圖推論。
    但為了效率，我們先在全圖上做 forward 得到初始 table embedding（用於粗排），
    然後在建立子圖後再做一次 forward。
    """
    model.train()
    total_loss = 0.0
    indices = list(range(len(query_vectors)))
    random.shuffle(indices)

    progress = epoch / float(hps['NUM_EPOCHS'])
    curr_temp = hps['TEMP_END'] if epoch > hps['NUM_EPOCHS'] * 0.7 else hps['TEMP_START']
    curr_smooth = hps['SMOOTH_START'] + (hps['SMOOTH_END'] - hps['SMOOTH_START']) * progress

    batch_iterator = tqdm(range(0, len(indices), hps['BATCH_SIZE']), desc=f"Epoch {epoch}/{hps['NUM_EPOCHS']}")

    for start in batch_iterator:
        end = min(start + hps['BATCH_SIZE'], len(indices))
        batch_idx = indices[start:end]

        optimizer.zero_grad()

        # 先用全圖 forward 得到基礎 table embedding（用於粗排候選和計算損失）
        with autocast(enabled=(device.type == 'cuda')):
            # 全圖 forward（也讓模型學習全圖的表示）
            base_table_emb = model.forward(data.x_dict, data.edge_index_dict)

            batch_loss = 0.0
            batch_count = 0

            for i_in_batch, sample_idx in enumerate(batch_idx):
                q_vec = query_vectors[sample_idx:sample_idx+1]  # (1, embed_dim)
                pos_list = pos_indices_lists[sample_idx]
                hard_negs = hard_neg_indices[sample_idx]

                if edge_mode == "E0":
                    # E0：不用子圖，直接用全圖 embedding
                    q_norm = F.normalize(q_vec, p=2, dim=1)
                    logits = torch.matmul(q_norm, base_table_emb.T).squeeze(0) / curr_temp
                    label = torch.tensor([pos_list[0]], dtype=torch.long, device=device)
                    loss_sample = F.cross_entropy(logits.unsqueeze(0), label, label_smoothing=curr_smooth)
                else:
                    # Query-Aware：建立子圖
                    subgraph, table_mapping = build_training_subgraph(
                        data, q_vec, pos_list, hard_negs,
                        base_table_emb.detach(),
                        subgraph_k=subgraph_k,
                        edge_mode=edge_mode,
                        device=device,
                    )

                    # 在子圖上 forward
                    x_dict_out = model.hetero_sage(subgraph.x_dict, subgraph.edge_index_dict)
                    if 'table' in x_dict_out:
                        x_table = x_dict_out['table']
                        x_table = model.norm(x_table)
                        sub_table_emb = model.proj_head(x_table)
                        sub_table_emb = F.normalize(sub_table_emb, p=2, dim=1)
                    else:
                        continue

                    q_norm = F.normalize(q_vec, p=2, dim=1)
                    logits = torch.matmul(q_norm, sub_table_emb.T).squeeze(0) / curr_temp

                    # 正確答案在子圖中的索引
                    pos_new_idx = table_mapping.get(pos_list[0], -1)
                    if pos_new_idx == -1:
                        continue

                    label = torch.tensor([pos_new_idx], dtype=torch.long, device=device)
                    loss_sample = F.cross_entropy(logits.unsqueeze(0), label, label_smoothing=curr_smooth)

                batch_loss += loss_sample
                batch_count += 1

            if batch_count > 0:
                loss = batch_loss / batch_count
            else:
                continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hps['CLIP_GRAD_NORM'])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    scheduler.step()
    return total_loss / max(1, len(batch_iterator)), curr_temp, curr_smooth


def evaluate_retrieval_query_aware(
    model, data, query_vectors, pos_indices_lists,
    hps, device, edge_mode, subgraph_k,
):
    """
    Query-Aware 評估：在子圖上推論後重新排序
    """
    model.eval()
    with torch.no_grad():
        # 先得到固定 table embedding（用於粗排）
        base_table_emb = model.forward(data.x_dict, data.edge_index_dict)
        q_all_norm = F.normalize(query_vectors, p=2, dim=1)

        if edge_mode == "E0":
            # 直接用固定表示評估
            score_mat = compute_scores_chunked(q_all_norm, base_table_emb, hps['CHUNK_SIZE'])

            topk_vals = [1, 5, 10]
            hits = {k: 0 for k in topk_vals}
            mrr_sum = 0.0

            score_mat_cpu = score_mat.cpu()
            num_queries = score_mat_cpu.size(0)

            for q_i in range(num_queries):
                rank_indices = torch.argsort(score_mat_cpu[q_i], descending=True)
                pos_set = set(pos_indices_lists[q_i])
                best_rank = float('inf')
                for pos_idx in pos_set:
                    rank_pos = (rank_indices == pos_idx).nonzero(as_tuple=True)[0]
                    if rank_pos.numel() > 0:
                        rank = rank_pos.item() + 1
                        best_rank = min(best_rank, rank)
                if best_rank < float('inf'):
                    mrr_sum += 1.0 / best_rank
                    for k in topk_vals:
                        if best_rank <= k:
                            hits[k] += 1

            return {
                'mrr': mrr_sum / num_queries,
                **{f'recall@{k}': hits[k] / num_queries for k in topk_vals}
            }
        else:
            # Query-Aware 評估
            topk_vals = [1, 5, 10]
            hits = {k: 0 for k in topk_vals}
            mrr_sum = 0.0
            num_queries = len(query_vectors)

            # 粗排分數
            coarse_scores = torch.matmul(q_all_norm, base_table_emb.T)

            for q_i in range(num_queries):
                q_vec = query_vectors[q_i:q_i+1]
                pos_set = set(pos_indices_lists[q_i])

                # 粗排取候選
                _, coarse_top = torch.topk(
                    coarse_scores[q_i],
                    k=min(subgraph_k, coarse_scores.size(1))
                )
                candidate_indices = coarse_top.cpu().tolist()

                # 確保正確答案在候選中（用於準確評估）
                for pos_idx in pos_set:
                    if pos_idx not in set(candidate_indices):
                        candidate_indices.append(pos_idx)

                # 建子圖
                from evaluate_retrieval_query_aware import build_query_subgraph
                subgraph, table_mapping = build_query_subgraph(
                    data, q_vec, candidate_indices,
                    edge_mode=edge_mode, device=device
                )

                # 子圖 forward
                x_dict_out = model.hetero_sage(subgraph.x_dict, subgraph.edge_index_dict)
                if 'table' in x_dict_out:
                    x_table = x_dict_out['table']
                    x_table = model.norm(x_table)
                    sub_table_emb = model.proj_head(x_table)
                    sub_table_emb = F.normalize(sub_table_emb, p=2, dim=1)
                else:
                    continue

                q_norm = F.normalize(q_vec, p=2, dim=1)
                rerank_scores = torch.matmul(q_norm, sub_table_emb.T).squeeze(0)
                _, reranked_local = torch.topk(rerank_scores, k=min(10, rerank_scores.size(0)))

                # 映射回原索引
                new_to_old = {v: k for k, v in table_mapping.items()}
                reranked_orig = [new_to_old[idx.item()] for idx in reranked_local]

                # 找最佳排名
                best_rank = float('inf')
                for rank, orig_idx in enumerate(reranked_orig, 1):
                    if orig_idx in pos_set:
                        best_rank = min(best_rank, rank)

                if best_rank < float('inf'):
                    mrr_sum += 1.0 / best_rank
                    for k in topk_vals:
                        if best_rank <= k:
                            hits[k] += 1

            return {
                'mrr': mrr_sum / num_queries,
                **{f'recall@{k}': hits[k] / num_queries for k in topk_vals}
            }


def main(dataset: str = None):
    if dataset is None:
        dataset = DATASETS[0]

    global KEY_FIELDS
    KEY_FIELDS = get_key_fields(dataset)

    graph_file = GRAPH_FILE.format(dataset=dataset)
    query_file = QUERY_FILE.format(dataset=dataset)
    val_graph_file = VAL_GRAPH_FILE.format(dataset=dataset)
    val_query_file = VAL_QUERY_FILE.format(dataset=dataset)
    save_path = SAVE_PATH.format(dataset=dataset)

    print(f"\n{'='*60}")
    print(f"Query-Aware 訓練 DATASET: {dataset}")
    print(f"KEY_FIELDS: {KEY_FIELDS}")
    print(f"Edge Mode: {QUERY_EDGE_MODE}")
    print(f"Subgraph K: {SUBGRAPH_K}")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的設備: {device}")

    param_combinations = [BEST_PARAMS]

    # 載入圖
    try:
        data_cpu = torch.load(graph_file, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"錯誤：找不到 {graph_file}，請先執行 build_graph.py。")
        return

    embed_dim = data_cpu['table'].x.size(1)

    try:
        id_to_idx = data_cpu.metadata_maps['table_id_to_idx']
    except (AttributeError, KeyError):
        print("錯誤：無法讀取映射表。")
        return

    # 載入訓練數據
    queries_text, pos_indices_lists, hard_neg_indices = load_training_data(
        query_file, id_to_idx, data=data_cpu, num_hard_negatives=NUM_HARD_NEGATIVES
    )
    if not queries_text:
        return

    embedder = get_embedder(model_name=MODEL_NAME, device=device)
    query_vectors = embed_queries(queries_text, embedder, device)

    # 載入驗證集數據
    print(f"\n從 {val_graph_file} 載入驗證集圖...")
    try:
        val_data_cpu = torch.load(val_graph_file, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"警告：找不到驗證集圖 {val_graph_file}，將不使用驗證集。")
        val_data_cpu = None
        val_query_vectors = None
        val_pos_indices_lists = None

    if val_data_cpu is not None:
        try:
            val_id_to_idx = val_data_cpu.metadata_maps['table_id_to_idx']
        except (AttributeError, KeyError):
            print("警告：無法讀取驗證集映射表。")
            val_data_cpu = None
            val_query_vectors = None
            val_pos_indices_lists = None

        if val_data_cpu is not None:
            val_queries_text, val_pos_indices_lists, _ = load_training_data(
                val_query_file, val_id_to_idx, data=val_data_cpu, num_hard_negatives=0
            )
            if val_queries_text:
                print(f"驗證集包含 {len(val_queries_text)} 個樣本。")
                val_query_vectors = embed_queries(val_queries_text, embedder, device)
            else:
                print("警告：驗證集查詢為空。")
                val_data_cpu = None
                val_query_vectors = None
                val_pos_indices_lists = None

    del embedder
    torch.cuda.empty_cache()

    # 訓練
    best_recall_global = -1.0
    best_params_global = None
    best_model_state_global = None

    base_config = {
        'NUM_EPOCHS': NUM_EPOCHS,
        'WARMUP_EPOCHS': WARMUP_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'CHUNK_SIZE': CHUNK_SIZE,
    }

    def objective(params):
        nonlocal best_recall_global, best_params_global, best_model_state_global

        print(f"\n[Training] Params: {params}")

        current_hps = {**base_config, **params, 'USE_HARD_NEG': True}

        data = data_cpu.clone().to(device)

        # 建立包含 query 節點的 metadata
        qa_metadata = get_query_aware_metadata(data, edge_mode=QUERY_EDGE_MODE)

        model, optimizer, scheduler, scaler = setup_components(embed_dim, qa_metadata, current_hps, device)
        current_hard_neg_indices = hard_neg_indices.copy()

        best_val_recall = -1.0
        best_epoch = 0
        patience_counter = 0
        best_model_state = None

        try:
            for epoch in range(1, current_hps['NUM_EPOCHS'] + 1):
                if epoch > 1 and epoch % REMINING_INTERVAL == 0:
                    current_hard_neg_indices = mine_hard_negatives_topk(
                        model, data, query_vectors, pos_indices_lists,
                        num_hard_negatives=NUM_HARD_NEGATIVES, device=device
                    )

                avg_loss, _, _ = train_one_epoch_query_aware(
                    model, data, optimizer, scaler, scheduler,
                    query_vectors, pos_indices_lists, current_hard_neg_indices,
                    current_hps, epoch, device,
                    edge_mode=QUERY_EDGE_MODE,
                    subgraph_k=SUBGRAPH_K,
                )

                # 驗證集評估
                if val_data_cpu is not None and val_query_vectors is not None:
                    val_data = val_data_cpu.clone().to(device)
                    val_metrics = evaluate_retrieval_query_aware(
                        model, val_data, val_query_vectors, val_pos_indices_lists,
                        current_hps, device,
                        edge_mode=QUERY_EDGE_MODE,
                        subgraph_k=SUBGRAPH_K,
                    )
                    val_recall = val_metrics['recall@10']

                    print(f"  Epoch {epoch}/{current_hps['NUM_EPOCHS']} | Loss: {avg_loss:.4f} | Val Recall@10: {val_recall:.4f} | Val MRR: {val_metrics['mrr']:.4f}")

                    if val_recall > best_val_recall:
                        best_val_recall = val_recall
                        best_epoch = epoch
                        patience_counter = 0
                        best_model_state = copy.deepcopy(model.state_dict())
                        print(f"    [新最佳驗證結果] Epoch {epoch} | Val Recall@10: {val_recall:.4f}")
                    else:
                        patience_counter += 1
                        print(f"    [未提升] 耐心計數: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

                    del val_data
                    torch.cuda.empty_cache()

                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        print(f"\n  [Early Stopping] 驗證集性能連續 {EARLY_STOPPING_PATIENCE} 個 epoch 未提升，停止訓練。")
                        print(f"  最佳 Epoch: {best_epoch} | 最佳 Val Recall@10: {best_val_recall:.4f}")
                        break
                else:
                    if epoch % 5 == 0 or epoch == current_hps['NUM_EPOCHS']:
                        print(f"  Epoch {epoch}/{current_hps['NUM_EPOCHS']} | Loss: {avg_loss:.4f}")

            # 使用最佳模型
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(f"\n已載入驗證集上最佳的模型（Epoch {best_epoch}）")

            # 最終評估（在訓練集上）
            metrics = evaluate_retrieval_query_aware(
                model, data, query_vectors, pos_indices_lists,
                current_hps, device,
                edge_mode=QUERY_EDGE_MODE,
                subgraph_k=SUBGRAPH_K,
            )
            train_recall = metrics['recall@10']
            print(f"  訓練集結果 -> Recall@10: {train_recall:.4f} | MRR: {metrics['mrr']:.4f}")

            # 驗證集最終評估
            if val_data_cpu is not None and val_query_vectors is not None:
                val_data = val_data_cpu.clone().to(device)
                val_metrics = evaluate_retrieval_query_aware(
                    model, val_data, val_query_vectors, val_pos_indices_lists,
                    current_hps, device,
                    edge_mode=QUERY_EDGE_MODE,
                    subgraph_k=SUBGRAPH_K,
                )
                print(f"  驗證集結果 -> Recall@10: {val_metrics['recall@10']:.4f} | MRR: {val_metrics['mrr']:.4f}")
                final_recall = val_metrics['recall@10']
                del val_data
            else:
                final_recall = train_recall

            # 記錄結果
            results_file = RESULTS_FILE
            Path(results_file).parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                row = [params.get(k) for k in ['LEARNING_RATE', 'HIDDEN_CHANNELS', 'DROPOUT', 'WEIGHT_DECAY']]
                row += [QUERY_EDGE_MODE, SUBGRAPH_K]
                row += [metrics['mrr'], metrics['recall@1'], metrics['recall@5'], final_recall]
                if best_epoch > 0:
                    row.append(best_epoch)
                writer.writerow(row)

            if final_recall > best_recall_global:
                best_recall_global = final_recall
                best_params_global = params
                best_model_state_global = copy.deepcopy(model.state_dict())
                print(f"  [更新最佳結果] Best Recall@10: {best_recall_global:.4f}")

            del model, optimizer, scheduler, scaler, data
            torch.cuda.empty_cache()

            return {'loss': -final_recall, 'status': STATUS_OK, 'metrics': metrics, 'best_epoch': best_epoch}

        except Exception as e:
            print(f"  [Error] Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'loss': 0.0, 'status': STATUS_OK}

    # 執行
    if USE_HYPEROPT:
        space = {
            'LEARNING_RATE': hp.loguniform('LEARNING_RATE', np.log(5e-5), np.log(1e-3)),
            'HIDDEN_CHANNELS': hp.choice('HIDDEN_CHANNELS', [128, 256, 512, 768, 1024]),
            'DROPOUT': hp.uniform('DROPOUT', 0.1, 0.5),
            'WEIGHT_DECAY': hp.loguniform('WEIGHT_DECAY', np.log(1e-3), np.log(5e-2)),
            'SAGE_AGGR': hp.choice('SAGE_AGGR', ['mean', 'max', 'sum', 'min']),
            'HETERO_AGGR': hp.choice('HETERO_AGGR', ['mean', 'max', 'sum', 'min']),
            'CLIP_GRAD_NORM': hp.uniform('CLIP_GRAD_NORM', 0.3, 1.0),
            'TEMP_START': hp.uniform('TEMP_START', 0.03, 0.1),
            'TEMP_END': hp.uniform('TEMP_END', 0.01, 0.05),
            'SMOOTH_START': hp.uniform('SMOOTH_START', 0.05, 0.2),
            'SMOOTH_END': hp.uniform('SMOOTH_END', 0.03, 0.1),
        }
        print(f"開始 Hyperopt 搜尋 (Max Evals: {MAX_EVALS})...")
        trials = Trials()
        best = fmin(objective, space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
    else:
        objective(BEST_PARAMS)

    if best_model_state_global is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': best_model_state_global,
            'hps': best_params_global,
            'best_recall': best_recall_global,
            'edge_mode': QUERY_EDGE_MODE,
            'subgraph_k': SUBGRAPH_K,
        }, save_path)
        print(f"模型已儲存至 {save_path}")


if __name__ == '__main__':
    for ds in DATASETS:
        graph_path = Path(f'/user_data/TabGNN/data/processed/train/{ds}/graph.pt')
        if not graph_path.exists():
            print(f"跳過 {ds}：找不到 {graph_path}")
            continue
        main(dataset=ds)