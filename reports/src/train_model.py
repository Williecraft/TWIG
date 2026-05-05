#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    # "feta",
    # "e2ewtq",
     "mmqa",
]


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

SAVE_PATH = "/user_data/TabGNN/checkpoints/{dataset}/model.pt"
RESULTS_FILE = "/user_data/TabGNN/results/grid_search_results.csv"

MODEL_NAME = 'BAAI/bge-m3'
NUM_EPOCHS = 30
WARMUP_EPOCHS = 2
BATCH_SIZE = 128

# 訓練超參數（當 USE_HYPEROPT=False 時使用的預設值）
CLIP_GRAD_NORM = 0.60
CHUNK_SIZE = 1024
TEMP_START = 0.05
TEMP_END = 0.03
SMOOTH_START = 0.120
SMOOTH_END = 0.060

# Grid Search / Hyperopt 設定
USE_HYPEROPT = True  # True: 使用 Hyperopt | False: 使用 BEST_PARAMS
MAX_EVALS = 300     # Hyperopt 嘗試次數
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
REMINING_INTERVAL = 1  # 每幾個 epoch 重新挖掘困難負樣本

# Early Stopping 設定
EARLY_STOPPING_PATIENCE = 10  # 若驗證集性能連續 N 個 epoch 未提升則停止訓練
# ===========================


def make_key(item: dict, key_fields: tuple) -> str:
    """根據 key_fields 組合產生唯一鍵"""
    return "|".join(str(item.get(f, "")) for f in key_fields)


def get_embedder(model_name: str = MODEL_NAME, device: str = 'cuda') -> SentenceTransformer:
    """載入 SentenceTransformer 嵌入模型"""
    return SentenceTransformer(model_name, device=device)


class DiffusionModel(nn.Module):
    """異構圖 GNN 模型，使用 GraphSAGE 聚合節點資訊"""

    def __init__(self, embed_dim: int, hidden_channels: int, metadata, dropout: float = 0.2, 
                 sage_aggr: str = 'mean', hetero_aggr: str = 'sum'):
        super().__init__()
        self.sage = GraphSAGE(
            in_channels=embed_dim,
            hidden_channels=hidden_channels,
            num_layers=2,
            out_channels=hidden_channels,
            aggr=sage_aggr,  # 鄰居聚合方式
        )
        self.hetero_sage = to_hetero(self.sage, metadata, aggr=hetero_aggr)  # 跨邊類型聚合
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

    def score_tables(self, data: HeteroData, query_vec: torch.Tensor):
        """計算查詢向量與所有表格的相似度分數"""
        table_embeddings = self.forward(data.x_dict, data.edge_index_dict)
        query_vec_norm = F.normalize(query_vec.to(table_embeddings.device), p=2, dim=1)
        scores = torch.matmul(query_vec_norm, table_embeddings.T).squeeze(0)
        return scores


def load_training_data(query_file_path: str, id_to_idx: dict, data: HeteroData = None, num_hard_negatives: int = 1):
    """
    載入 training query 資料
    
    回傳:
      1. queries_text: List[str]
      2. pos_indices_lists: List[List[int]]
      3. hard_neg_indices: List[List[int]]
    """
    
    # 如果 data 有 table_meta，我們根據當前的 KEY_FIELDS 重建 id_to_idx
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

    # 建立鄰居索引 (用於困難負樣本挖掘)
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

            # 從 ground_truth_list 取得所有正確表格（以 file_name|sheet_name 為鍵）
            ground_truth_list = temp.get('ground_truth_list', []) or []
            pos_indices_list = []
            for gt in ground_truth_list:
                # 檢查所有 key fields 是否存在（允許空字串，但不允許 None）
                if all(gt.get(f) is not None for f in KEY_FIELDS):
                    key = make_key(gt, KEY_FIELDS)
                    idx = id_to_idx.get(key, -1)
                    if idx != -1:
                        pos_indices_list.append(idx)
            
            if not pos_indices_list:
                continue

            # 採樣困難負樣本（排除所有正確表格）
            all_neighbors = set()
            for pos_idx in pos_indices_list:
                all_neighbors.update(table_neighbors.get(pos_idx, set()))
            # 移除正確答案
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
                    # 儲存所有正確表格索引（第一個作為主要標籤，其餘用於評估）
                    training_samples.append((question, pos_indices_list, hard_negs))

    if not training_samples:
        print("警告：沒有載入任何有效的訓練樣本。")
        return [], [], []

    queries_text, pos_indices_lists, hard_neg_indices = [list(t) for t in zip(*training_samples)]

    # 驗證索引範圍
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
    """Query-Aware 困難負樣本挖掘：找出每個查詢最容易搞混的 K 張錯誤表格（支援多表格）"""
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
                # 排除所有正確表格
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
    model = DiffusionModel(
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


def train_one_epoch(model, data, optimizer, scaler, scheduler, query_vectors, pos_indices_lists, hard_neg_indices, hps, epoch, device):
    """執行一個訓練週期（支援多表格，使用第一個正確表格作為主要標籤）"""
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
        q_batch = query_vectors[batch_idx]
        # 使用第一個正確表格作為主要標籤
        labels = torch.tensor([pos_indices_lists[i][0] for i in batch_idx], dtype=torch.long, device=device)

        optimizer.zero_grad()

        with autocast(enabled=(device.type == 'cuda')):
            table_emb = model.forward(data.x_dict, data.edge_index_dict)
            q_batch_norm = F.normalize(q_batch, p=2, dim=1)
            logits = compute_scores_chunked(q_batch_norm, table_emb, hps['CHUNK_SIZE']) / curr_temp

            loss_in_batch = F.cross_entropy(logits, labels, label_smoothing=curr_smooth)

            # Hard Negative Margin Loss
            loss_hard = 0.0
            if hps.get('USE_HARD_NEG', False):
                batch_hard_negs = [hard_neg_indices[i] for i in batch_idx]
                hard_negs_tensor = torch.tensor(batch_hard_negs, device=device, dtype=torch.long)
                mask = (hard_negs_tensor != -1)
                safe_hard_negs = hard_negs_tensor.clone()
                safe_hard_negs[~mask] = 0

                pos_scores = logits[range(len(batch_idx)), labels]
                neg_scores = torch.gather(logits, 1, safe_hard_negs)

                margin = 0.2 / curr_temp
                losses = F.relu(neg_scores - pos_scores.unsqueeze(1) + margin) * mask.float()
                loss_hard = losses.sum() / (mask.sum() + 1e-9)

            loss = loss_in_batch + 0.5 * loss_hard

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hps['CLIP_GRAD_NORM'])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    scheduler.step()
    return total_loss / max(1, len(batch_iterator)), curr_temp, curr_smooth


def evaluate_retrieval(model, data, query_vectors, pos_indices_lists, hps, device):
    """評估模型的檢索性能（支援多表格，任一正確表格命中即算成功）"""
    model.eval()
    with torch.no_grad():
        table_emb = model.forward(data.x_dict, data.edge_index_dict)
        q_all_norm = F.normalize(query_vectors, p=2, dim=1)
        score_mat = compute_scores_chunked(q_all_norm, table_emb, hps['CHUNK_SIZE'])

        topk_vals = [1, 5, 10]
        hits = {k: 0 for k in topk_vals}
        mrr_sum = 0.0

        # 將分數矩陣移至 CPU 進行處理，避免在大量迴圈中頻繁存取 GPU 導致潛在錯誤
        score_mat_cpu = score_mat.cpu()
        num_queries = score_mat_cpu.size(0)

        for q_i in range(num_queries):
            # 在 CPU 上進行排序
            rank_indices = torch.argsort(score_mat_cpu[q_i], descending=True)
            pos_set = set(pos_indices_lists[q_i])
            
            # 找出所有正確表格中最高排名的那個
            best_rank = float('inf')
            
            # 優化：與其對每個 pos_idx 搜尋，不如只檢查存在的 ID
            # 但因為 num_tables 可能很大，且 pos_set 很小，直接搜尋 pos_idx 在 rank_indices 的位置可能不快
            # 替代方案：建立 id -> rank 映射? 對於單次查詢不划算。
            # 保持原邏輯但移至 CPU，通常 vector scan 在 CPU 也很快 (6k elements)
            
            for pos_idx in pos_set:
                # CPU 上的 nonzero
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


def main(dataset: str = None):
    if dataset is None:
        dataset = DATASETS[0]
    
    # 根據資料集設定 KEY_FIELDS
    global KEY_FIELDS
    KEY_FIELDS = get_key_fields(dataset)

    graph_file = GRAPH_FILE.format(dataset=dataset)
    query_file = QUERY_FILE.format(dataset=dataset)
    val_graph_file = VAL_GRAPH_FILE.format(dataset=dataset)
    val_query_file = VAL_QUERY_FILE.format(dataset=dataset)
    save_path = SAVE_PATH.format(dataset=dataset)

    print(f"\n{'='*60}")
    print(f"訓練 DATASET: {dataset}")
    print(f"KEY_FIELDS: {KEY_FIELDS}")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的設備: {device}")

    # 建立參數組合
    # if USE_GRID_SEARCH:
    #     PARAM_GRID = {
    #         'LEARNING_RATE': [5e-5, 1e-4, 3e-4, 5e-4, 1e-3],
    #         'HIDDEN_CHANNELS': [256, 512, 768],
    #         'DROPOUT': [0.1, 0.2, 0.3, 0.4, 0.5],
    #         'WEIGHT_DECAY': [1e-3, 1e-2, 5e-2]
    #     }
    #     keys, values = zip(*PARAM_GRID.items())
    #     param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # else:
    param_combinations = [BEST_PARAMS]
    keys = list(BEST_PARAMS.keys())

    print(f"總共 {len(param_combinations)} 組參數組合。")

    # 初始化結果 CSV
    with open(RESULTS_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        headers = ['LEARNING_RATE', 'HIDDEN_CHANNELS', 'DROPOUT', 'WEIGHT_DECAY', 'MRR', 'Recall@1', 'Recall@5', 'Recall@10', 'Best_Epoch']
        writer.writerow(headers)

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
    queries_text, pos_indices_lists, hard_neg_indices = load_training_data(query_file, id_to_idx, data=data_cpu, num_hard_negatives=NUM_HARD_NEGATIVES)
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

    # 定義 Hyperopt 目標函數
    best_recall_global = -1.0
    best_params_global = None
    best_model_state_global = None

    base_config = {
        'NUM_EPOCHS': NUM_EPOCHS,
        'WARMUP_EPOCHS': WARMUP_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'CHUNK_SIZE': CHUNK_SIZE,  # 保持固定，除非也想優化
    }

    def objective(params):
        nonlocal best_recall_global, best_params_global, best_model_state_global
        
        print(f"\n[Hyperopt] Testing params: {params}")
        
        current_hps = {**base_config, **params, 'USE_HARD_NEG': True}
        data = data_cpu.clone().to(device)
        model, optimizer, scheduler, scaler = setup_components(embed_dim, data.metadata(), current_hps, device)
        current_hard_neg_indices = hard_neg_indices.copy()
        
        # Early stopping 變量 (改為 Recall@10)
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

                avg_loss, _, _ = train_one_epoch(
                    model, data, optimizer, scaler, scheduler,
                    query_vectors, pos_indices_lists, current_hard_neg_indices, current_hps, epoch, device
                )

                # 驗證集評估
                if val_data_cpu is not None and val_query_vectors is not None:
                    val_data = val_data_cpu.clone().to(device)
                    val_metrics = evaluate_retrieval(
                        model, val_data, val_query_vectors, val_pos_indices_lists, current_hps, device
                    )
                    val_recall = val_metrics['recall@10']
                    
                    print(f"  Epoch {epoch}/{current_hps['NUM_EPOCHS']} | Loss: {avg_loss:.4f} | Val Recall@10: {val_recall:.4f} | Val MRR: {val_metrics['mrr']:.4f}")
                    
                    # 檢查是否為最佳驗證 Recall@10
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
                    
                    # Early Stopping 檢查
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        print(f"\n  [Early Stopping] 驗證集性能連續 {EARLY_STOPPING_PATIENCE} 個 epoch 未提升，停止訓練。")
                        print(f"  最佳 Epoch: {best_epoch} | 最佳 Val Recall@10: {best_val_recall:.4f}")
                        break
                else:
                    # 如果沒有驗證集，只打印訓練損失
                    if epoch % 5 == 0 or epoch == current_hps['NUM_EPOCHS']:
                        print(f"  Epoch {epoch}/{current_hps['NUM_EPOCHS']} | Loss: {avg_loss:.4f}")

            # 使用最佳模型（如果有驗證集）
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(f"\n已載入驗證集上最佳的模型（Epoch {best_epoch}）")
            
            # 最終評估（在訓練集上）
            metrics = evaluate_retrieval(model, data, query_vectors, pos_indices_lists, current_hps, device)
            train_recall = metrics['recall@10']
            print(f"  訓練集結果 -> Recall@10: {train_recall:.4f} | MRR: {metrics['mrr']:.4f}")
            
            # 最終評估（在驗證集上，如果有的話）
            if val_data_cpu is not None and val_query_vectors is not None:
                val_data = val_data_cpu.clone().to(device)
                val_metrics = evaluate_retrieval(
                    model, val_data, val_query_vectors, val_pos_indices_lists, current_hps, device
                )
                print(f"  驗證集結果 -> Recall@10: {val_metrics['recall@10']:.4f} | MRR: {val_metrics['mrr']:.4f}")
                final_recall = val_metrics['recall@10']  # 使用驗證集 Recall@10 作為最終指標
                del val_data
            else:
                final_recall = train_recall  # 如果沒有驗證集，使用訓練集 Recall@10

            # 記錄結果
            with open(RESULTS_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                row = [params.get(k) for k in ['LEARNING_RATE', 'HIDDEN_CHANNELS', 'DROPOUT', 'WEIGHT_DECAY']]
                row += [metrics['mrr'], metrics['recall@1'], metrics['recall@5'], final_recall]
                if best_epoch > 0:
                    row.append(best_epoch)  # 添加最佳 epoch
                writer.writerow(row)

            if final_recall > best_recall_global:
                best_recall_global = final_recall
                best_params_global = params
                best_model_state_global = copy.deepcopy(model.state_dict())
                print(f"  [更新最佳結果] Best Recall@10: {best_recall_global:.4f} (Epoch {best_epoch if best_epoch > 0 else current_hps['NUM_EPOCHS']})")

            # 清理
            del model, optimizer, scheduler, scaler, data
            torch.cuda.empty_cache()
            
            return {'loss': -final_recall, 'status': STATUS_OK, 'metrics': metrics, 'best_epoch': best_epoch}

        except Exception as e:
            print(f"  [Error] Training failed with params {params}: {e}")
            import traceback
            traceback.print_exc()
            return {'loss': 0.0, 'status': STATUS_OK} # Return 0 loss

    # 執行搜尋或單次執行
    if USE_HYPEROPT:
        space = {
            'LEARNING_RATE': hp.loguniform('LEARNING_RATE', np.log(5e-5), np.log(1e-3)),
            'HIDDEN_CHANNELS': hp.choice('HIDDEN_CHANNELS', [128, 256, 512, 768, 1024]),
            'DROPOUT': hp.uniform('DROPOUT', 0.1, 0.5),
            'WEIGHT_DECAY': hp.loguniform('WEIGHT_DECAY', np.log(1e-3), np.log(5e-2)),
            'SAGE_AGGR': hp.choice('SAGE_AGGR', ['mean', 'max', 'sum', 'min']),
            'HETERO_AGGR': hp.choice('HETERO_AGGR', ['mean', 'max', 'sum', 'min']),
            # 訓練超參數
            'CLIP_GRAD_NORM': hp.uniform('CLIP_GRAD_NORM', 0.3, 1.0),
            'TEMP_START': hp.uniform('TEMP_START', 0.03, 0.1),
            'TEMP_END': hp.uniform('TEMP_END', 0.01, 0.05),
            'SMOOTH_START': hp.uniform('SMOOTH_START', 0.05, 0.2),
            'SMOOTH_END': hp.uniform('SMOOTH_END', 0.03, 0.1),
        }
        
        print(f"開始 Hyperopt 搜尋 (Max Evals: {MAX_EVALS})...")
        trials = Trials()
        best = fmin(objective, space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
        
        print(f"\n========================================")
        print(f"Hyperopt 完成！最佳 Recall@10: {best_recall_global:.4f}")
        print(f"最佳參數: {best_params_global}")
        print("========================================")
        
    else:
        # 單次執行 Best Params
        objective(BEST_PARAMS)

    if best_model_state_global is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': best_model_state_global,
            'hps': best_params_global,
            'best_recall': best_recall_global
        }, save_path)
        print(f"模型已儲存至 {save_path}")


if __name__ == '__main__':
    for ds in DATASETS:
        graph_path = Path(f'/user_data/TabGNN/data/processed/train/{ds}/graph.pt')
        if not graph_path.exists():
            print(f"跳過 {ds}：找不到 {graph_path}")
            continue
        main(dataset=ds)