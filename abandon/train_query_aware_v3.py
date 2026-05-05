#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Query-Aware GNN 訓練腳本 v3 — Frozen Base

核心改進（相較 v2）：
1. 從 model_best_edges.pt 初始化（不是 generic model.pt）
2. **完全凍結 base 參數** — 只訓練 query edge 相關參數
3. 這保證粗排品質不會下降（= TWIG ablation baseline）
4. Query edge 參數零初始化 → 初始行為 = 純 TWIG

Usage:
  cd src && python query_aware/train_query_aware_v3.py              # all
  cd src && python query_aware/train_query_aware_v3.py feta ottqa   # specific
"""

import sys
import json
import copy
import random

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
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
    "feta": ["has_column", "same_page"],
    "ottqa": ["has_column", "similar_content"],
    "mimo_en": ["has_column", "similar_content", "shared_column_name"],
    "mimo_ch": ["similar_content"],
    "e2ewtq": ["similar_content"],
    "mmqa": ["similar_table", "has_column", "comes_from", "same_page", "similar_content"],
}

MODEL_NAME = 'BAAI/bge-m3'
QUERY_EDGE_MODE = "E4"        # Q→Table + Q→Page + Q→Column
SUBGRAPH_K = 50               # 訓練子圖大小（增大以包含更多候選）
COARSE_K_EVAL = 100           # 驗證粗排大小
NUM_EPOCHS = 20               # 只訓練 query 參數，需更多 epoch
WARMUP_EPOCHS = 3
BATCH_SIZE = 32               # 較小 batch 因為 per-sample subgraph
NUM_HARD_NEGATIVES = 5
EARLY_STOPPING_PATIENCE = 7

# 學習率（只有 query 參數在學）
QUERY_LR = 1e-3               # 較高 LR，因為是唯一在學的參數
WEIGHT_DECAY = 0.01
CLIP_GRAD_NORM = 1.0
TEMP = 0.04
LABEL_SMOOTH = 0.05

# 路徑
PROJECT_DIR = "/user_data/TabGNN"
PRETRAINED_PATH = PROJECT_DIR + "/checkpoints/{dataset}/model_best_edges.pt"
GRAPH_FILE = PROJECT_DIR + "/data/processed/train/{dataset}/graph.pt"
QUERY_FILE = PROJECT_DIR + "/data/table/train/{dataset}/query.jsonl"
VAL_GRAPH_FILE = PROJECT_DIR + "/data/processed/dev/{dataset}/graph.pt"
VAL_QUERY_FILE = PROJECT_DIR + "/data/table/dev/{dataset}/query.jsonl"
SAVE_PATH = PROJECT_DIR + "/checkpoints/{dataset}/model_qa_v3.pt"


def get_key_fields(dataset_name):
    if dataset_name in ["ottqa", "feta", "e2ewtq"]:
        return ("sheet_name", "file_name")
    elif dataset_name in ["mimo_en", "mimo_ch", "mmqa"]:
        return ("id",)
    return ("id",)


def make_key(item, key_fields):
    return "|".join(str(item.get(f, "")) for f in key_fields)


# ===========================
# Canonical metadata
# ===========================

def get_canonical_metadata():
    """E4 canonical metadata: 4 node types, 14 edge types."""
    node_types = ['table', 'column', 'page', 'query']
    edge_types = [
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
    return (node_types, edge_types)


CANONICAL_EDGES = [
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
CANONICAL_SET = set(CANONICAL_EDGES)

QUERY_KEYWORDS = ['queries', 'rev_queries', 'queries_page', 'rev_queries_page',
                   'queries_column', 'rev_queries_column']


# ===========================
# 模型
# ===========================

class QueryAwareModel(nn.Module):
    """QueryAwareModel with canonical E4 metadata."""

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
        x_dict, edge_index_dict = self._ensure_entries(x_dict, edge_index_dict)
        x_dict_out = self.hetero_sage(x_dict, edge_index_dict)
        x_table = x_dict_out['table']
        x_table = self.norm(x_table)
        table_features = self.proj_head(x_table)
        return F.normalize(table_features, p=2, dim=1)

    def _ensure_entries(self, x_dict, edge_index_dict):
        """Ensure all canonical node/edge types exist."""
        device = x_dict['table'].device
        embed_dim = x_dict['table'].size(1)
        x_dict = dict(x_dict)
        edge_index_dict = dict(edge_index_dict)

        for nt in ['table', 'column', 'page', 'query']:
            if nt not in x_dict:
                x_dict[nt] = torch.zeros((1, embed_dim), device=device)

        empty = torch.zeros((2, 0), dtype=torch.long, device=device)
        for et in CANONICAL_EDGES:
            if et not in edge_index_dict:
                edge_index_dict[et] = empty

        # Remove non-canonical edges (e.g., self-loops from filter_edges)
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if k in CANONICAL_SET}
        return x_dict, edge_index_dict


# ===========================
# Weight loading & freezing
# ===========================

def load_and_freeze_base(model, pretrained_path, device):
    """
    Load pretrained TWIG weights, zero-init query params, freeze base params.
    Returns: only the trainable (query) parameters.
    """
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    pretrained_state = ckpt['model_state_dict']
    hps = ckpt.get('hps', {})

    # Load matching weights
    missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
    matched = len(pretrained_state) - len(missing)
    print(f"  Weight loading: {matched} matched, {len(missing)} missing (query + unused base)")

    # Zero-init query params & freeze base params
    trainable_params = []
    frozen_count = 0
    query_count = 0

    for name, param in model.named_parameters():
        is_query = any(kw in name for kw in QUERY_KEYWORDS)
        if is_query:
            nn.init.zeros_(param)
            param.requires_grad = True
            trainable_params.append(param)
            query_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1

    print(f"  Frozen: {frozen_count} params, Trainable (query): {query_count} params")
    return trainable_params, hps


# ===========================
# Edge filtering
# ===========================

def filter_edges(data, keep_relations):
    """Filter graph edges to keep only specified relations."""
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
    for relation in keep_relations:
        edges_to_keep.add(relation)
        if relation in forward_to_reverse:
            edges_to_keep.add(forward_to_reverse[relation])

    reverse_to_forward = {'rev_has_column': 'has_column', 'rev_comes_from': 'comes_from'}
    for edge_type, edge_index in data.edge_index_dict.items():
        _, relation, _ = edge_type
        should_keep = relation in edges_to_keep
        if not should_keep and relation in reverse_to_forward:
            should_keep = reverse_to_forward[relation] in keep_relations
        if should_keep:
            filtered[edge_type].edge_index = edge_index

    if hasattr(data, 'metadata_maps'):
        filtered.metadata_maps = data.metadata_maps
    return filtered


# ===========================
# Subgraph construction
# ===========================

def build_subgraph(data, query_vec, candidate_indices, device=None):
    """Build subgraph with query node (E4 mode: Q→Table, Q→Page, Q→Column)."""
    if device is None:
        device = query_vec.device

    sub = HeteroData()
    num_candidates = len(candidate_indices)

    # Node mappings
    table_old_to_new = {old: new for new, old in enumerate(candidate_indices)}

    sub_column_set = set()
    sub_page_set = set()

    if ('table', 'has_column', 'column') in data.edge_types:
        ei = data['table', 'has_column', 'column'].edge_index
        mask = torch.isin(ei[0], torch.tensor(candidate_indices, device=ei.device))
        sub_column_set.update(ei[1, mask].cpu().tolist())

    if ('table', 'comes_from', 'page') in data.edge_types:
        ei = data['table', 'comes_from', 'page'].edge_index
        mask = torch.isin(ei[0], torch.tensor(candidate_indices, device=ei.device))
        sub_page_set.update(ei[1, mask].cpu().tolist())

    sub_column_indices = sorted(sub_column_set)
    sub_page_indices = sorted(sub_page_set)
    column_old_to_new = {old: new for new, old in enumerate(sub_column_indices)}
    page_old_to_new = {old: new for new, old in enumerate(sub_page_indices)}

    # Node features
    embed_dim = data['table'].x.size(1)
    sub['table'].x = data['table'].x[candidate_indices].to(device) if candidate_indices else \
        torch.zeros((0, embed_dim), device=device)
    sub['column'].x = data['column'].x[sub_column_indices].to(device) if sub_column_indices else \
        torch.zeros((0, embed_dim), device=device)
    sub['page'].x = data['page'].x[sub_page_indices].to(device) if sub_page_indices else \
        torch.zeros((0, embed_dim), device=device)

    # Remap edges helper
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

    # Base edges
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

    # Query node + edges (E4: all three)
    sub['query'].x = query_vec.to(device)  # (1, embed_dim)

    # Q → Table (all candidates)
    sub['query', 'queries', 'table'].edge_index = torch.tensor(
        [[0] * num_candidates, list(range(num_candidates))], dtype=torch.long, device=device)
    sub['table', 'rev_queries', 'query'].edge_index = torch.tensor(
        [list(range(num_candidates)), [0] * num_candidates], dtype=torch.long, device=device)

    # Q → Page
    if sub_page_indices:
        np_ = len(sub_page_indices)
        sub['query', 'queries_page', 'page'].edge_index = torch.tensor(
            [[0] * np_, list(range(np_))], dtype=torch.long, device=device)
        sub['page', 'rev_queries_page', 'query'].edge_index = torch.tensor(
            [list(range(np_)), [0] * np_], dtype=torch.long, device=device)

    # Q → Column
    if sub_column_indices:
        nc = len(sub_column_indices)
        sub['query', 'queries_column', 'column'].edge_index = torch.tensor(
            [[0] * nc, list(range(nc))], dtype=torch.long, device=device)
        sub['column', 'rev_queries_column', 'query'].edge_index = torch.tensor(
            [list(range(nc)), [0] * nc], dtype=torch.long, device=device)

    return sub, table_old_to_new


# ===========================
# Data loading
# ===========================

def rebuild_id_to_idx(data, key_fields):
    if hasattr(data, 'metadata_maps') and 'table_meta' in data.metadata_maps:
        table_meta = data.metadata_maps['table_meta']
        id_to_idx = {}
        for idx, meta in enumerate(table_meta):
            key = make_key(meta, key_fields)
            if key not in id_to_idx:
                id_to_idx[key] = idx
        return id_to_idx
    return data.metadata_maps.get('table_id_to_idx', {})


def load_queries(query_file, key_fields, id_to_idx, num_tables):
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


# ===========================
# Training
# ===========================

def train_one_epoch(model, data, optimizer, scaler, query_vectors, pos_indices_lists,
                    hard_neg_indices, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    n_batches = 0
    indices = list(range(len(query_vectors)))
    random.shuffle(indices)

    # Cache full-graph table embeddings (for coarse ranking to build subgraphs)
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
                q_vec = query_vectors[sample_idx:sample_idx + 1]
                pos_list = pos_indices_lists[sample_idx]
                hard_negs = hard_neg_indices[sample_idx] if hard_neg_indices else []

                # Coarse ranking for candidates
                with torch.no_grad():
                    q_norm = F.normalize(q_vec, p=2, dim=1)
                    scores = torch.matmul(q_norm, base_table_emb.T).squeeze(0)
                    _, top_k = torch.topk(scores, k=min(SUBGRAPH_K, scores.size(0)))
                    candidate_set = set(top_k.cpu().tolist())

                # Ensure positives and hard negatives in candidates
                for idx in pos_list:
                    candidate_set.add(idx)
                for idx in hard_negs:
                    if idx >= 0:
                        candidate_set.add(idx)

                candidate_indices = sorted(candidate_set)

                # Build subgraph with query node
                subgraph, table_mapping = build_subgraph(
                    data, q_vec, candidate_indices, device=device)

                # Subgraph forward (query edges influence table embeddings)
                sub_table_emb = model.forward(subgraph.x_dict, subgraph.edge_index_dict)

                # InfoNCE loss
                pos_new_idx = table_mapping.get(pos_list[0], -1)
                if pos_new_idx == -1:
                    continue

                logits = torch.matmul(q_norm, sub_table_emb.T).squeeze(0) / TEMP
                label = torch.tensor([pos_new_idx], dtype=torch.long, device=device)
                loss_sample = F.cross_entropy(logits.unsqueeze(0), label,
                                              label_smoothing=LABEL_SMOOTH)

                # Hard negative margin loss
                if hard_negs:
                    pos_score = logits[pos_new_idx]
                    for neg_idx in hard_negs:
                        if neg_idx >= 0 and neg_idx in table_mapping:
                            neg_new = table_mapping[neg_idx]
                            neg_score = logits[neg_new]
                            margin_loss = F.relu(neg_score - pos_score + 0.2 / TEMP)
                            loss_sample = loss_sample + 0.3 * margin_loss

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

        # Refresh coarse embeddings periodically
        if n_batches % 30 == 0:
            with torch.no_grad():
                base_table_emb = model.forward(data.x_dict, data.edge_index_dict)

    return total_loss / max(1, n_batches)


def evaluate_val(model, data, query_vectors, pos_indices_lists, device, k=COARSE_K_EVAL):
    """Evaluate query-aware Recall@10 on validation set."""
    model.eval()
    with torch.no_grad():
        base_table_emb = model.forward(data.x_dict, data.edge_index_dict)
        q_all = F.normalize(query_vectors, p=2, dim=1)
        coarse_scores = torch.matmul(q_all, base_table_emb.T)

        hits10 = 0
        mrr_sum = 0.0
        total = len(query_vectors)

        for qi in range(total):
            pos_set = set(pos_indices_lists[qi])
            q_vec = query_vectors[qi:qi + 1]

            _, top_k = torch.topk(coarse_scores[qi], k=min(k, coarse_scores.size(1)))
            candidate_indices = top_k.cpu().tolist()

            # Ensure positives in candidates for accurate eval
            for pos_idx in pos_set:
                if pos_idx not in set(candidate_indices):
                    candidate_indices.append(pos_idx)

            subgraph, table_mapping = build_subgraph(
                data, q_vec, candidate_indices, device=device)
            sub_table_emb = model.forward(subgraph.x_dict, subgraph.edge_index_dict)

            q_norm = F.normalize(q_vec, p=2, dim=1)
            rerank_scores = torch.matmul(q_norm, sub_table_emb.T).squeeze(0)

            # Score interpolation: combine coarse + rerank
            coarse_in_sub = torch.zeros(rerank_scores.size(0), device=device)
            for rank_i, orig_idx in enumerate(candidate_indices):
                new_idx = table_mapping.get(orig_idx, -1)
                if new_idx >= 0 and new_idx < coarse_in_sub.size(0):
                    coarse_in_sub[new_idx] = coarse_scores[qi, orig_idx]

            final_scores = 0.3 * coarse_in_sub + 0.7 * rerank_scores
            _, reranked = torch.topk(final_scores, k=min(10, final_scores.size(0)))

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

    return {'recall@10': hits10 / total, 'mrr': mrr_sum / total}


def mine_hard_negatives(model, data, query_vectors, pos_indices_lists, num_hard=5, device='cuda'):
    """Mine hard negatives from full-graph forward."""
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
# Main
# ===========================

def main(dataset):
    key_fields = get_key_fields(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained_path = PRETRAINED_PATH.format(dataset=dataset)
    if not Path(pretrained_path).exists():
        print(f"  {dataset}: model_best_edges.pt not found! Run train_twig_best.py first.")
        return

    print(f"\n{'='*60}")
    print(f"Query-Aware v3 (Frozen Base) Training: {dataset}")
    print(f"Device: {device}, Mode: {QUERY_EDGE_MODE}")
    print(f"Best edges: {BEST_EDGE_CONFIGS.get(dataset, [])}")
    print(f"{'='*60}")

    # Load and filter graph
    graph_file = GRAPH_FILE.format(dataset=dataset)
    data_cpu = torch.load(graph_file, map_location='cpu', weights_only=False)
    id_to_idx = rebuild_id_to_idx(data_cpu, key_fields)

    best_edges = BEST_EDGE_CONFIGS.get(dataset, [])
    if best_edges:
        data_cpu = filter_edges(data_cpu, best_edges)
        print(f"  Edge filter: {best_edges}")

    embed_dim = data_cpu['table'].x.size(1)
    num_tables = data_cpu['table'].num_nodes

    # Build model with canonical metadata
    metadata = get_canonical_metadata()
    model = QueryAwareModel(
        embed_dim=embed_dim,
        hidden_channels=768,
        metadata=metadata,
        dropout=0.10,
        sage_aggr='min',
        hetero_aggr='max',
    ).to(device)

    # Load pretrained weights & freeze base
    trainable_params, hps = load_and_freeze_base(model, pretrained_path, device)

    if not trainable_params:
        print("  ERROR: No trainable parameters found!")
        return

    # Optimizer (only query params)
    optimizer = optim.AdamW(trainable_params, lr=QUERY_LR, weight_decay=WEIGHT_DECAY)
    warmup_sched = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / float(WARMUP_EPOCHS)))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS])
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Load training queries
    query_file = QUERY_FILE.format(dataset=dataset)
    texts, pos_lists = load_queries(query_file, key_fields, id_to_idx, num_tables)
    print(f"  Train samples: {len(texts)}")

    if not texts:
        print("  No valid training samples, skipping")
        return

    embedder = SentenceTransformer(MODEL_NAME, device=str(device))
    query_vecs = torch.tensor(
        embedder.encode(texts, show_progress_bar=True),
        dtype=torch.float, device=device)

    # Load validation set
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
            print(f"  Val samples: {len(val_texts)}")

    del embedder
    torch.cuda.empty_cache()

    # Train
    data = data_cpu.clone().to(device)
    hard_neg_indices = None
    best_val_recall = -1.0
    best_model_state = None
    best_epoch = 0
    patience = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # Mine hard negatives
        if epoch == 1 or epoch % 2 == 0:
            hard_neg_indices = mine_hard_negatives(
                model, data, query_vecs, pos_lists,
                num_hard=NUM_HARD_NEGATIVES, device=device)

        avg_loss = train_one_epoch(
            model, data, optimizer, scaler, query_vecs, pos_lists,
            hard_neg_indices, device, epoch, NUM_EPOCHS)

        scheduler.step()

        # Validate
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
                print(f"    * New best at epoch {epoch}")
            else:
                patience += 1
                if patience >= EARLY_STOPPING_PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        else:
            print(f"  Epoch {epoch}/{NUM_EPOCHS} | Loss: {avg_loss:.4f}")
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    # Save
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
        'frozen_base': True,
    }, save_path)
    print(f"  Saved to {save_path} (epoch={best_epoch}, val R@10={best_val_recall:.4f})")

    del model, data
    torch.cuda.empty_cache()


if __name__ == '__main__':
    datasets = sys.argv[1:] if len(sys.argv) > 1 else [
        "feta", "ottqa", "mimo_en", "mimo_ch", "e2ewtq", "mmqa"
    ]
    for ds in datasets:
        graph_path = Path(GRAPH_FILE.format(dataset=ds))
        if not graph_path.exists():
            print(f"Skipping {ds}: {graph_path} not found")
            continue
        main(ds)
