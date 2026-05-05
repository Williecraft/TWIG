#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fair Comparison: TWIG (best edge config) vs Query-Aware v2 (same edge config)

For each dataset:
1. Train TWIG with the best ablation edge config → save checkpoint
2. Train QA v2 using that checkpoint + same edges → save checkpoint
3. Evaluate both on test set → compare

This isolates the single variable: does adding a query node help?

Usage:
  cd src && python run_fair_comparison.py
  cd src && python run_fair_comparison.py --datasets feta ottqa
"""

import sys
import os
import json
import copy
import random
import argparse
from pathlib import Path

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

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ===========================
# Config
# ===========================

PROJECT_DIR = "/user_data/TabGNN"
MODEL_NAME = 'BAAI/bge-m3'

# TWIG best ablation edge configs (from results/edge_ablation_extended/)
TWIG_BEST_EDGES = {
    "feta":    ["has_column", "comes_from"],          # A24, R@10=0.9900
    "ottqa":   ["has_column"],                        # A16, R@10=0.9874
    "mimo_en": ["comes_from"],                        # A8,  R@10=0.7392
    "mimo_ch": ["same_page"],                         # A4,  R@10=0.7012
    "e2ewtq":  ["similar_content"],                   # A2,  R@10=0.9180
    "mmqa":    [],                                    # A0,  R@10=0.7802
}

# TWIG training HPs (same as run_edge_ablation.py)
TWIG_HPS = {
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
    'NUM_EPOCHS': 30,
    'WARMUP_EPOCHS': 2,
    'BATCH_SIZE': 128,
    'CHUNK_SIZE': 1024,
}

# QA v2 training HPs
QA_NUM_EPOCHS = 15
QA_WARMUP = 2
QA_BASE_LR = 1e-4
QA_QUERY_LR = 5e-4
QA_WEIGHT_DECAY = 0.03
QA_BATCH_SIZE = 64
QA_SUBGRAPH_K = 30
QA_COARSE_K_EVAL = 50
QA_TEMP = 0.04
QA_LABEL_SMOOTH = 0.08
QA_CLIP_GRAD = 0.60
QA_NUM_HARD_NEG = 3
QA_PATIENCE = 5

# Paths
GRAPH_TRAIN = PROJECT_DIR + "/data/processed/train/{dataset}/graph.pt"
GRAPH_DEV = PROJECT_DIR + "/data/processed/dev/{dataset}/graph.pt"
GRAPH_TEST = PROJECT_DIR + "/data/processed/test/{dataset}/graph.pt"
QUERY_TRAIN = PROJECT_DIR + "/data/table/train/{dataset}/query.jsonl"
QUERY_DEV = PROJECT_DIR + "/data/table/dev/{dataset}/query.jsonl"
QUERY_TEST = PROJECT_DIR + "/data/table/test/{dataset}/query.jsonl"
TWIG_SAVE = PROJECT_DIR + "/checkpoints/{dataset}/model_fair_twig.pt"
QA_SAVE = PROJECT_DIR + "/checkpoints/{dataset}/model_fair_qa.pt"
RESULTS_DIR = PROJECT_DIR + "/results/fair_comparison"


# ===========================
# Shared utilities
# ===========================

def get_key_fields(ds):
    if ds in ["ottqa", "feta", "e2ewtq"]:
        return ("sheet_name", "file_name")
    return ("id",)

def make_key(item, key_fields):
    return "|".join(str(item.get(f, "")) for f in key_fields)

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


def filter_edges(data, keep_relations):
    """Filter graph to keep only specified edge types."""
    if not keep_relations:
        # A0: no edges — keep nodes, remove all edges
        filtered = HeteroData()
        for node_type in data.node_types:
            for attr_name in data[node_type].keys():
                filtered[node_type][attr_name] = data[node_type][attr_name]
        # Add self-loops so each node type has at least one edge
        for node_type in filtered.node_types:
            filtered[node_type, f'_self_loop_{node_type}', node_type].edge_index = \
                torch.tensor([[0], [0]], dtype=torch.long)
        if hasattr(data, 'metadata_maps'):
            filtered.metadata_maps = data.metadata_maps
        return filtered

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

    # Self-loops for orphan node types
    if len(filtered.edge_types) > 0:
        dest_types = {dst for _, _, dst in filtered.edge_types}
        for node_type in filtered.node_types:
            if node_type not in dest_types:
                filtered[node_type, f'_self_loop_{node_type}', node_type].edge_index = \
                    torch.tensor([[0], [0]], dtype=torch.long)

    if hasattr(data, 'metadata_maps'):
        filtered.metadata_maps = data.metadata_maps
    return filtered


def filter_edges_qa(data, keep_relations):
    """Filter for QA model (no self-loops, _ensure_query_entries handles it)."""
    if not keep_relations:
        filtered = HeteroData()
        for node_type in data.node_types:
            for attr_name in data[node_type].keys():
                filtered[node_type][attr_name] = data[node_type][attr_name]
        if hasattr(data, 'metadata_maps'):
            filtered.metadata_maps = data.metadata_maps
        return filtered

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

    if hasattr(data, 'metadata_maps'):
        filtered.metadata_maps = data.metadata_maps
    return filtered


# ===========================
# TWIG DiffusionModel
# ===========================

class DiffusionModel(nn.Module):
    def __init__(self, embed_dim, hidden_channels, metadata, dropout=0.2,
                 sage_aggr='mean', hetero_aggr='sum'):
        super().__init__()
        self.sage = GraphSAGE(
            in_channels=embed_dim, hidden_channels=hidden_channels,
            num_layers=2, out_channels=hidden_channels, aggr=sage_aggr,
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
        x_dict_out = self.hetero_sage(x_dict, edge_index_dict)
        x_table = x_dict_out['table']
        x_table = self.norm(x_table)
        table_features = self.proj_head(x_table)
        return F.normalize(table_features, p=2, dim=1)


# ===========================
# QueryAwareModel
# ===========================

def get_canonical_metadata():
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


class QueryAwareModel(nn.Module):
    def __init__(self, embed_dim, hidden_channels, metadata, dropout=0.2,
                 sage_aggr='mean', hetero_aggr='sum'):
        super().__init__()
        self.sage = GraphSAGE(
            in_channels=embed_dim, hidden_channels=hidden_channels,
            num_layers=2, out_channels=hidden_channels, aggr=sage_aggr,
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
        x_dict, edge_index_dict = self._ensure(x_dict, edge_index_dict)
        x_dict_out = self.hetero_sage(x_dict, edge_index_dict)
        x_table = x_dict_out['table']
        x_table = self.norm(x_table)
        table_features = self.proj_head(x_table)
        return F.normalize(table_features, p=2, dim=1)

    def _ensure(self, x_dict, edge_index_dict):
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
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if k in CANONICAL_SET}
        return x_dict, edge_index_dict


def build_subgraph(data, query_vec, candidate_indices, device=None):
    if device is None:
        device = query_vec.device
    sub = HeteroData()
    sub_table_indices = list(candidate_indices)
    table_old_to_new = {old: new for new, old in enumerate(sub_table_indices)}
    num_candidates = len(candidate_indices)

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

    sub['table'].x = data['table'].x[sub_table_indices].to(device) if sub_table_indices else torch.zeros((0, data['table'].x.size(1)), device=device)
    sub['column'].x = data['column'].x[sub_column_indices].to(device) if sub_column_indices else torch.zeros((0, data['table'].x.size(1)), device=device)
    sub['page'].x = data['page'].x[sub_page_indices].to(device) if sub_page_indices else torch.zeros((0, data['table'].x.size(1)), device=device)

    def remap_edges(edge_type, src_map, dst_map):
        if edge_type not in data.edge_types:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        ei = data[edge_type].edge_index
        src_tensor = torch.tensor(list(src_map.keys()), device=ei.device)
        dst_tensor = torch.tensor(list(dst_map.keys()), device=ei.device)
        mask = torch.isin(ei[0], src_tensor) & torch.isin(ei[1], dst_tensor)
        if mask.sum() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        new_src = [src_map[s] for s in ei[0, mask].cpu().tolist()]
        new_dst = [dst_map[d] for d in ei[1, mask].cpu().tolist()]
        return torch.tensor([new_src, new_dst], dtype=torch.long, device=device)

    sub['table', 'has_column', 'column'].edge_index = remap_edges(('table', 'has_column', 'column'), table_old_to_new, column_old_to_new)
    sub['column', 'rev_has_column', 'table'].edge_index = remap_edges(('column', 'rev_has_column', 'table'), column_old_to_new, table_old_to_new)
    sub['table', 'comes_from', 'page'].edge_index = remap_edges(('table', 'comes_from', 'page'), table_old_to_new, page_old_to_new)
    sub['page', 'rev_comes_from', 'table'].edge_index = remap_edges(('page', 'rev_comes_from', 'table'), page_old_to_new, table_old_to_new)
    for rel in ['same_page', 'similar_table', 'shared_column_name']:
        sub['table', rel, 'table'].edge_index = remap_edges(('table', rel, 'table'), table_old_to_new, table_old_to_new)
    sub['column', 'similar_content', 'column'].edge_index = remap_edges(('column', 'similar_content', 'column'), column_old_to_new, column_old_to_new)

    # Query node + edges
    sub['query'].x = query_vec.to(device)
    qi = 0
    sub['query', 'queries', 'table'].edge_index = torch.tensor(
        [[qi] * num_candidates, list(range(num_candidates))], dtype=torch.long, device=device)
    sub['table', 'rev_queries', 'query'].edge_index = torch.tensor(
        [list(range(num_candidates)), [qi] * num_candidates], dtype=torch.long, device=device)
    if sub_page_indices:
        np_ = len(sub_page_indices)
        sub['query', 'queries_page', 'page'].edge_index = torch.tensor(
            [[qi] * np_, list(range(np_))], dtype=torch.long, device=device)
        sub['page', 'rev_queries_page', 'query'].edge_index = torch.tensor(
            [list(range(np_)), [qi] * np_], dtype=torch.long, device=device)
    if sub_column_indices:
        nc = len(sub_column_indices)
        sub['query', 'queries_column', 'column'].edge_index = torch.tensor(
            [[qi] * nc, list(range(nc))], dtype=torch.long, device=device)
        sub['column', 'rev_queries_column', 'query'].edge_index = torch.tensor(
            [list(range(nc)), [qi] * nc], dtype=torch.long, device=device)

    return sub, table_old_to_new


# ===========================
# Phase 1: Train TWIG with best edge config
# ===========================

def train_twig(dataset, device):
    key_fields = get_key_fields(dataset)
    best_edges = TWIG_BEST_EDGES[dataset]
    save_path = TWIG_SAVE.format(dataset=dataset)

    print(f"\n{'='*60}")
    print(f"Phase 1: Train TWIG | {dataset} | edges: {best_edges or 'none'}")
    print(f"{'='*60}")

    data_cpu = torch.load(GRAPH_TRAIN.format(dataset=dataset), map_location='cpu', weights_only=False)
    data_cpu = filter_edges(data_cpu, best_edges)
    id_to_idx = rebuild_id_to_idx(data_cpu, key_fields)
    embed_dim = data_cpu['table'].x.size(1)
    num_tables = data_cpu['table'].num_nodes

    texts, pos_lists = load_queries(QUERY_TRAIN.format(dataset=dataset), key_fields, id_to_idx, num_tables)
    print(f"  Train samples: {len(texts)}")
    if not texts:
        print("  No valid samples, skip")
        return None

    data = data_cpu.clone().to(device)
    model = DiffusionModel(
        embed_dim=embed_dim, hidden_channels=TWIG_HPS['HIDDEN_CHANNELS'],
        metadata=data.metadata(), dropout=TWIG_HPS['DROPOUT'],
        sage_aggr=TWIG_HPS['SAGE_AGGR'], hetero_aggr=TWIG_HPS['HETERO_AGGR'],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=TWIG_HPS['LEARNING_RATE'],
                            weight_decay=TWIG_HPS['WEIGHT_DECAY'])
    warmup = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / float(TWIG_HPS['WARMUP_EPOCHS'])))
    cosine = CosineAnnealingLR(optimizer, T_max=TWIG_HPS['NUM_EPOCHS'] - TWIG_HPS['WARMUP_EPOCHS'])
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[TWIG_HPS['WARMUP_EPOCHS']])
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    embedder = SentenceTransformer(MODEL_NAME, device=str(device))
    query_vecs = torch.tensor(embedder.encode(texts, show_progress_bar=True), dtype=torch.float, device=device)

    # Val data
    val_data, val_vecs, val_pos = None, None, None
    val_graph_path = GRAPH_DEV.format(dataset=dataset)
    val_query_path = QUERY_DEV.format(dataset=dataset)
    if Path(val_graph_path).exists() and Path(val_query_path).exists():
        val_data_cpu = torch.load(val_graph_path, map_location='cpu', weights_only=False)
        val_data_cpu = filter_edges(val_data_cpu, best_edges)
        val_id = rebuild_id_to_idx(val_data_cpu, key_fields)
        val_texts, val_pos = load_queries(val_query_path, key_fields, val_id, val_data_cpu['table'].num_nodes)
        if val_texts:
            val_vecs = torch.tensor(embedder.encode(val_texts, show_progress_bar=True), dtype=torch.float, device=device)
            val_data = val_data_cpu.clone().to(device)
            print(f"  Val samples: {len(val_texts)}")

    del embedder
    torch.cuda.empty_cache()

    # Hard negatives
    hard_neg_indices = None
    best_val_recall = -1.0
    best_model_state = None
    best_epoch = 0
    patience = 0

    for epoch in range(1, TWIG_HPS['NUM_EPOCHS'] + 1):
        # Mine hard negatives
        if epoch == 1 or epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                table_emb = model.forward(data.x_dict, data.edge_index_dict)
                q_norm = F.normalize(query_vecs, p=2, dim=1)
                hard_neg_indices = []
                for start in range(0, len(query_vecs), 1024):
                    end = min(start + 1024, len(query_vecs))
                    sim = torch.matmul(q_norm[start:end], table_emb.T)
                    for i in range(end - start):
                        for pos_idx in pos_lists[start + i]:
                            if 0 <= pos_idx < num_tables:
                                sim[i, pos_idx] = -float('inf')
                    _, topk = torch.topk(sim, k=min(3, num_tables - 1), dim=1)
                    hard_neg_indices.extend(topk.tolist())

        # Train
        model.train()
        total_loss = 0.0
        indices = list(range(len(query_vecs)))
        random.shuffle(indices)
        progress = epoch / float(TWIG_HPS['NUM_EPOCHS'])
        curr_temp = TWIG_HPS['TEMP_END'] if epoch > TWIG_HPS['NUM_EPOCHS'] * 0.7 else TWIG_HPS['TEMP_START']
        curr_smooth = TWIG_HPS['SMOOTH_START'] + (TWIG_HPS['SMOOTH_END'] - TWIG_HPS['SMOOTH_START']) * progress

        for start in range(0, len(indices), TWIG_HPS['BATCH_SIZE']):
            end = min(start + TWIG_HPS['BATCH_SIZE'], len(indices))
            batch_idx = indices[start:end]
            q_batch = query_vecs[batch_idx]
            labels = torch.tensor([pos_lists[i][0] for i in batch_idx], dtype=torch.long, device=device)
            optimizer.zero_grad()
            with autocast(enabled=(device.type == 'cuda')):
                table_emb = model.forward(data.x_dict, data.edge_index_dict)
                q_norm = F.normalize(q_batch, p=2, dim=1)
                logits = torch.matmul(q_norm, table_emb.T) / curr_temp
                loss = F.cross_entropy(logits, labels, label_smoothing=curr_smooth)
                # Hard negative margin loss
                if hard_neg_indices:
                    batch_hard = [hard_neg_indices[i] for i in batch_idx]
                    hard_t = torch.tensor(batch_hard, device=device, dtype=torch.long)
                    mask = (hard_t != -1)
                    safe = hard_t.clone(); safe[~mask] = 0
                    pos_scores = logits[range(len(batch_idx)), labels]
                    neg_scores = torch.gather(logits, 1, safe)
                    margin = 0.2 / curr_temp
                    hl = F.relu(neg_scores - pos_scores.unsqueeze(1) + margin) * mask.float()
                    loss = loss + 0.5 * hl.sum() / (mask.sum() + 1e-9)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TWIG_HPS['CLIP_GRAD_NORM'])
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(1, len(range(0, len(indices), TWIG_HPS['BATCH_SIZE'])))

        # Validate
        if val_data is not None and val_vecs is not None:
            model.eval()
            with torch.no_grad():
                t_emb = model.forward(val_data.x_dict, val_data.edge_index_dict)
                q_n = F.normalize(val_vecs, p=2, dim=1)
                scores = torch.matmul(q_n, t_emb.T)
                hits10 = 0
                for qi in range(len(val_vecs)):
                    _, topk = torch.topk(scores[qi], k=min(10, scores.size(1)))
                    topk_set = set(topk.cpu().tolist())
                    if any(p in topk_set for p in val_pos[qi]):
                        hits10 += 1
                val_r10 = hits10 / len(val_vecs)

            print(f"  Epoch {epoch}/{TWIG_HPS['NUM_EPOCHS']} | Loss: {avg_loss:.4f} | Val R@10: {val_r10:.4f}")
            if val_r10 > best_val_recall:
                best_val_recall = val_r10
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        else:
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}/{TWIG_HPS['NUM_EPOCHS']} | Loss: {avg_loss:.4f}")
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'hps': TWIG_HPS,
        'best_edges': best_edges,
        'best_epoch': best_epoch,
    }, save_path)
    print(f"  TWIG saved: {save_path} (epoch {best_epoch}, val R@10={best_val_recall:.4f})")

    del model, data
    if val_data is not None:
        del val_data
    torch.cuda.empty_cache()
    return save_path


# ===========================
# Phase 2: Train QA v2 from TWIG checkpoint
# ===========================

def train_qa(dataset, twig_ckpt_path, device):
    key_fields = get_key_fields(dataset)
    best_edges = TWIG_BEST_EDGES[dataset]
    save_path = QA_SAVE.format(dataset=dataset)

    print(f"\n{'='*60}")
    print(f"Phase 2: Train QA v2 | {dataset} | edges: {best_edges or 'none'}")
    print(f"{'='*60}")

    data_cpu = torch.load(GRAPH_TRAIN.format(dataset=dataset), map_location='cpu', weights_only=False)
    data_cpu = filter_edges_qa(data_cpu, best_edges)
    id_to_idx = rebuild_id_to_idx(data_cpu, key_fields)
    embed_dim = data_cpu['table'].x.size(1)
    num_tables = data_cpu['table'].num_nodes

    texts, pos_lists = load_queries(QUERY_TRAIN.format(dataset=dataset), key_fields, id_to_idx, num_tables)
    if not texts:
        print("  No valid samples, skip")
        return None

    metadata = get_canonical_metadata()
    model = QueryAwareModel(
        embed_dim=embed_dim, hidden_channels=TWIG_HPS['HIDDEN_CHANNELS'],
        metadata=metadata, dropout=TWIG_HPS['DROPOUT'],
        sage_aggr=TWIG_HPS['SAGE_AGGR'], hetero_aggr=TWIG_HPS['HETERO_AGGR'],
    ).to(device)

    # Load TWIG pretrained weights + zero-init query params
    ckpt = torch.load(twig_ckpt_path, map_location=device, weights_only=False)
    missing, _ = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    query_keywords = ['queries', 'rev_queries', 'queries_page', 'rev_queries_page',
                      'queries_column', 'rev_queries_column']
    zero_count = 0
    for name, param in model.named_parameters():
        if any(kw in name for kw in query_keywords):
            nn.init.zeros_(param)
            zero_count += 1
    print(f"  Loaded TWIG weights, zero-init {zero_count} query params")

    # Differential LR
    base_params, query_params = [], []
    for name, param in model.named_parameters():
        if any(kw in name for kw in query_keywords):
            query_params.append(param)
        else:
            base_params.append(param)

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': QA_BASE_LR},
        {'params': query_params, 'lr': QA_QUERY_LR},
    ], weight_decay=QA_WEIGHT_DECAY)

    warmup = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / float(QA_WARMUP)))
    cosine = CosineAnnealingLR(optimizer, T_max=QA_NUM_EPOCHS - QA_WARMUP)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[QA_WARMUP])
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    embedder = SentenceTransformer(MODEL_NAME, device=str(device))
    query_vecs = torch.tensor(embedder.encode(texts, show_progress_bar=True), dtype=torch.float, device=device)

    # Val data
    val_data_cpu, val_vecs, val_pos = None, None, None
    val_graph_path = GRAPH_DEV.format(dataset=dataset)
    val_query_path = QUERY_DEV.format(dataset=dataset)
    if Path(val_graph_path).exists() and Path(val_query_path).exists():
        val_data_cpu = torch.load(val_graph_path, map_location='cpu', weights_only=False)
        val_data_cpu = filter_edges_qa(val_data_cpu, best_edges)
        val_id = rebuild_id_to_idx(val_data_cpu, key_fields)
        val_texts, val_pos = load_queries(val_query_path, key_fields, val_id, val_data_cpu['table'].num_nodes)
        if val_texts:
            val_vecs = torch.tensor(embedder.encode(val_texts, show_progress_bar=True), dtype=torch.float, device=device)
            print(f"  Val samples: {len(val_texts)}")

    del embedder
    torch.cuda.empty_cache()

    data = data_cpu.clone().to(device)
    hard_neg_indices = None
    best_val_recall = -1.0
    best_model_state = None
    best_epoch = 0
    patience = 0

    for epoch in range(1, QA_NUM_EPOCHS + 1):
        # Hard negative mining
        if epoch == 1 or epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                table_emb = model.forward(data.x_dict, data.edge_index_dict)
                q_norm = F.normalize(query_vecs, p=2, dim=1)
                hard_neg_indices = []
                for start in range(0, len(query_vecs), 1024):
                    end = min(start + 1024, len(query_vecs))
                    sim = torch.matmul(q_norm[start:end], table_emb.T)
                    for i in range(end - start):
                        for pos_idx in pos_lists[start + i]:
                            if 0 <= pos_idx < num_tables:
                                sim[i, pos_idx] = -float('inf')
                    _, topk = torch.topk(sim, k=min(QA_NUM_HARD_NEG, num_tables - 1), dim=1)
                    hard_neg_indices.extend(topk.tolist())

        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0
        indices = list(range(len(query_vecs)))
        random.shuffle(indices)

        with torch.no_grad():
            base_table_emb = model.forward(data.x_dict, data.edge_index_dict)

        for start in range(0, len(indices), QA_BATCH_SIZE):
            end = min(start + QA_BATCH_SIZE, len(indices))
            batch_idx = indices[start:end]
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_count = 0

            with autocast(enabled=(device.type == 'cuda')):
                for si in batch_idx:
                    q_vec = query_vecs[si:si + 1]
                    pos_list = pos_lists[si]
                    hard_negs = hard_neg_indices[si] if hard_neg_indices else []

                    with torch.no_grad():
                        q_n = F.normalize(q_vec, p=2, dim=1)
                        scores = torch.matmul(q_n, base_table_emb.T).squeeze(0)
                        _, top_k = torch.topk(scores, k=min(QA_SUBGRAPH_K, scores.size(0)))
                        cand_set = set(top_k.cpu().tolist())

                    for idx in pos_list:
                        cand_set.add(idx)
                    for idx in hard_negs:
                        if idx >= 0:
                            cand_set.add(idx)

                    cand_indices = sorted(cand_set)
                    subgraph, table_mapping = build_subgraph(data, q_vec, cand_indices, device=device)
                    sub_emb = model.forward(subgraph.x_dict, subgraph.edge_index_dict)

                    pos_new = table_mapping.get(pos_list[0], -1)
                    if pos_new == -1:
                        continue

                    logits = torch.matmul(q_n, sub_emb.T).squeeze(0) / QA_TEMP
                    label = torch.tensor([pos_new], dtype=torch.long, device=device)
                    loss_s = F.cross_entropy(logits.unsqueeze(0), label, label_smoothing=QA_LABEL_SMOOTH)
                    batch_loss += loss_s
                    batch_count += 1

                if batch_count == 0:
                    continue
                loss = batch_loss / batch_count

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=QA_CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

            if n_batches % 20 == 0:
                with torch.no_grad():
                    base_table_emb = model.forward(data.x_dict, data.edge_index_dict)

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)

        # Validate
        if val_data_cpu is not None and val_vecs is not None:
            val_data = val_data_cpu.clone().to(device)
            model.eval()
            with torch.no_grad():
                vt_emb = model.forward(val_data.x_dict, val_data.edge_index_dict)
                vq_n = F.normalize(val_vecs, p=2, dim=1)
                v_scores = torch.matmul(vq_n, vt_emb.T)
                hits10 = 0
                for qi in range(len(val_vecs)):
                    _, topk = torch.topk(v_scores[qi], k=min(10, v_scores.size(1)))
                    if any(p in set(topk.cpu().tolist()) for p in val_pos[qi]):
                        hits10 += 1
                val_r10 = hits10 / len(val_vecs)
            del val_data
            torch.cuda.empty_cache()

            print(f"  Epoch {epoch}/{QA_NUM_EPOCHS} | Loss: {avg_loss:.4f} | Val R@10: {val_r10:.4f}")
            if val_r10 > best_val_recall:
                best_val_recall = val_r10
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= QA_PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        else:
            print(f"  Epoch {epoch}/{QA_NUM_EPOCHS} | Loss: {avg_loss:.4f}")
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'hps': TWIG_HPS,
        'best_edges': best_edges,
        'best_epoch': best_epoch,
    }, save_path)
    print(f"  QA saved: {save_path} (epoch {best_epoch}, val R@10={best_val_recall:.4f})")

    del model, data
    torch.cuda.empty_cache()
    return save_path


# ===========================
# Phase 3: Evaluate both on test set
# ===========================

def evaluate(dataset, device):
    key_fields = get_key_fields(dataset)
    best_edges = TWIG_BEST_EDGES[dataset]
    twig_path = TWIG_SAVE.format(dataset=dataset)
    qa_path = QA_SAVE.format(dataset=dataset)
    test_graph_path = GRAPH_TEST.format(dataset=dataset)
    test_query_path = QUERY_TEST.format(dataset=dataset)

    for p, name in [(twig_path, "TWIG ckpt"), (qa_path, "QA ckpt"),
                     (test_graph_path, "test graph"), (test_query_path, "test queries")]:
        if not Path(p).exists():
            print(f"  Skip {dataset}: {name} not found ({p})")
            return None

    print(f"\n{'='*60}")
    print(f"Phase 3: Evaluate | {dataset} | edges: {best_edges or 'none'}")
    print(f"{'='*60}")

    # Load test graph
    data_full = torch.load(test_graph_path, map_location='cpu', weights_only=False)
    id_to_idx = rebuild_id_to_idx(data_full, key_fields)
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    embed_dim = data_full['table'].x.size(1)
    num_tables = data_full['table'].num_nodes

    # Filtered graphs
    data_twig_filtered = filter_edges(data_full, best_edges)
    data_qa_filtered = filter_edges_qa(data_full, best_edges)

    # Load TWIG model
    twig_ckpt = torch.load(twig_path, map_location=device, weights_only=False)
    data_twig_gpu = data_twig_filtered.clone().to(device)
    twig_model = DiffusionModel(
        embed_dim=embed_dim, hidden_channels=TWIG_HPS['HIDDEN_CHANNELS'],
        metadata=data_twig_gpu.metadata(), dropout=TWIG_HPS['DROPOUT'],
        sage_aggr=TWIG_HPS['SAGE_AGGR'], hetero_aggr=TWIG_HPS['HETERO_AGGR'],
    ).to(device)
    twig_model.load_state_dict(twig_ckpt['model_state_dict'], strict=False)
    twig_model.eval()

    # Load QA model
    qa_ckpt = torch.load(qa_path, map_location=device, weights_only=False)
    metadata = get_canonical_metadata()
    qa_model = QueryAwareModel(
        embed_dim=embed_dim, hidden_channels=TWIG_HPS['HIDDEN_CHANNELS'],
        metadata=metadata, dropout=TWIG_HPS['DROPOUT'],
        sage_aggr=TWIG_HPS['SAGE_AGGR'], hetero_aggr=TWIG_HPS['HETERO_AGGR'],
    ).to(device)
    qa_model.load_state_dict(qa_ckpt['model_state_dict'], strict=False)
    qa_model.eval()

    # Load queries
    texts, pos_lists = load_queries(test_query_path, key_fields, id_to_idx, num_tables)
    print(f"  Test queries: {len(texts)} (with ground truth)")

    embedder = SentenceTransformer(MODEL_NAME, device=str(device))
    query_vecs = torch.tensor(embedder.encode(texts, show_progress_bar=True), dtype=torch.float, device=device)
    query_vecs = F.normalize(query_vecs, p=2, dim=1)
    del embedder
    torch.cuda.empty_cache()

    data_qa_gpu = data_qa_filtered.clone().to(device)

    # TWIG evaluation
    with torch.no_grad():
        twig_emb = twig_model.forward(data_twig_gpu.x_dict, data_twig_gpu.edge_index_dict)
        coarse_scores = torch.matmul(query_vecs, twig_emb.T)

    twig_metrics = compute_metrics(coarse_scores, pos_lists, [1, 5, 10])
    print(f"  TWIG:  R@1={twig_metrics['R@1']:.4f}  R@5={twig_metrics['R@5']:.4f}  R@10={twig_metrics['R@10']:.4f}")

    # QA evaluation with alpha search
    coarse_k = 100
    alpha_candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    qa_rerank_data = []
    with torch.no_grad():
        for qi in tqdm(range(len(texts)), desc="  QA Rerank", leave=False):
            q_vec = query_vecs[qi:qi + 1]
            top_k_scores, top_k_idx = torch.topk(coarse_scores[qi], k=min(coarse_k, coarse_scores.size(1)))
            cand_indices = top_k_idx.cpu().tolist()

            subgraph, table_mapping = build_subgraph(data_qa_gpu, q_vec, cand_indices, device=device)
            sub_emb = qa_model.forward(subgraph.x_dict, subgraph.edge_index_dict)
            rerank_scores = torch.matmul(q_vec, sub_emb.T).squeeze(0)

            coarse_in_sub = torch.zeros(rerank_scores.size(0), device=device)
            for ri, orig_idx in enumerate(cand_indices):
                new_idx = table_mapping.get(orig_idx, -1)
                if 0 <= new_idx < coarse_in_sub.size(0):
                    coarse_in_sub[new_idx] = top_k_scores[ri]

            qa_rerank_data.append((cand_indices, table_mapping, rerank_scores.cpu(), coarse_in_sub.cpu()))

    best_alpha = 0.0
    best_r10 = 0.0
    best_qa_metrics = {}

    for alpha in alpha_candidates:
        hits = {1: 0, 5: 0, 10: 0}
        mrr_sum = 0.0
        total = len(texts)

        for qi in range(total):
            cand_indices, table_mapping, rerank_scores, coarse_in_sub = qa_rerank_data[qi]
            final = alpha * coarse_in_sub + (1 - alpha) * rerank_scores
            _, reranked = torch.topk(final, k=min(10, final.size(0)))
            new_to_old = {v: k for k, v in table_mapping.items()}
            reranked_orig = [new_to_old.get(idx.item(), -1) for idx in reranked]

            pos_set = set(pos_lists[qi])
            best_rank = float('inf')
            for rank, oidx in enumerate(reranked_orig, 1):
                if oidx in pos_set:
                    best_rank = min(best_rank, rank)
            if best_rank <= 10:
                hits[10] += 1
            if best_rank <= 5:
                hits[5] += 1
            if best_rank <= 1:
                hits[1] += 1
            if best_rank < float('inf'):
                mrr_sum += 1.0 / best_rank

        metrics = {f'R@{k}': hits[k] / total for k in [1, 5, 10]}
        metrics['MRR'] = mrr_sum / total
        r10 = metrics['R@10']
        marker = " *" if r10 > best_r10 else ""
        print(f"    alpha={alpha:.1f}: R@1={metrics['R@1']:.4f} R@5={metrics['R@5']:.4f} R@10={r10:.4f}{marker}")

        if r10 > best_r10:
            best_r10 = r10
            best_alpha = alpha
            best_qa_metrics = metrics

    print(f"\n  === {dataset} Final ===")
    print(f"  TWIG (best edges): R@1={twig_metrics['R@1']:.4f}  R@5={twig_metrics['R@5']:.4f}  R@10={twig_metrics['R@10']:.4f}")
    print(f"  QA v2 (alpha={best_alpha:.1f}): R@1={best_qa_metrics['R@1']:.4f}  R@5={best_qa_metrics['R@5']:.4f}  R@10={best_qa_metrics['R@10']:.4f}")
    delta = best_qa_metrics['R@10'] - twig_metrics['R@10']
    print(f"  Δ R@10: {'+' if delta >= 0 else ''}{delta:.4f}")

    del twig_model, qa_model, data_twig_gpu, data_qa_gpu
    torch.cuda.empty_cache()

    return {
        'dataset': dataset,
        'best_edges': best_edges,
        'best_alpha': best_alpha,
        'twig': twig_metrics,
        'qa_v2': best_qa_metrics,
        'delta_r10': delta,
    }


def compute_metrics(score_matrix, pos_lists, ks):
    total = score_matrix.size(0)
    hits = {k: 0 for k in ks}
    mrr_sum = 0.0
    for qi in range(total):
        _, topk = torch.topk(score_matrix[qi], k=min(max(ks), score_matrix.size(1)))
        topk_list = topk.cpu().tolist()
        pos_set = set(pos_lists[qi])
        best_rank = float('inf')
        for rank, idx in enumerate(topk_list, 1):
            if idx in pos_set:
                best_rank = min(best_rank, rank)
                break
        for k in ks:
            if best_rank <= k:
                hits[k] += 1
        if best_rank < float('inf'):
            mrr_sum += 1.0 / best_rank
    return {f'R@{k}': hits[k] / total for k in ks} | {'MRR': mrr_sum / total}


# ===========================
# Main
# ===========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=["feta", "ottqa", "mimo_en", "mimo_ch", "e2ewtq", "mmqa"])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--skip-train', action='store_true', help='Skip training, only evaluate')
    args = parser.parse_args()

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    all_results = {}

    for ds in args.datasets:
        if not Path(GRAPH_TRAIN.format(dataset=ds)).exists():
            print(f"Skip {ds}: no training graph")
            continue

        if not args.skip_train:
            # Phase 1: Train TWIG
            twig_path = train_twig(ds, device)
            if twig_path is None:
                continue

            # Phase 2: Train QA v2
            qa_path = train_qa(ds, twig_path, device)
            if qa_path is None:
                continue

        # Phase 3: Evaluate
        results = evaluate(ds, device)
        if results:
            all_results[ds] = results
            with open(f"{RESULTS_DIR}/{ds}.json", 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    if all_results:
        print(f"\n\n{'='*80}")
        print("Fair Comparison Summary: TWIG (best edges) vs QA v2 (same edges + query node)")
        print(f"{'='*80}")
        print(f"{'Dataset':>10} | {'Edges':>20} | {'TWIG R@10':>9} | {'QA R@10':>8} | {'Δ R@10':>7} | {'α':>3}")
        print("-" * 80)

        for ds, r in all_results.items():
            edges_str = '+'.join(TWIG_BEST_EDGES[ds]) if TWIG_BEST_EDGES[ds] else 'none'
            delta = r['delta_r10']
            tag = '✓' if delta > 0 else ' '
            print(f"{ds:>10} | {edges_str:>20} | {r['twig']['R@10']:>9.4f} | {r['qa_v2']['R@10']:>8.4f} | {'+' if delta >= 0 else ''}{delta:>6.4f}{tag} | {r['best_alpha']:.1f}")

        with open(f"{RESULTS_DIR}/summary.json", 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
