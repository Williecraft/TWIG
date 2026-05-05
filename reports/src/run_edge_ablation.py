#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Edge-Type Ablation Study with Binary Encoding (A0-A63)

Usage:
  cd ~/TabGNN/src && python run_edge_ablation.py --datasets mimo_en mimo_ch mmqa e2ewtq feta --gpu 0 --ablation $(seq 0 63)
  cd ~/TabGNN/src && python run_edge_ablation.py --datasets feta --gpu 0 --ablation $(seq 0 63)
  cd ~/TabGNN/src && python run_edge_ablation.py --datasets ottqa --gpu 1 --ablation $(seq 0 63)


Binary encoding system for edge configurations:
  Bit position: 5  4  3  2  1  0
  Edge:        tt tc tp sp cc sc
  
Edge definitions:
  tt = similar_table
  tc = has_column
  tp = comes_from
  sp = same_page
  cc = similar_content
  sc = col_same_name

Example:
  A0  (000000) = No edges
  A2  (000010) = cc only
  A6  (000110) = sp + cc
  A63 (111111) = All edges
"""

# ========= CRITICAL: Parse GPU argument BEFORE importing torch =========
# CUDA_VISIBLE_DEVICES must be set before torch is imported
import argparse
import sys
import os

# Quick parse of GPU argument only (before importing heavy libraries)
_temp_parser = argparse.ArgumentParser(add_help=False)
_temp_parser.add_argument('--gpu', type=int, default=0)
_temp_args, _ = _temp_parser.parse_known_args()

# Set GPU BEFORE importing torch
if _temp_args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_temp_args.gpu)

# Now safe to import torch and other libraries
import copy
import json
import csv
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, to_hetero
from sentence_transformers import SentenceTransformer

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

# Reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

# ========= Default Configuration =========

# Default datasets to process (can be overridden with --datasets)
DEFAULT_DATASETS = [
    "mimo_en",
    "mimo_ch",
    "mmqa",
    # "ottqa",
    # "feta",
    "e2ewtq",
]

# Default ablation configurations to run (can be overridden with --ablation)
DEFAULT_ABLATION = list(range(64))  # All 64 configurations (A0-A63)

# Edge definitions (in bit order: MSB to LSB)
ALL_EDGE_RELATIONS = [
    'similar_table',      # tt - bit 5
    'has_column',         # tc - bit 4
    'comes_from',         # tp - bit 3
    'same_page',          # sp - bit 2
    'similar_content',    # cc - bit 1
    'shared_column_name', # sc - bit 0 (table-table edge based on shared column names)
]

EDGE_ABBREV = ['tt', 'tc', 'tp', 'sp', 'cc', 'sc']

# Paths
PROJECT_DIR = "/user_data/TabGNN"
RESULTS_DIR = f"{PROJECT_DIR}/results/edge_ablation_extended"

# Training Hyperparameters
HPS = {
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
NUM_HARD_NEGATIVES = 3
REMINING_INTERVAL = 1
EARLY_STOPPING_PATIENCE = 10

# ══════════════════════════════════════════════════════════════
# Import model utilities
# ══════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
from train_model import (
    DiffusionModel, get_embedder, make_key,
    compute_scores_chunked, mine_hard_negatives_topk,
)


# ══════════════════════════════════════════════════════════════
# Binary Encoding Functions
# ══════════════════════════════════════════════════════════════

def binary_to_edges(code: int) -> list:
    """
    Convert binary code to list of edge types.
    
    Args:
        code: Integer 0-63 representing edge configuration
        
    Returns:
        List of edge type names
        
    Example:
        binary_to_edges(2) -> ['similar_content']  # 000010
        binary_to_edges(6) -> ['same_page', 'similar_content']  # 000110
    """
    edges = []
    for i, edge in enumerate(ALL_EDGE_RELATIONS):
        bit_pos = len(ALL_EDGE_RELATIONS) - 1 - i
        if code & (1 << bit_pos):
            edges.append(edge)
    return edges


def get_config_label(code: int) -> str:
    """
    Generate human-readable label for configuration.
    
    Args:
        code: Configuration code 0-63
        
    Returns:
        Label string like "tt + tc + sp"
    """
    if code == 0:
        return "No edges"
    
    abbrevs = []
    for i in range(6):
        bit_pos = 5 - i
        if code & (1 << bit_pos):
            abbrevs.append(EDGE_ABBREV[i])
    
    return " + ".join(abbrevs)


def get_binary_repr(code: int) -> str:
    """Get 6-bit binary representation string."""
    return format(code, '06b')


# ══════════════════════════════════════════════════════════════
# Graph manipulation helpers
# ══════════════════════════════════════════════════════════════

def filter_edges(data, keep_relations: list):
    """
    Return a copy of `data` where only edge types whose relation name
    is in `keep_relations` are retained. Node features are untouched.
    
    反向邊處理：當正向邊被過濾掉時，對應的反向邊也會被過濾掉。
    例如：如果 'has_column' 不在 keep_relations 中，那麼 'rev_has_column' 也不會被保留。
    """
    from torch_geometric.data import HeteroData

    filtered = HeteroData()

    # Copy node features
    for node_type in data.node_types:
        for attr_name in data[node_type].keys():
            filtered[node_type][attr_name] = data[node_type][attr_name]

    # 定義正向邊與反向邊的映射關係
    forward_to_reverse = {
        'has_column': 'rev_has_column',
        'comes_from': 'rev_comes_from',
        'similar_table': 'similar_table',  # 對稱邊，反向邊是自己
        'same_page': 'same_page',          # 對稱邊，反向邊是自己
        'similar_content': 'similar_content',  # 對稱邊，反向邊是自己
        'shared_column_name': 'shared_column_name',  # 對稱邊，反向邊是自己
    }
    
    reverse_to_forward = {
        'rev_has_column': 'has_column',
        'rev_comes_from': 'comes_from',
    }

    # 構建實際應該保留的邊集合（包括對應的反向邊）
    edges_to_keep = set()
    for relation in keep_relations:
        edges_to_keep.add(relation)
        # 如果是正向邊，也保留對應的反向邊
        if relation in forward_to_reverse:
            edges_to_keep.add(forward_to_reverse[relation])

    # Copy only the selected edge types
    for edge_type, edge_index in data.edge_index_dict.items():
        src_type, relation, dst_type = edge_type
        
        # 檢查這條邊是否應該保留
        should_keep = False
        
        if relation in edges_to_keep:
            should_keep = True
        elif relation in reverse_to_forward:
            # 這是反向邊，檢查對應的正向邊是否在 keep_relations 中
            forward_relation = reverse_to_forward[relation]
            if forward_relation in keep_relations:
                should_keep = True
        
        if should_keep:
            filtered[src_type, relation, dst_type].edge_index = edge_index

    # Ensure every node type appears as a destination
    # Check if edge_index_dict exists and is not empty first
    if len(filtered.edge_types) > 0:
        dest_types = set()
        for (_, _, dst_type) in filtered.edge_types:
            dest_types.add(dst_type)

        for node_type in filtered.node_types:
            if node_type not in dest_types:
                n = filtered[node_type].x.size(0)
                self_loop = torch.tensor([[0], [0]], dtype=torch.long)
                filtered[node_type, f'_self_loop_{node_type}', node_type].edge_index = self_loop
    else:
        # No edges retained - add self-loops for all node types
        for node_type in filtered.node_types:
            self_loop = torch.tensor([[0], [0]], dtype=torch.long)
            filtered[node_type, f'_self_loop_{node_type}', node_type].edge_index = self_loop

    # Copy metadata_maps if present
    if hasattr(data, 'metadata_maps'):
        filtered.metadata_maps = data.metadata_maps

    return filtered


# ══════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════

def load_queries(query_path, id_to_idx, key_fields):
    """Load queries and return (texts, pos_indices_lists)."""
    texts = []
    pos_lists = []
    with open(query_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            gt_list = obj.get('ground_truth_list', []) or []
            pos_indices = []
            for gt in gt_list:
                if all(gt.get(field) is not None for field in key_fields):
                    key = make_key(gt, key_fields)
                    idx = id_to_idx.get(key, -1)
                    if idx != -1:
                        pos_indices.append(idx)
            if not pos_indices:
                continue
            questions = obj.get('questions', [])
            if not questions and 'question' in obj:
                questions = [obj['question']]
            for q in questions:
                if q and q.strip():
                    texts.append(q.strip())
                    pos_lists.append(pos_indices)
    return texts, pos_lists


def build_id_to_idx(data, key_fields):
    """Rebuild id_to_idx from graph metadata."""
    if hasattr(data, 'metadata_maps') and 'table_meta' in data.metadata_maps:
        table_meta = data.metadata_maps['table_meta']
        mapping = {}
        for idx, meta in enumerate(table_meta):
            key = make_key(meta, key_fields)
            if key not in mapping:
                mapping[key] = idx
        return mapping
    return data.metadata_maps.get('table_id_to_idx', {})


def build_graph_hard_negatives(data, num_hard_negatives, pos_indices_lists):
    """Build graph-structure-based hard negatives from column similarity edges."""
    has_col_key = ('table', 'has_column', 'column')
    sim_col_key = ('column', 'similar_content', 'column')
    name_col_key = ('column', 'col_same_name', 'column')

    if has_col_key not in data.edge_index_dict:
        return [[-1] * num_hard_negatives for _ in pos_indices_lists]

    # Use both column similarity edges if available
    col_edges = []
    if sim_col_key in data.edge_index_dict:
        col_edges.append(data[sim_col_key].edge_index.cpu())
    if name_col_key in data.edge_index_dict:
        col_edges.append(data[name_col_key].edge_index.cpu())
    
    if not col_edges:
        return [[-1] * num_hard_negatives for _ in pos_indices_lists]

    t2c = data[has_col_key].edge_index.cpu()
    c2t_map = {}
    for i in range(t2c.size(1)):
        c2t_map[t2c[1, i].item()] = t2c[0, i].item()

    table_neighbors = {}
    for c2c in col_edges:
        for i in range(c2c.size(1)):
            c_src, c_dst = c2c[0, i].item(), c2c[1, i].item()
            if c_src in c2t_map and c_dst in c2t_map:
                t_src, t_dst = c2t_map[c_src], c2t_map[c_dst]
                if t_src != t_dst:
                    table_neighbors.setdefault(t_src, set()).add(t_dst)

    hard_neg_lists = []
    for pos_list in pos_indices_lists:
        all_neighbors = set()
        for pos_idx in pos_list:
            all_neighbors.update(table_neighbors.get(pos_idx, set()))
        all_neighbors -= set(pos_list)
        neighbors = list(all_neighbors)
        if len(neighbors) >= num_hard_negatives:
            hard_negs = random.sample(neighbors, num_hard_negatives)
        else:
            hard_negs = neighbors + [-1] * (num_hard_negatives - len(neighbors))
        hard_neg_lists.append(hard_negs)
    return hard_neg_lists


# ══════════════════════════════════════════════════════════════
# Training & Evaluation
# ══════════════════════════════════════════════════════════════

def setup_model(embed_dim, metadata, hps, device):
    """Initialize model, optimizer, scheduler, and scaler."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
    import torch.optim as optim

    model = DiffusionModel(
        embed_dim=embed_dim,
        hidden_channels=hps['HIDDEN_CHANNELS'],
        metadata=metadata,
        dropout=hps['DROPOUT'],
        sage_aggr=hps.get('SAGE_AGGR', 'mean'),
        hetero_aggr=hps.get('HETERO_AGGR', 'sum'),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=hps['LEARNING_RATE'],
                            weight_decay=hps['WEIGHT_DECAY'])
    warmup_scheduler = LambdaLR(
        optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / float(hps['WARMUP_EPOCHS'])))
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=hps['NUM_EPOCHS'] - hps['WARMUP_EPOCHS'])
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[hps['WARMUP_EPOCHS']])
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    return model, optimizer, scheduler, scaler


def train_one_epoch(model, data, optimizer, scaler, scheduler,
                    query_vectors, pos_indices_lists, hard_neg_indices,
                    hps, epoch, device):
    """One epoch of training."""
    model.train()
    total_loss = 0.0
    indices = list(range(len(query_vectors)))
    random.shuffle(indices)

    progress = epoch / float(hps['NUM_EPOCHS'])
    curr_temp = hps['TEMP_END'] if epoch > hps['NUM_EPOCHS'] * 0.7 else hps['TEMP_START']
    curr_smooth = hps['SMOOTH_START'] + (hps['SMOOTH_END'] - hps['SMOOTH_START']) * progress

    batch_count = 0
    for start in range(0, len(indices), hps['BATCH_SIZE']):
        end = min(start + hps['BATCH_SIZE'], len(indices))
        batch_idx = indices[start:end]
        q_batch = query_vectors[batch_idx]
        labels = torch.tensor(
            [pos_indices_lists[i][0] for i in batch_idx],
            dtype=torch.long, device=device)

        optimizer.zero_grad()

        with autocast(enabled=(device.type == 'cuda')):
            table_emb = model.forward(data.x_dict, data.edge_index_dict)
            q_batch_norm = F.normalize(q_batch, p=2, dim=1)
            logits = compute_scores_chunked(q_batch_norm, table_emb,
                                            hps['CHUNK_SIZE']) / curr_temp
            loss_in_batch = F.cross_entropy(logits, labels,
                                            label_smoothing=curr_smooth)

            # Hard Negative Margin Loss
            loss_hard = 0.0
            batch_hard_negs = [hard_neg_indices[i] for i in batch_idx]
            hard_negs_tensor = torch.tensor(batch_hard_negs, device=device,
                                            dtype=torch.long)
            mask = (hard_negs_tensor != -1)
            if mask.any():
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
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=hps['CLIP_GRAD_NORM'])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        batch_count += 1

    scheduler.step()
    return total_loss / max(1, batch_count)


def _full_recall_at_k(retrieved_indices, relevant_set, k):
    """Recall@k = proportion of relevant items found in top-k."""
    if not relevant_set:
        return 0.0
    retrieved_set = set(retrieved_indices[:k])
    return len(retrieved_set.intersection(relevant_set)) / len(relevant_set)


def evaluate_recall(model, data, query_vectors, pos_indices_lists, hps, device, dataset=''):
    """
    Evaluate Full Recall@k with dataset-specific metrics.
    MMQA uses R@2, R@5, R@10 while others use R@1, R@5, R@10.
    """
    # Dataset-specific k value for first metric
    k1 = 2 if dataset == 'mmqa' else 1
    metric_name_k1 = f'recall@{k1}'
    model.eval()
    with torch.no_grad():
        table_emb = model.forward(data.x_dict, data.edge_index_dict)
        q_norm = F.normalize(query_vectors, p=2, dim=1)
        score_mat = compute_scores_chunked(q_norm, table_emb, hps['CHUNK_SIZE'])

        score_mat_cpu = score_mat.cpu()
        num_queries = score_mat_cpu.size(0)
        top_k = min(10, score_mat_cpu.size(1))

        recall_k1_sum = recall5_sum = recall10_sum = 0.0
        eval_count = 0

        for q_i in range(num_queries):
            pos_set = set(pos_indices_lists[q_i])
            if not pos_set:
                continue
            eval_count += 1

            _, top_indices = torch.topk(score_mat_cpu[q_i], k=top_k)
            retrieved = top_indices.tolist()

            recall_k1_sum += _full_recall_at_k(retrieved, pos_set, k1)
            recall5_sum += _full_recall_at_k(retrieved, pos_set, 5)
            recall10_sum += _full_recall_at_k(retrieved, pos_set, 10)

    if eval_count == 0:
        return {metric_name_k1: 0.0, 'recall@5': 0.0, 'recall@10': 0.0}

    return {
        metric_name_k1: recall_k1_sum / eval_count,
        'recall@5': recall5_sum / eval_count,
        'recall@10': recall10_sum / eval_count,
    }


# ══════════════════════════════════════════════════════════════
# Incremental saving utilities
# ══════════════════════════════════════════════════════════════

def save_result_incremental(result, dataset, results_dir):
    """Save single result incrementally by merging with existing results."""
    results_csv = f"{results_dir}/results.csv"
    results_json = f"{results_dir}/results.json"
    
    # Load existing results
    existing_results = []
    if os.path.exists(results_json):
        try:
            with open(results_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing_results = data.get('results', [])
        except Exception as e:
            print(f"    Warning: Could not load existing results: {e}")
    
    # Merge: update if exists, insert if new
    results_dict = {r['code']: r for r in existing_results}
    results_dict[result['code']] = result
    final_results = sorted(results_dict.values(), key=lambda x: x['code'])
    
    # Save CSV - include both recall@1 and recall@2
    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['config', 'code', 'binary', 'label', 'num_edges', 'kept_edges',
                      'recall@1', 'recall@2', 'recall@5', 'recall@10',
                      'best_epoch', 'elapsed_min']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(final_results)
    
    # Save JSON
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Edge-Type Ablation Study (Binary Encoding)',
            'dataset': dataset,
            'edge_types': ALL_EDGE_RELATIONS,
            'edge_abbreviations': EDGE_ABBREV,
            'edge_order': 'tt tc tp sp cc sc (bit 5 to 0)',
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': HPS,
            'results': final_results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"    💾 Saved {result['config']} to results files")


def load_completed_configs_for_dataset(dataset, results_dir):
    """Load completed configs for a specific dataset."""
    tracking_file = f"{results_dir}/{dataset}/completed_configs.json"
    if not os.path.exists(tracking_file):
        return set()
    try:
        with open(tracking_file, 'r', encoding='utf-8') as f:
            configs = json.load(f)
            return set(configs)
    except Exception:
        return set()


def update_completed_configs_for_dataset(dataset, config_code, results_dir):
    """Update tracking file with newly completed config for a specific dataset."""
    tracking_file = f"{results_dir}/{dataset}/completed_configs.json"
    completed = load_completed_configs_for_dataset(dataset, results_dir)
    completed.add(config_code)
    serializable = sorted(list(completed))
    with open(tracking_file, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════
# Main ablation runner
# ══════════════════════════════════════════════════════════════

def run_single_ablation(config_code, dataset, device):
    """Run one ablation experiment end-to-end."""
    
    # Get configuration
    edges = binary_to_edges(config_code)
    config_name = f"A{config_code}"
    config_label = get_config_label(config_code)
    binary_repr = get_binary_repr(config_code)
    
    # Get key fields for dataset
    if dataset in ["ottqa", "feta", "e2ewtq"]:
        key_fields = ("sheet_name", "file_name")
    elif dataset in ["mimo_en", "mimo_ch", "mmqa"]:
        key_fields = ("id",)
    else:
        key_fields = ("id",)
    
    # Paths
    train_graph = f"{PROJECT_DIR}/data/processed/train/{dataset}/graph.pt"
    dev_graph = f"{PROJECT_DIR}/data/processed/dev/{dataset}/graph.pt"
    test_graph = f"{PROJECT_DIR}/data/processed/test/{dataset}/graph.pt"
    train_query = f"{PROJECT_DIR}/data/table/train/{dataset}/query.jsonl"
    dev_query = f"{PROJECT_DIR}/data/table/dev/{dataset}/query.jsonl"
    test_query = f"{PROJECT_DIR}/data/table/test/{dataset}/query.jsonl"
    
    print(f"\n{'='*70}")
    print(f"  [{config_name}] {config_label}")
    print(f"  Binary: {binary_repr} | Edges: {len(edges)}/{len(ALL_EDGE_RELATIONS)}")
    print(f"  Dataset: {dataset}")
    print(f"{'='*70}")

    t_start = time.time()

    # ── 1. Load and filter train graph ──
    print("  Loading train graph...")
    train_data_full = torch.load(train_graph, map_location='cpu', weights_only=False)
    
    # Check if all required edges exist in the graph
    available_edges = set()
    for src, rel, dst in train_data_full.edge_types:
        available_edges.add(rel)
    
    missing_edges = []
    for edge in edges:
        if edge not in available_edges:
            missing_edges.append(edge)
    
    if missing_edges:
        print(f"  ⚠️  Skipping: Required edges not in graph: {missing_edges}")
        return {
            'config': config_name,
            'code': config_code,
            'binary': binary_repr,
            'label': config_label,
            'num_edges': len(edges),
            'kept_edges': ', '.join(edges),
            'recall@1': 0.0,
            'recall@5': 0.0,
            'recall@10': 0.0,
            'best_epoch': 0,
            'elapsed_min': 0.0,
            'skipped': True,
            'skip_reason': f"Missing edges: {', '.join(missing_edges)}"
        }
    
    train_data = filter_edges(train_data_full, edges)
    del train_data_full

    train_id_to_idx = build_id_to_idx(train_data, key_fields)
    embed_dim = train_data['table'].x.size(1)

    # ── 2. Load training queries ──
    train_texts, train_pos_lists = load_queries(train_query, train_id_to_idx, key_fields)
    print(f"  Train queries: {len(train_texts)}")

    hard_neg_lists = build_graph_hard_negatives(
        train_data, NUM_HARD_NEGATIVES, train_pos_lists)

    # ── 3. Embed queries ──
    embedder = get_embedder(device=str(device))
    print("  Embedding train queries...")
    train_q_vecs = torch.tensor(
        embedder.encode(train_texts, show_progress_bar=True),
        dtype=torch.float, device=device)

    # ── 4. Load and filter dev graph ──
    print("  Loading dev graph...")
    dev_data_full = torch.load(dev_graph, map_location='cpu', weights_only=False)
    dev_data = filter_edges(dev_data_full, edges)
    del dev_data_full

    dev_id_to_idx = build_id_to_idx(dev_data, key_fields)
    dev_texts, dev_pos_lists = load_queries(dev_query, dev_id_to_idx, key_fields)
    print(f"  Dev queries: {len(dev_texts)}")

    dev_q_vecs = None
    if dev_texts:
        dev_q_vecs = torch.tensor(
            embedder.encode(dev_texts, show_progress_bar=False),
            dtype=torch.float, device=device)

    del embedder
    torch.cuda.empty_cache()

    # ── 5. Train ──
    train_data_gpu = train_data.clone().to(device)
    model, optimizer, scheduler, scaler = setup_model(
        embed_dim, train_data_gpu.metadata(), HPS, device)

    best_val_r10 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, HPS['NUM_EPOCHS'] + 1):
        if epoch > 1 and epoch % REMINING_INTERVAL == 0:
            hard_neg_lists = mine_hard_negatives_topk(
                model, train_data_gpu, train_q_vecs, train_pos_lists,
                num_hard_negatives=NUM_HARD_NEGATIVES, device=str(device))

        avg_loss = train_one_epoch(
            model, train_data_gpu, optimizer, scaler, scheduler,
            train_q_vecs, train_pos_lists, hard_neg_lists,
            HPS, epoch, device)

        # Validation
        if dev_q_vecs is not None:
            dev_data_gpu = dev_data.clone().to(device)
            val_metrics = evaluate_recall(
                model, dev_data_gpu, dev_q_vecs, dev_pos_lists, HPS, device, dataset)
            val_r10 = val_metrics['recall@10']
            del dev_data_gpu
            torch.cuda.empty_cache()

            if val_r10 > best_val_r10:
                best_val_r10 = val_r10
                best_epoch = epoch
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"    Epoch {epoch:2d} | Loss {avg_loss:.4f} | "
                      f"Val R@10 {val_r10:.4f} | "
                      f"Best {best_val_r10:.4f} (ep {best_epoch}) | "
                      f"Pat {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"    Early stopping at epoch {epoch}.")
                break
        else:
            if epoch % 5 == 0 or epoch == 1:
                print(f"    Epoch {epoch:2d} | Loss {avg_loss:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Loaded best model from epoch {best_epoch}")

    # ── 6. Evaluate on test set ──
    print("  Loading test graph...")
    test_data_full = torch.load(test_graph, map_location='cpu', weights_only=False)
    test_data = filter_edges(test_data_full, edges)
    del test_data_full

    test_id_to_idx = build_id_to_idx(test_data, key_fields)
    test_data_gpu = test_data.clone().to(device)

    test_model = DiffusionModel(
        embed_dim=embed_dim,
        hidden_channels=HPS['HIDDEN_CHANNELS'],
        metadata=test_data_gpu.metadata(),
        dropout=HPS['DROPOUT'],
        sage_aggr=HPS.get('SAGE_AGGR', 'mean'),
        hetero_aggr=HPS.get('HETERO_AGGR', 'sum'),
    ).to(device)
    test_model.load_state_dict(model.state_dict(), strict=False)
    test_model.eval()

    test_texts, test_pos_lists = load_queries(test_query, test_id_to_idx, key_fields)
    print(f"  Test queries: {len(test_texts)}")

    test_embedder = get_embedder(device=str(device))
    test_q_vecs = torch.tensor(
        test_embedder.encode(test_texts, show_progress_bar=False),
        dtype=torch.float, device=device)
    del test_embedder
    torch.cuda.empty_cache()

    test_metrics = evaluate_recall(
        test_model, test_data_gpu, test_q_vecs, test_pos_lists, HPS, device, dataset)

    elapsed = time.time() - t_start

    # ── Clean up ──
    del model, test_model, train_data_gpu, test_data_gpu
    del train_q_vecs, dev_q_vecs, test_q_vecs
    torch.cuda.empty_cache()

    # Get metric names (mmqa uses R@2, others use R@1)
    k1 = 2 if dataset == 'mmqa' else 1
    metric_k1_key = f'recall@{k1}'
    
    # Store both R@1 and R@2 to support mixed results
    result = {
        'config': config_name,
        'code': config_code,
        'binary': binary_repr,
        'label': config_label,
        'num_edges': len(edges),
        'kept_edges': ', '.join(edges) if edges else 'none',
        'recall@1': test_metrics.get('recall@1', 0.0),
        'recall@2': test_metrics.get('recall@2', 0.0),
        'recall@5': test_metrics['recall@5'],
        'recall@10': test_metrics['recall@10'],
        'best_epoch': best_epoch,
        'elapsed_min': round(elapsed / 60, 1),
    }
    return result


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Edge-Type Ablation Study with Binary Encoding (A0-A63)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run with default datasets and ablation configs
  python run_edge_ablation.py
  
  # Run specific datasets only
  python run_edge_ablation.py --datasets feta ottqa
  
  # Run specific ablation configurations
  python run_edge_ablation.py --ablation 0 4 8 16 32 63
  
  # Combine both
  python run_edge_ablation.py --datasets feta --ablation 0 63
        '''
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help=f'Datasets to process. Default: {" ".join(DEFAULT_DATASETS)}'
    )
    
    parser.add_argument(
        '--ablation',
        nargs='+',
        type=int,
        default=None,
        help=f'Ablation configuration codes (0-63). Default: {DEFAULT_ABLATION}'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID to use (0, 1, 2, etc.). If not specified, uses CUDA_VISIBLE_DEVICES or GPU 0'
    )
    
    args = parser.parse_args()
    
    # Use command line arguments if provided, otherwise use defaults
    DATASETS = args.datasets if args.datasets is not None else DEFAULT_DATASETS
    ABLATION = sorted(set(args.ablation)) if args.ablation is not None else DEFAULT_ABLATION
    
    # Display GPU info (already set at import time)
    print(f"Using GPU: {args.gpu} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    
    print(f"╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  Edge-Type Ablation Study (Binary Encoding A0-A63)               ║")
    print(f"║  Edge order: {' '.join(EDGE_ABBREV):<49s} ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Datasets: {DATASETS}")
    print(f"Ablation configs: {ABLATION}")
    print(f"Total experiments: {len(DATASETS) * len(ABLATION)}\n")
    
    # Process each dataset
    for dataset in DATASETS:
        print(f"\n{'='*70}")
        print(f"  PROCESSING DATASET: {dataset.upper()}")
        print(f"{'='*70}")
        
        results_dir = f"{RESULTS_DIR}/{dataset}"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # Get completed configs for this dataset from tracking file
        existing_codes = load_completed_configs_for_dataset(dataset, RESULTS_DIR)
        print(f"Already completed configs from tracking file: {sorted(existing_codes)}")

        # Filter out already run configs
        configs_to_run = [c for c in ABLATION if c not in existing_codes]
        
        if len(configs_to_run) < len(ABLATION):
            skipped = set(ABLATION) - set(configs_to_run)
            print(f"Skipping already computed configs: {sorted(skipped)}")

        print(f"Configs to run: {configs_to_run}\n")

        # Run each config and save incrementally
        for config_code in configs_to_run:
            try:
                result = run_single_ablation(config_code, dataset, device)

                # Check if config was skipped due to missing edges
                if result.get('skipped', False):
                    print(f"    ⚠️  {result.get('skip_reason', 'Unknown reason')}\\n")
                    continue  # Don't save or track skipped configs

                # Print immediate result
                k1 = 2 if dataset == 'mmqa' else 1
                metric_k1_key = f'recall@{k1}'
                
                print(f"\\n  ┌─ Result for {result['config']} ({result['binary']})")
                print(f"  │  Label: {result['label']}")
                print(f"  │  R@{k1}  = {result.get(metric_k1_key, 0.0):.4f}")
                print(f"  │  R@5  = {result['recall@5']:.4f}")
                print(f"  │  R@10 = {result['recall@10']:.4f}")
                print(f"  │  Best Epoch = {result['best_epoch']}")
                print(f"  └─ Time = {result['elapsed_min']} min")
                
                # ✅ INCREMENTAL SAVE: Save result immediately after completion
                save_result_incremental(result, dataset, results_dir)
                
                # ✅ UPDATE TRACKING: Mark this config as completed
                update_completed_configs_for_dataset(dataset, config_code, RESULTS_DIR)
                print(f"    ✓ Marked {result['config']} as completed in tracking file")

            except Exception as e:
                print(f"\n  !! ERROR in A{config_code}: {e}")
                import traceback
                traceback.print_exc()
        
        # ── Final Summary for this dataset ──
        results_json = f"{results_dir}/results.json"
        if os.path.exists(results_json):
            with open(results_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                final_results = data.get('results', [])
            
            if final_results:
                # Determine metric label
                k1 = 2 if dataset == 'mmqa' else 1
                metric_label = f'R@{k1}'
                metric_key = f'recall@{k1}'
                
                print(f"\n{'='*90}")
                print(f"  ABLATION RESULTS SUMMARY — {dataset.upper()}")
                print(f"{'='*90}")
                print(f"  {'Config':<7} {'Binary':<8} {'Label':<25} {metric_label:>6} {'R@5':>6} {'R@10':>6}")
                print(f"  {'─'*7} {'─'*8} {'─'*25} {'─'*6} {'─'*6} {'─'*6}")
                for r in final_results:
                    print(f"  {r['config']:<7} {r['binary']:<8} {r['label']:<25} "
                          f"{r.get(metric_key, 0.0):.4f} {r['recall@5']:.4f} {r['recall@10']:.4f}")
                print(f"{'='*90}")

    print(f"\n{'='*70}")
    print(f"  ALL DATASETS COMPLETED")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()