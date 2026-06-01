#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Query-Aware v2 Edge-Type Ablation Study (A0-A63)

For each of the 64 edge configurations, this script:
  1. Trains a base TWIG model (using that edge config)
  2. Fine-tunes a Query-Aware v2 model on top (from TWIG checkpoint)
  3. Evaluates on the test set with QA reranking
  4. Saves results incrementally

Usage:
  cd ~/TabGNN/src/query_aware
  python run_qa_edge_ablation.py --datasets feta --gpu 0 --ablation $(seq 0 63)
  python run_qa_edge_ablation.py --datasets feta ottqa --gpu 0
"""

# ========= CRITICAL: Parse GPU argument BEFORE importing torch =========
import argparse
import sys
import os

_temp_parser = argparse.ArgumentParser(add_help=False)
_temp_parser.add_argument('--gpu', type=int, default=0)
_temp_args, _ = _temp_parser.parse_known_args()
if _temp_args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_temp_args.gpu)

import copy
import json
import csv
import math
import random
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
import torch.optim as optim
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphSAGE, to_hetero, GraphNorm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_NAME = 'BAAI/bge-m3'

DEFAULT_DATASETS = ["feta", "ottqa", "mimo_en", "mimo_ch", "e2ewtq", "mmqa"]
DEFAULT_ABLATION = list(range(64))

# Edge definitions (bit order: MSB to LSB)
ALL_EDGE_RELATIONS = [
    'similar_table',      # tt - bit 5
    'has_column',         # tc - bit 4
    'comes_from',         # tp - bit 3
    'same_page',          # sp - bit 2
    'similar_content',    # cc - bit 1
    'shared_column_name', # sc - bit 0
]
EDGE_ABBREV = ['tt', 'tc', 'tp', 'sp', 'cc', 'sc']

RESULTS_DIR = str(PROJECT_DIR / "results" / "qa_edge_ablation")

# --- Phase 1: Base TWIG training hyperparameters ---
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
TWIG_NUM_HARD_NEGATIVES = 3
TWIG_REMINING_INTERVAL = 1
TWIG_EARLY_STOPPING_PATIENCE = 10

# --- Phase 2: Query-Aware v2 fine-tuning hyperparameters ---
QA_QUERY_EDGE_MODE = "E4"
QA_SUBGRAPH_K = 30
QA_COARSE_K_EVAL = 50
QA_NUM_EPOCHS = 15
QA_WARMUP_EPOCHS = 2
QA_BATCH_SIZE = 64
QA_NUM_HARD_NEGATIVES = 3
QA_EARLY_STOPPING_PATIENCE = 5
QA_BASE_LR = 1e-4
QA_QUERY_LR = 5e-4
QA_WEIGHT_DECAY = 0.03
QA_CLIP_GRAD_NORM = 0.60
QA_TEMP = 0.04
QA_LABEL_SMOOTH = 0.08

# --- Phase 3: Evaluation hyperparameters ---
EVAL_COARSE_K = 100
EVAL_TOP_K = 10
EVAL_ALPHA = 0.0  # 不使用插值，直接用 QA rerank 分數


# ══════════════════════════════════════════════════════════════
# Binary Encoding Functions (same as run_edge_ablation.py)
# ══════════════════════════════════════════════════════════════

def binary_to_edges(code: int) -> list:
    edges = []
    for i, edge in enumerate(ALL_EDGE_RELATIONS):
        bit_pos = len(ALL_EDGE_RELATIONS) - 1 - i
        if code & (1 << bit_pos):
            edges.append(edge)
    return edges


def get_config_label(code: int) -> str:
    if code == 0:
        return "No edges"
    abbrevs = []
    for i in range(6):
        bit_pos = 5 - i
        if code & (1 << bit_pos):
            abbrevs.append(EDGE_ABBREV[i])
    return " + ".join(abbrevs)


def get_binary_repr(code: int) -> str:
    return format(code, '06b')


# ══════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════

def get_key_fields(dataset_name: str) -> tuple:
    if dataset_name in ["ottqa", "feta", "e2ewtq"]:
        return ("sheet_name", "file_name")
    elif dataset_name in ["mimo_en", "mimo_ch", "mmqa"]:
        return ("id",)
    return ("id",)


def make_key(item: dict, key_fields: tuple) -> str:
    return "|".join(str(item.get(f, "")) for f in key_fields)


def build_id_to_idx(data, key_fields):
    if hasattr(data, 'metadata_maps') and 'table_meta' in data.metadata_maps:
        table_meta = data.metadata_maps['table_meta']
        mapping = {}
        for idx, meta in enumerate(table_meta):
            key = make_key(meta, key_fields)
            if key not in mapping:
                mapping[key] = idx
        return mapping
    return data.metadata_maps.get('table_id_to_idx', {})


def filter_edges(data, keep_relations: list):
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

    if len(filtered.edge_types) > 0:
        dest_types = {dst for _, _, dst in filtered.edge_types}
        for node_type in filtered.node_types:
            if node_type not in dest_types:
                self_loop = torch.tensor([[0], [0]], dtype=torch.long)
                filtered[node_type, f'_self_loop_{node_type}', node_type].edge_index = self_loop
    else:
        for node_type in filtered.node_types:
            self_loop = torch.tensor([[0], [0]], dtype=torch.long)
            filtered[node_type, f'_self_loop_{node_type}', node_type].edge_index = self_loop

    if hasattr(data, 'metadata_maps'):
        filtered.metadata_maps = data.metadata_maps

    return filtered


def load_queries(query_path, id_to_idx, key_fields):
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


def compute_scores_chunked(q_vecs, table_emb, chunk_size=1024):
    scores = []
    for start in range(0, q_vecs.size(0), chunk_size):
        end = min(start + chunk_size, q_vecs.size(0))
        scores.append(torch.matmul(q_vecs[start:end], table_emb.T))
    return torch.cat(scores, dim=0)


# ══════════════════════════════════════════════════════════════
# Phase 1: Base TWIG Model (from run_edge_ablation.py)
# ══════════════════════════════════════════════════════════════

class DiffusionModel(nn.Module):
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
        x_dict = self.hetero_sage(x_dict, edge_index_dict)
        x = x_dict['table']
        x = self.norm(x)
        x = self.proj_head(x)
        return F.normalize(x, p=2, dim=1)


def mine_hard_negatives_topk(model, data, query_vecs, pos_lists,
                              num_hard_negatives=3, device='cuda'):
    model.eval()
    with torch.no_grad():
        table_emb = model.forward(data.x_dict, data.edge_index_dict)
        q_norm = F.normalize(query_vecs, p=2, dim=1)
        num_tables = table_emb.size(0)
        all_negs = []
        for start in range(0, len(query_vecs), 1024):
            end = min(start + 1024, len(query_vecs))
            sim = torch.matmul(q_norm[start:end], table_emb.T)
            for i in range(end - start):
                for pos_idx in pos_lists[start + i]:
                    if 0 <= pos_idx < num_tables:
                        sim[i, pos_idx] = -float('inf')
            _, topk = torch.topk(sim, k=min(num_hard_negatives, num_tables - 1), dim=1)
            all_negs.extend(topk.tolist())
    model.train()
    return all_negs


def _full_recall_at_k(retrieved_indices, relevant_set, k):
    if not relevant_set:
        return 0.0
    retrieved_set = set(retrieved_indices[:k])
    return len(retrieved_set.intersection(relevant_set)) / len(relevant_set)


def train_twig(edges, dataset, key_fields, device, embedder):
    """Train a base TWIG model with the given edge configuration. Returns model state dict."""
    hps = TWIG_HPS

    train_graph_path = str(PROJECT_DIR / f"data/processed/train/{dataset}/graph.pt")
    dev_graph_path = str(PROJECT_DIR / f"data/processed/dev/{dataset}/graph.pt")
    train_query_path = str(PROJECT_DIR / f"data/table/train/{dataset}/query.jsonl")
    dev_query_path = str(PROJECT_DIR / f"data/table/dev/{dataset}/query.jsonl")

    # Load and filter train graph
    train_data_full = torch.load(train_graph_path, map_location='cpu', weights_only=False)
    available_edges = {rel for _, rel, _ in train_data_full.edge_types}
    missing = [e for e in edges if e not in available_edges]
    if missing:
        print(f"    [TWIG] Skipping: missing edges {missing}")
        return None

    train_data = filter_edges(train_data_full, edges)
    del train_data_full
    train_id_to_idx = build_id_to_idx(train_data, key_fields)
    embed_dim = train_data['table'].x.size(1)

    train_texts, train_pos_lists = load_queries(train_query_path, train_id_to_idx, key_fields)
    if not train_texts:
        print(f"    [TWIG] No valid training queries")
        return None

    print(f"    [TWIG] Embedding {len(train_texts)} train queries...")
    train_q_vecs = torch.tensor(
        embedder.encode(train_texts, show_progress_bar=False),
        dtype=torch.float, device=device)

    # Dev
    dev_data_full = torch.load(dev_graph_path, map_location='cpu', weights_only=False)
    dev_data = filter_edges(dev_data_full, edges)
    del dev_data_full
    dev_id_to_idx = build_id_to_idx(dev_data, key_fields)
    dev_texts, dev_pos_lists = load_queries(dev_query_path, dev_id_to_idx, key_fields)
    dev_q_vecs = None
    if dev_texts:
        dev_q_vecs = torch.tensor(
            embedder.encode(dev_texts, show_progress_bar=False),
            dtype=torch.float, device=device)

    # Setup model
    train_data_gpu = train_data.clone().to(device)
    model = DiffusionModel(
        embed_dim=embed_dim,
        hidden_channels=hps['HIDDEN_CHANNELS'],
        metadata=train_data_gpu.metadata(),
        dropout=hps['DROPOUT'],
        sage_aggr=hps['SAGE_AGGR'],
        hetero_aggr=hps['HETERO_AGGR'],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=hps['LEARNING_RATE'],
                            weight_decay=hps['WEIGHT_DECAY'])
    warmup_sched = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / float(hps['WARMUP_EPOCHS'])))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=hps['NUM_EPOCHS'] - hps['WARMUP_EPOCHS'])
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[hps['WARMUP_EPOCHS']])
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Initial hard negatives from graph structure
    hard_neg_lists = [[-1] * TWIG_NUM_HARD_NEGATIVES for _ in train_pos_lists]

    best_val_r10 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, hps['NUM_EPOCHS'] + 1):
        if epoch > 1 and epoch % TWIG_REMINING_INTERVAL == 0:
            hard_neg_lists = mine_hard_negatives_topk(
                model, train_data_gpu, train_q_vecs, train_pos_lists,
                num_hard_negatives=TWIG_NUM_HARD_NEGATIVES, device=str(device))

        # Train one epoch
        model.train()
        total_loss = 0.0
        indices = list(range(len(train_q_vecs)))
        random.shuffle(indices)
        progress = epoch / float(hps['NUM_EPOCHS'])
        curr_temp = hps['TEMP_END'] if epoch > hps['NUM_EPOCHS'] * 0.7 else hps['TEMP_START']
        curr_smooth = hps['SMOOTH_START'] + (hps['SMOOTH_END'] - hps['SMOOTH_START']) * progress
        batch_count = 0

        for start in range(0, len(indices), hps['BATCH_SIZE']):
            end = min(start + hps['BATCH_SIZE'], len(indices))
            batch_idx = indices[start:end]
            q_batch = train_q_vecs[batch_idx]
            labels = torch.tensor(
                [train_pos_lists[i][0] for i in batch_idx],
                dtype=torch.long, device=device)

            optimizer.zero_grad()
            with autocast(enabled=(device.type == 'cuda')):
                table_emb = model.forward(train_data_gpu.x_dict, train_data_gpu.edge_index_dict)
                q_batch_norm = F.normalize(q_batch, p=2, dim=1)
                logits = compute_scores_chunked(q_batch_norm, table_emb, hps['CHUNK_SIZE']) / curr_temp
                loss = F.cross_entropy(logits, labels, label_smoothing=curr_smooth)

                # Hard negative margin loss
                batch_hard_negs = [hard_neg_lists[i] for i in batch_idx]
                hard_negs_tensor = torch.tensor(batch_hard_negs, device=device, dtype=torch.long)
                mask = (hard_negs_tensor != -1)
                if mask.any():
                    safe_hard_negs = hard_negs_tensor.clone()
                    safe_hard_negs[~mask] = 0
                    pos_scores = logits[range(len(batch_idx)), labels]
                    neg_scores = torch.gather(logits, 1, safe_hard_negs)
                    margin = 0.2 / curr_temp
                    losses = F.relu(neg_scores - pos_scores.unsqueeze(1) + margin) * mask.float()
                    loss = loss + 0.5 * losses.sum() / (mask.sum() + 1e-9)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hps['CLIP_GRAD_NORM'])
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            batch_count += 1

        scheduler.step()

        # Validation
        if dev_q_vecs is not None:
            dev_data_gpu = dev_data.clone().to(device)
            model.eval()
            with torch.no_grad():
                table_emb = model.forward(dev_data_gpu.x_dict, dev_data_gpu.edge_index_dict)
                q_norm = F.normalize(dev_q_vecs, p=2, dim=1)
                score_mat = compute_scores_chunked(q_norm, table_emb, hps['CHUNK_SIZE'])
                score_mat_cpu = score_mat.cpu()

                r10_sum = 0.0
                eval_count = 0
                for q_i in range(score_mat_cpu.size(0)):
                    pos_set = set(dev_pos_lists[q_i])
                    if not pos_set:
                        continue
                    eval_count += 1
                    _, top_indices = torch.topk(score_mat_cpu[q_i], k=min(10, score_mat_cpu.size(1)))
                    r10_sum += _full_recall_at_k(top_indices.tolist(), pos_set, 10)
                val_r10 = r10_sum / max(1, eval_count)

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
                print(f"      [TWIG] Epoch {epoch:2d} | Loss {total_loss / max(1, batch_count):.4f} | "
                      f"Val R@10 {val_r10:.4f} | Best {best_val_r10:.4f} (ep {best_epoch})")

            if patience_counter >= TWIG_EARLY_STOPPING_PATIENCE:
                print(f"      [TWIG] Early stopping at epoch {epoch}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    result_state = copy.deepcopy(model.state_dict())

    del model, train_data_gpu, train_q_vecs
    if dev_q_vecs is not None:
        del dev_q_vecs
    torch.cuda.empty_cache()

    return result_state


# ══════════════════════════════════════════════════════════════
# Phase 2: Query-Aware v2 Model
# ══════════════════════════════════════════════════════════════

def get_canonical_metadata(edge_mode="E4"):
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


class QueryAwareModel(nn.Module):
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
        x_dict, edge_index_dict = self._ensure_query_entries(x_dict, edge_index_dict)
        x_dict_out = self.hetero_sage(x_dict, edge_index_dict)
        x_table = x_dict_out['table']
        x_table = self.norm(x_table)
        table_features = self.proj_head(x_table)
        return F.normalize(table_features, p=2, dim=1)

    def _ensure_query_entries(self, x_dict, edge_index_dict):
        device = x_dict['table'].device
        embed_dim = x_dict['table'].size(1)
        x_dict = dict(x_dict)
        edge_index_dict = dict(edge_index_dict)

        for nt in ['table', 'column', 'page', 'query']:
            if nt not in x_dict:
                x_dict[nt] = torch.zeros((1, embed_dim), device=device)

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

        canonical_set = set(canonical_edges)
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if k in canonical_set}

        return x_dict, edge_index_dict


def load_pretrained_into_qa_model(model, pretrained_state):
    """Load TWIG weights into QA model, zero-init query parameters."""
    missing, _ = model.load_state_dict(pretrained_state, strict=False)

    query_keywords = ['queries', 'rev_queries', 'queries_page', 'rev_queries_page',
                      'queries_column', 'rev_queries_column']
    zero_count = 0
    for name, param in model.named_parameters():
        if any(kw in name for kw in query_keywords):
            nn.init.zeros_(param)
            zero_count += 1

    print(f"      [QA] Loaded TWIG weights: {len(pretrained_state) - len(missing)} matched, "
          f"{len(missing)} missing. Zero-init {zero_count} query params.")


def build_subgraph(data, query_vec, candidate_indices, edge_mode="E4", device=None):
    if device is None:
        device = query_vec.device

    sub = HeteroData()
    num_candidates = len(candidate_indices)

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

    # Node features
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

    # Query node and edges
    if edge_mode != "E0":
        sub['query'].x = query_vec.to(device)
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


def train_qa_v2(twig_state_dict, edges, dataset, key_fields, device, embedder):
    """Fine-tune Query-Aware v2 model from TWIG checkpoint. Returns QA model state dict."""
    train_graph_path = str(PROJECT_DIR / f"data/processed/train/{dataset}/graph.pt")
    val_graph_path = str(PROJECT_DIR / f"data/processed/dev/{dataset}/graph.pt")
    train_query_path = str(PROJECT_DIR / f"data/table/train/{dataset}/query.jsonl")
    val_query_path = str(PROJECT_DIR / f"data/table/dev/{dataset}/query.jsonl")

    # Load and filter
    data_cpu = torch.load(train_graph_path, map_location='cpu', weights_only=False)
    if edges:
        data_cpu = filter_edges(data_cpu, edges)
    id_to_idx = build_id_to_idx(data_cpu, key_fields)
    embed_dim = data_cpu['table'].x.size(1)
    num_tables = data_cpu['table'].num_nodes

    # Build QA model with canonical metadata
    metadata = get_canonical_metadata(QA_QUERY_EDGE_MODE)
    model = QueryAwareModel(
        embed_dim=embed_dim,
        hidden_channels=768,
        metadata=metadata,
        dropout=0.10,
        sage_aggr='min',
        hetero_aggr='max',
    ).to(device)

    # Load TWIG weights
    load_pretrained_into_qa_model(model, twig_state_dict)

    # Differential LR
    query_keywords = ['queries', 'rev_queries', 'queries_page', 'rev_queries_page',
                      'queries_column', 'rev_queries_column']
    base_params = []
    query_params = []
    for name, param in model.named_parameters():
        if any(kw in name for kw in query_keywords):
            query_params.append(param)
        else:
            base_params.append(param)

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': QA_BASE_LR},
        {'params': query_params, 'lr': QA_QUERY_LR},
    ], weight_decay=QA_WEIGHT_DECAY)

    warmup_sched = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / float(QA_WARMUP_EPOCHS)))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=QA_NUM_EPOCHS - QA_WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[QA_WARMUP_EPOCHS])
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Load queries
    texts, pos_lists = load_queries(train_query_path, id_to_idx, key_fields)
    if not texts:
        print(f"      [QA] No valid training queries")
        return None

    print(f"      [QA] Embedding {len(texts)} train queries...")
    query_vecs = torch.tensor(
        embedder.encode(texts, show_progress_bar=False),
        dtype=torch.float, device=device)

    # Load validation
    val_data_cpu = None
    val_query_vecs = None
    val_pos_lists = None
    if Path(val_graph_path).exists() and Path(val_query_path).exists():
        val_data_cpu = torch.load(val_graph_path, map_location='cpu', weights_only=False)
        if edges:
            val_data_cpu = filter_edges(val_data_cpu, edges)
        val_id_to_idx = build_id_to_idx(val_data_cpu, key_fields)
        val_texts, val_pos_lists = load_queries(val_query_path, val_id_to_idx, key_fields)
        if val_texts:
            val_query_vecs = torch.tensor(
                embedder.encode(val_texts, show_progress_bar=False),
                dtype=torch.float, device=device)

    # Train
    data = data_cpu.clone().to(device)
    hard_neg_indices = None
    best_val_recall = -1.0
    best_model_state = None
    best_epoch = 0
    patience = 0

    for epoch in range(1, QA_NUM_EPOCHS + 1):
        # Mine hard negatives
        if epoch == 1 or epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                table_emb = model.forward(data.x_dict, data.edge_index_dict)
                q_norm = F.normalize(query_vecs, p=2, dim=1)
                all_negs = []
                for start in range(0, len(query_vecs), 1024):
                    end = min(start + 1024, len(query_vecs))
                    sim = torch.matmul(q_norm[start:end], table_emb.T)
                    for i in range(end - start):
                        for pos_idx in pos_lists[start + i]:
                            if 0 <= pos_idx < table_emb.size(0):
                                sim[i, pos_idx] = -float('inf')
                    _, topk = torch.topk(sim, k=min(QA_NUM_HARD_NEGATIVES, table_emb.size(0) - 1), dim=1)
                    all_negs.extend(topk.tolist())
                hard_neg_indices = all_negs

        # Train one epoch (subgraph-based)
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
                for sample_idx in batch_idx:
                    q_vec = query_vecs[sample_idx:sample_idx + 1]
                    pos_list = pos_lists[sample_idx]
                    hard_negs = hard_neg_indices[sample_idx] if hard_neg_indices else []

                    with torch.no_grad():
                        q_norm = F.normalize(q_vec, p=2, dim=1)
                        scores = torch.matmul(q_norm, base_table_emb.T).squeeze(0)
                        _, top_k = torch.topk(scores, k=min(QA_SUBGRAPH_K, scores.size(0)))
                        candidate_set = set(top_k.cpu().tolist())

                    for idx in pos_list:
                        candidate_set.add(idx)
                    for idx in hard_negs:
                        if idx >= 0:
                            candidate_set.add(idx)

                    candidate_indices = sorted(candidate_set)
                    subgraph, table_mapping = build_subgraph(
                        data, q_vec, candidate_indices,
                        edge_mode=QA_QUERY_EDGE_MODE, device=device)

                    sub_table_emb = model.forward(subgraph.x_dict, subgraph.edge_index_dict)

                    pos_new_idx = table_mapping.get(pos_list[0], -1)
                    if pos_new_idx == -1:
                        continue

                    logits = torch.matmul(F.normalize(q_vec, p=2, dim=1), sub_table_emb.T).squeeze(0) / QA_TEMP
                    label = torch.tensor([pos_new_idx], dtype=torch.long, device=device)
                    loss_sample = F.cross_entropy(logits.unsqueeze(0), label, label_smoothing=QA_LABEL_SMOOTH)
                    batch_loss += loss_sample
                    batch_count += 1

                if batch_count == 0:
                    continue
                loss = batch_loss / batch_count

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=QA_CLIP_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

            if n_batches % 20 == 0:
                with torch.no_grad():
                    base_table_emb = model.forward(data.x_dict, data.edge_index_dict)

        scheduler.step()

        # Validation
        if val_data_cpu is not None and val_query_vecs is not None:
            val_data = val_data_cpu.clone().to(device)
            model.eval()
            with torch.no_grad():
                val_base_emb = model.forward(val_data.x_dict, val_data.edge_index_dict)
                val_q_norm = F.normalize(val_query_vecs, p=2, dim=1)
                val_coarse = torch.matmul(val_q_norm, val_base_emb.T)

                hits10 = 0
                for qi in range(len(val_query_vecs)):
                    pos_set = set(val_pos_lists[qi])
                    q_vec = val_query_vecs[qi:qi + 1]
                    _, top_k = torch.topk(val_coarse[qi], k=min(QA_COARSE_K_EVAL, val_coarse.size(1)))
                    candidates = top_k.cpu().tolist()
                    for pos_idx in pos_set:
                        if pos_idx not in set(candidates):
                            candidates.append(pos_idx)

                    subgraph, table_mapping = build_subgraph(
                        val_data, q_vec, candidates,
                        edge_mode=QA_QUERY_EDGE_MODE, device=device)
                    sub_emb = model.forward(subgraph.x_dict, subgraph.edge_index_dict)
                    rerank = torch.matmul(F.normalize(q_vec, p=2, dim=1), sub_emb.T).squeeze(0)
                    _, reranked = torch.topk(rerank, k=min(10, rerank.size(0)))
                    new_to_old = {v: k for k, v in table_mapping.items()}
                    reranked_orig = [new_to_old[idx.item()] for idx in reranked]
                    if any(orig_idx in pos_set for orig_idx in reranked_orig[:10]):
                        hits10 += 1

                val_r10 = hits10 / max(1, len(val_query_vecs))

            del val_data
            torch.cuda.empty_cache()

            if val_r10 > best_val_recall:
                best_val_recall = val_r10
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1

            if epoch % 3 == 0 or epoch == 1:
                print(f"      [QA] Epoch {epoch}/{QA_NUM_EPOCHS} | Loss {total_loss / max(1, n_batches):.4f} | "
                      f"Val R@10 {val_r10:.4f} | Best {best_val_recall:.4f} (ep {best_epoch})")

            if patience >= QA_EARLY_STOPPING_PATIENCE:
                print(f"      [QA] Early stopping at epoch {epoch}")
                break
        else:
            if epoch % 3 == 0 or epoch == 1:
                print(f"      [QA] Epoch {epoch}/{QA_NUM_EPOCHS} | Loss {total_loss / max(1, n_batches):.4f}")
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    result = copy.deepcopy(model.state_dict())
    del model, data, query_vecs
    if val_query_vecs is not None:
        del val_query_vecs
    torch.cuda.empty_cache()
    return result


# ══════════════════════════════════════════════════════════════
# Phase 3: QA Evaluation
# ══════════════════════════════════════════════════════════════

def full_recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def reciprocal_rank(retrieved, relevant):
    for i, rid in enumerate(retrieved, 1):
        if rid in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved, relevant, k):
    dcg = sum(1.0 / math.log2(i + 1) for i, rid in enumerate(retrieved[:k], 1) if rid in relevant)
    num_rel = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_qa(twig_state_dict, qa_state_dict, edges, dataset, key_fields, device, embedder):
    """Evaluate QA v2 model on test set."""
    test_graph_path = str(PROJECT_DIR / f"data/processed/test/{dataset}/graph.pt")
    test_query_path = str(PROJECT_DIR / f"data/table/test/{dataset}/query.jsonl")

    if not Path(test_graph_path).exists():
        print(f"      [Eval] Skipping: {test_graph_path} not found")
        return None

    data_full = torch.load(test_graph_path, map_location=device, weights_only=False)
    id_to_idx = build_id_to_idx(data_full, key_fields)
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    mapping_keys = set(idx_to_id.values())
    embed_dim = data_full['table'].x.size(1)

    # Filtered graph for QA subgraphs
    if edges:
        data_filtered = filter_edges(data_full, edges)
    else:
        data_filtered = data_full

    # Load QA model
    metadata = get_canonical_metadata(QA_QUERY_EDGE_MODE)
    qa_model = QueryAwareModel(
        embed_dim=embed_dim,
        hidden_channels=768,
        metadata=metadata,
        dropout=0.10,
        sage_aggr='min',
        hetero_aggr='max',
    ).to(device)
    qa_model.load_state_dict(qa_state_dict, strict=False)
    qa_model.eval()

    # Load TWIG model for coarse ranking (on full graph)
    twig_metadata = data_full.metadata()
    twig_model = DiffusionModel(
        embed_dim=embed_dim,
        hidden_channels=768,
        metadata=twig_metadata,
        dropout=0.10,
        sage_aggr='min',
        hetero_aggr='max',
    ).to(device)
    twig_model.load_state_dict(twig_state_dict, strict=False)
    twig_model.eval()

    # Parse queries
    queries = []
    with open(test_query_path, 'r', encoding='utf-8') as f:
        for line in f:
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
            queries.append((question, gt_keys))

    questions = [q for q, _ in queries]
    relevants = [gt for _, gt in queries]
    total = len(queries)
    eval_count = sum(1 for gt in relevants if len(gt) > 0)

    # Embed queries
    print(f"      [Eval] Embedding {total} test queries...")
    query_vecs = torch.tensor(
        embedder.encode(questions, show_progress_bar=False),
        dtype=torch.float, device=device)
    query_vecs = F.normalize(query_vecs, p=2, dim=1)

    # Coarse ranking with TWIG
    data_full = data_full.to(device)
    data_filtered = data_filtered.to(device)
    with torch.no_grad():
        table_emb_fixed = twig_model.forward(data_full.x_dict, data_full.edge_index_dict)
        coarse_scores = torch.matmul(query_vecs, table_emb_fixed.T)

    del twig_model
    torch.cuda.empty_cache()

    # TWIG baseline
    e0_r1 = e0_r5 = e0_r10 = e0_mrr = 0.0
    for qi in range(total):
        gt = relevants[qi]
        if not gt:
            continue
        _, top_idx = torch.topk(coarse_scores[qi], k=min(EVAL_TOP_K, coarse_scores.size(1)))
        retrieved = [idx_to_id.get(idx.item(), "") for idx in top_idx]
        e0_r1 += full_recall_at_k(retrieved, gt, 1)
        e0_r5 += full_recall_at_k(retrieved, gt, 5)
        e0_r10 += full_recall_at_k(retrieved, gt, 10)
        e0_mrr += reciprocal_rank(retrieved, gt)

    # QA reranking
    r1 = r5 = r10 = mrr_sum = ndcg10_sum = 0.0
    with torch.no_grad():
        for qi in tqdm(range(total), desc="      QA Rerank", leave=False):
            gt = relevants[qi]
            if not gt:
                continue

            q_vec = query_vecs[qi:qi + 1]
            top_k_scores, top_k_idx = torch.topk(
                coarse_scores[qi], k=min(EVAL_COARSE_K, coarse_scores.size(1)))
            candidate_indices = top_k_idx.cpu().tolist()

            subgraph, table_mapping = build_subgraph(
                data_filtered, q_vec, candidate_indices,
                edge_mode=QA_QUERY_EDGE_MODE, device=device)
            sub_table_emb = qa_model.forward(subgraph.x_dict, subgraph.edge_index_dict)

            rerank_scores = torch.matmul(q_vec, sub_table_emb.T).squeeze(0)
            coarse_in_sub = torch.zeros(rerank_scores.size(0), device=device)
            for orig_rank, orig_idx in enumerate(candidate_indices):
                new_idx = table_mapping.get(orig_idx, -1)
                if 0 <= new_idx < coarse_in_sub.size(0):
                    coarse_in_sub[new_idx] = top_k_scores[orig_rank]

            final_scores = rerank_scores  # 直接用 QA rerank 分數（不與粗排插值）
            _, reranked = torch.topk(final_scores, k=min(EVAL_TOP_K, final_scores.size(0)))

            new_to_old = {v: k for k, v in table_mapping.items()}
            retrieved = [idx_to_id.get(new_to_old.get(idx.item(), -1), "") for idx in reranked]

            r1 += full_recall_at_k(retrieved, gt, 1)
            r5 += full_recall_at_k(retrieved, gt, 5)
            r10 += full_recall_at_k(retrieved, gt, 10)
            mrr_sum += reciprocal_rank(retrieved, gt)
            ndcg10_sum += ndcg_at_k(retrieved, gt, 10)

    del qa_model, data_full, data_filtered, query_vecs
    torch.cuda.empty_cache()

    results = {
        'eval_count': eval_count,
        'QA_R@1': r1 / max(1, eval_count),
        'QA_R@5': r5 / max(1, eval_count),
        'QA_R@10': r10 / max(1, eval_count),
        'QA_MRR': mrr_sum / max(1, eval_count),
        'QA_nDCG@10': ndcg10_sum / max(1, eval_count),
        'TWIG_R@1': e0_r1 / max(1, eval_count),
        'TWIG_R@5': e0_r5 / max(1, eval_count),
        'TWIG_R@10': e0_r10 / max(1, eval_count),
        'TWIG_MRR': e0_mrr / max(1, eval_count),
    }
    return results


# ══════════════════════════════════════════════════════════════
# Incremental result saving
# ══════════════════════════════════════════════════════════════

def save_result_incremental(result, dataset, results_dir):
    results_csv = f"{results_dir}/results.csv"
    results_json = f"{results_dir}/results.json"

    existing_results = []
    if os.path.exists(results_json):
        try:
            with open(results_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing_results = data.get('results', [])
        except Exception:
            pass

    results_dict = {r['code']: r for r in existing_results}
    results_dict[result['code']] = result
    final_results = sorted(results_dict.values(), key=lambda x: x['code'])

    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['config', 'code', 'binary', 'label', 'num_edges', 'kept_edges',
                      'QA_R@1', 'QA_R@5', 'QA_R@10', 'QA_MRR', 'QA_nDCG@10',
                      'TWIG_R@1', 'TWIG_R@5', 'TWIG_R@10', 'TWIG_MRR',
                      'twig_best_epoch', 'qa_best_epoch', 'elapsed_min']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(final_results)

    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Query-Aware v2 Edge Ablation (A0-A63)',
            'dataset': dataset,
            'edge_types': ALL_EDGE_RELATIONS,
            'edge_abbreviations': EDGE_ABBREV,
            'timestamp': datetime.now().isoformat(),
            'twig_hps': TWIG_HPS,
            'qa_hps': {
                'QUERY_EDGE_MODE': QA_QUERY_EDGE_MODE,
                'BASE_LR': QA_BASE_LR,
                'QUERY_LR': QA_QUERY_LR,
                'NUM_EPOCHS': QA_NUM_EPOCHS,
                'SUBGRAPH_K': QA_SUBGRAPH_K,
            },
            'results': final_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"      Saved {result['config']} to results")


def load_completed_configs(dataset, results_dir):
    tracking_file = f"{results_dir}/{dataset}/completed_configs.json"
    if not os.path.exists(tracking_file):
        return set()
    try:
        with open(tracking_file, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    except Exception:
        return set()


def update_completed_configs(dataset, config_code, results_dir):
    tracking_file = f"{results_dir}/{dataset}/completed_configs.json"
    completed = load_completed_configs(dataset, results_dir)
    completed.add(config_code)
    with open(tracking_file, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(completed)), f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════
# Main ablation runner
# ══════════════════════════════════════════════════════════════

def run_single_ablation(config_code, dataset, device, embedder):
    """Run one QA edge ablation: TWIG train -> QA fine-tune -> evaluate."""
    edges = binary_to_edges(config_code)
    config_name = f"A{config_code}"
    config_label = get_config_label(config_code)
    binary_repr = get_binary_repr(config_code)
    key_fields = get_key_fields(dataset)

    print(f"\n  {'='*65}")
    print(f"  [{config_name}] {config_label}")
    print(f"  Binary: {binary_repr} | Edges: {len(edges)}/6 | Dataset: {dataset}")
    print(f"  {'='*65}")

    t_start = time.time()

    # Phase 1: Train TWIG
    print(f"    Phase 1: Training TWIG base model...")
    twig_state = train_twig(edges, dataset, key_fields, device, embedder)
    if twig_state is None:
        return {
            'config': config_name, 'code': config_code, 'binary': binary_repr,
            'label': config_label, 'num_edges': len(edges),
            'kept_edges': ', '.join(edges) if edges else 'none',
            'skipped': True, 'skip_reason': 'TWIG training failed',
        }

    # Phase 2: Fine-tune QA v2
    print(f"    Phase 2: Fine-tuning Query-Aware v2...")
    qa_state = train_qa_v2(twig_state, edges, dataset, key_fields, device, embedder)
    if qa_state is None:
        return {
            'config': config_name, 'code': config_code, 'binary': binary_repr,
            'label': config_label, 'num_edges': len(edges),
            'kept_edges': ', '.join(edges) if edges else 'none',
            'skipped': True, 'skip_reason': 'QA training failed',
        }

    # Phase 3: Evaluate
    print(f"    Phase 3: Evaluating on test set...")
    eval_results = evaluate_qa(twig_state, qa_state, edges, dataset, key_fields, device, embedder)
    if eval_results is None:
        return {
            'config': config_name, 'code': config_code, 'binary': binary_repr,
            'label': config_label, 'num_edges': len(edges),
            'kept_edges': ', '.join(edges) if edges else 'none',
            'skipped': True, 'skip_reason': 'Evaluation failed',
        }

    elapsed = time.time() - t_start

    result = {
        'config': config_name,
        'code': config_code,
        'binary': binary_repr,
        'label': config_label,
        'num_edges': len(edges),
        'kept_edges': ', '.join(edges) if edges else 'none',
        'elapsed_min': round(elapsed / 60, 1),
        **eval_results,
    }

    del twig_state, qa_state
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Query-Aware v2 Edge Ablation Study (A0-A63)')

    parser.add_argument('--datasets', nargs='+', default=None,
                        help=f'Datasets to process. Default: {" ".join(DEFAULT_DATASETS)}')
    parser.add_argument('--ablation', nargs='+', type=int, default=None,
                        help='Ablation config codes (0-63). Default: all 64')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')

    args = parser.parse_args()
    DATASETS = args.datasets if args.datasets is not None else DEFAULT_DATASETS
    ABLATION = sorted(set(args.ablation)) if args.ablation is not None else DEFAULT_ABLATION

    print(f"Using GPU: {args.gpu}")
    print(f"Datasets: {DATASETS}")
    print(f"Ablation configs: {len(ABLATION)} configurations")
    print(f"Total experiments: {len(DATASETS) * len(ABLATION)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load embedder once
    print("Loading sentence embedder...")
    embedder = SentenceTransformer(MODEL_NAME, device=str(device))

    for dataset in DATASETS:
        print(f"\n{'='*70}")
        print(f"  DATASET: {dataset.upper()}")
        print(f"{'='*70}")

        results_dir = f"{RESULTS_DIR}/{dataset}"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        existing_codes = load_completed_configs(dataset, RESULTS_DIR)
        configs_to_run = [c for c in ABLATION if c not in existing_codes]

        if len(configs_to_run) < len(ABLATION):
            skipped = set(ABLATION) - set(configs_to_run)
            print(f"  Skipping already completed: {sorted(skipped)}")
        print(f"  Configs to run: {len(configs_to_run)}")

        for config_code in configs_to_run:
            try:
                result = run_single_ablation(config_code, dataset, device, embedder)

                if result.get('skipped', False):
                    print(f"    Skipped: {result.get('skip_reason', 'Unknown')}")
                    continue

                # Print result
                print(f"\n  Result for {result['config']} ({result['binary']}): {result['label']}")
                print(f"    TWIG:  R@1={result['TWIG_R@1']:.4f}  R@5={result['TWIG_R@5']:.4f}  "
                      f"R@10={result['TWIG_R@10']:.4f}  MRR={result['TWIG_MRR']:.4f}")
                print(f"    QA v2: R@1={result['QA_R@1']:.4f}  R@5={result['QA_R@5']:.4f}  "
                      f"R@10={result['QA_R@10']:.4f}  MRR={result['QA_MRR']:.4f}")
                dr10 = result['QA_R@10'] - result['TWIG_R@10']
                print(f"    Delta R@10: {'+'if dr10>0 else ''}{dr10:.4f}  |  Time: {result['elapsed_min']} min")

                save_result_incremental(result, dataset, results_dir)
                update_completed_configs(dataset, config_code, RESULTS_DIR)

            except Exception as e:
                print(f"\n  ERROR in A{config_code}: {e}")
                import traceback
                traceback.print_exc()

        # Summary
        results_json = f"{results_dir}/results.json"
        if os.path.exists(results_json):
            with open(results_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                final_results = data.get('results', [])

            if final_results:
                print(f"\n  {'='*90}")
                print(f"  QA EDGE ABLATION SUMMARY - {dataset.upper()}")
                print(f"  {'='*90}")
                print(f"  {'Config':<7} {'Binary':<8} {'Label':<28} {'TWIG R@10':>9} {'QA R@10':>9} {'Delta':>7}")
                print(f"  {'─'*7} {'─'*8} {'─'*28} {'─'*9} {'─'*9} {'─'*7}")
                for r in final_results:
                    dr = r.get('QA_R@10', 0) - r.get('TWIG_R@10', 0)
                    print(f"  {r['config']:<7} {r['binary']:<8} {r['label']:<28} "
                          f"{r.get('TWIG_R@10', 0):.4f}   {r.get('QA_R@10', 0):.4f}   "
                          f"{'+'if dr>0 else ''}{dr:.4f}")

    del embedder
    torch.cuda.empty_cache()
    print(f"\n{'='*70}")
    print(f"  ALL DATASETS COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
