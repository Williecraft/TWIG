#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Retrain TWIG with per-dataset best edge config and save model weights.

This replicates the edge ablation training (run_edge_ablation.py) but only
for the best config per dataset, and saves the trained model to:
  checkpoints/{dataset}/model_best_edges.pt

Usage:
  cd src && python query_aware/train_twig_best.py                # all datasets
  cd src && python query_aware/train_twig_best.py feta ottqa     # specific datasets
"""

import sys
import os
import copy
import random
import json

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path

# Import from parent src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_model import (
    DiffusionModel, get_embedder, make_key,
    compute_scores_chunked, mine_hard_negatives_topk,
)
from run_edge_ablation import filter_edges, build_id_to_idx, load_queries

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ===========================
# Config
# ===========================

BEST_EDGE_CONFIGS = {
    "feta": ["has_column", "same_page"],
    "ottqa": ["has_column", "similar_content"],
    "mimo_en": ["has_column", "similar_content", "shared_column_name"],
    "mimo_ch": ["similar_content"],
    "e2ewtq": ["similar_content"],
    "mmqa": ["similar_table", "has_column", "comes_from", "same_page", "similar_content"],
}

# Same hyperparameters as run_edge_ablation.py
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

PROJECT_DIR = "/user_data/TabGNN"


def get_key_fields(dataset_name):
    if dataset_name in ["ottqa", "feta", "e2ewtq"]:
        return ("sheet_name", "file_name")
    elif dataset_name in ["mimo_en", "mimo_ch", "mmqa"]:
        return ("id",)
    return ("id",)


def build_graph_hard_negatives(data, num_hard, pos_lists):
    """Build initial hard negatives from graph structure (random negatives)."""
    num_tables = data['table'].num_nodes
    all_negs = []
    for pos_indices in pos_lists:
        pos_set = set(pos_indices)
        negs = []
        for _ in range(num_hard):
            neg = random.randint(0, num_tables - 1)
            while neg in pos_set:
                neg = random.randint(0, num_tables - 1)
            negs.append(neg)
        all_negs.append(negs)
    return all_negs


def train_one_epoch(model, data, optimizer, scaler, scheduler,
                    query_vectors, pos_indices_lists, hard_neg_indices,
                    hps, epoch, device):
    """One epoch of training (same as run_edge_ablation.py)."""
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hps['CLIP_GRAD_NORM'])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        batch_count += 1

    scheduler.step()
    return total_loss / max(1, batch_count)


def evaluate_recall(model, data, query_vectors, pos_indices_lists, hps, device, dataset=''):
    """Evaluate Full Recall@k."""
    k1 = 2 if dataset == 'mmqa' else 1
    model.eval()
    with torch.no_grad():
        table_emb = model.forward(data.x_dict, data.edge_index_dict)
        q_norm = F.normalize(query_vectors, p=2, dim=1)
        score_mat = compute_scores_chunked(q_norm, table_emb, hps['CHUNK_SIZE'])
        score_mat_cpu = score_mat.cpu()

        recall_k1 = recall5 = recall10 = 0.0
        eval_count = 0

        for q_i in range(score_mat_cpu.size(0)):
            pos_set = set(pos_indices_lists[q_i])
            if not pos_set:
                continue
            eval_count += 1
            _, top_indices = torch.topk(score_mat_cpu[q_i], k=min(10, score_mat_cpu.size(1)))
            retrieved = top_indices.tolist()
            recall_k1 += len(set(retrieved[:k1]) & pos_set) / len(pos_set)
            recall5 += len(set(retrieved[:5]) & pos_set) / len(pos_set)
            recall10 += len(set(retrieved[:10]) & pos_set) / len(pos_set)

    if eval_count == 0:
        return {f'recall@{k1}': 0.0, 'recall@5': 0.0, 'recall@10': 0.0}
    return {
        f'recall@{k1}': recall_k1 / eval_count,
        'recall@5': recall5 / eval_count,
        'recall@10': recall10 / eval_count,
    }


def train_twig_best(dataset):
    """Train TWIG with best edge config for a dataset."""
    key_fields = get_key_fields(dataset)
    best_edges = BEST_EDGE_CONFIGS.get(dataset, [])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = f"{PROJECT_DIR}/checkpoints/{dataset}/model_best_edges.pt"

    # Skip if already trained
    if Path(save_path).exists():
        print(f"\n  {dataset}: model_best_edges.pt already exists, skipping")
        return save_path

    print(f"\n{'='*60}")
    print(f"Training TWIG with best edges: {dataset}")
    print(f"Edges: {best_edges}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Load and filter train graph
    train_graph = f"{PROJECT_DIR}/data/processed/train/{dataset}/graph.pt"
    train_data_full = torch.load(train_graph, map_location='cpu', weights_only=False)
    train_data = filter_edges(train_data_full, best_edges)
    del train_data_full

    train_id_to_idx = build_id_to_idx(train_data, key_fields)
    embed_dim = train_data['table'].x.size(1)

    # Load training queries
    train_query = f"{PROJECT_DIR}/data/table/train/{dataset}/query.jsonl"
    train_texts, train_pos_lists = load_queries(train_query, train_id_to_idx, key_fields)
    print(f"  Train queries: {len(train_texts)}")

    hard_neg_lists = build_graph_hard_negatives(train_data, NUM_HARD_NEGATIVES, train_pos_lists)

    # Embed queries
    embedder = get_embedder(device=str(device))
    print("  Embedding train queries...")
    train_q_vecs = torch.tensor(
        embedder.encode(train_texts, show_progress_bar=True),
        dtype=torch.float, device=device)

    # Load dev graph
    dev_graph = f"{PROJECT_DIR}/data/processed/dev/{dataset}/graph.pt"
    dev_query = f"{PROJECT_DIR}/data/table/dev/{dataset}/query.jsonl"
    dev_data = None
    dev_q_vecs = None
    dev_pos_lists = None

    if Path(dev_graph).exists():
        dev_data_full = torch.load(dev_graph, map_location='cpu', weights_only=False)
        dev_data = filter_edges(dev_data_full, best_edges)
        del dev_data_full
        dev_id_to_idx = build_id_to_idx(dev_data, key_fields)
        dev_texts, dev_pos_lists = load_queries(dev_query, dev_id_to_idx, key_fields)
        print(f"  Dev queries: {len(dev_texts)}")
        if dev_texts:
            dev_q_vecs = torch.tensor(
                embedder.encode(dev_texts, show_progress_bar=False),
                dtype=torch.float, device=device)

    del embedder
    torch.cuda.empty_cache()

    # Setup model
    train_data_gpu = train_data.clone().to(device)
    model = DiffusionModel(
        embed_dim=embed_dim,
        hidden_channels=HPS['HIDDEN_CHANNELS'],
        metadata=train_data_gpu.metadata(),
        dropout=HPS['DROPOUT'],
        sage_aggr=HPS.get('SAGE_AGGR', 'mean'),
        hetero_aggr=HPS.get('HETERO_AGGR', 'sum'),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=HPS['LEARNING_RATE'],
                            weight_decay=HPS['WEIGHT_DECAY'])
    warmup_scheduler = LambdaLR(
        optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / float(HPS['WARMUP_EPOCHS'])))
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=HPS['NUM_EPOCHS'] - HPS['WARMUP_EPOCHS'])
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[HPS['WARMUP_EPOCHS']])
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Train
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
        if dev_data is not None and dev_q_vecs is not None:
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
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Loaded best model from epoch {best_epoch}")

    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'hps': HPS,
        'best_edges': best_edges,
        'best_epoch': best_epoch,
        'best_val_r10': best_val_r10,
    }, save_path)
    print(f"  Model saved to {save_path} (epoch={best_epoch}, val R@10={best_val_r10:.4f})")

    del model, train_data_gpu
    torch.cuda.empty_cache()
    return save_path


if __name__ == '__main__':
    datasets = sys.argv[1:] if len(sys.argv) > 1 else [
        "feta", "ottqa", "mimo_en", "mimo_ch", "e2ewtq", "mmqa"
    ]
    for ds in datasets:
        graph_path = Path(f"{PROJECT_DIR}/data/processed/train/{ds}/graph.pt")
        if not graph_path.exists():
            print(f"Skipping {ds}: {graph_path} not found")
            continue
        train_twig_best(ds)
