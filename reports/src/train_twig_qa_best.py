#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用 QA ablation 最佳邊配置重新訓練 TWIG base model，
儲存為 checkpoints/{dataset}/model_qa_best_edges.pt

這是修正「Stage1 用 TWIG-best 邊 / Stage2 用 QA-best 邊」不一致問題的第一步。
接著 train_query_aware_v2.py 應從 model_qa_best_edges.pt fine-tune。

Usage:
  cd reports/src && python train_twig_qa_best.py
  cd reports/src && python train_twig_qa_best.py --datasets feta ottqa
"""

import argparse
import os
import sys

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument('--gpu', type=int, default=0)
_args, _ = _p.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)

import argparse
import copy
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphSAGE, to_hetero, GraphNorm
from torch import nn
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

MODEL_NAME = 'BAAI/bge-m3'

# QA ablation 最佳邊配置（與 train_query_aware_v2.py 的 BEST_EDGE_CONFIGS 一致）
QA_BEST_EDGE_CONFIGS = {
    "feta":    ["similar_table", "has_column", "comes_from", "same_page", "shared_column_name"],
    "ottqa":   ["similar_table", "has_column", "comes_from", "same_page", "similar_content", "shared_column_name"],
    "mimo_en": ["similar_table", "comes_from", "same_page", "shared_column_name"],
    "mimo_ch": ["has_column", "comes_from", "same_page", "similar_content"],
    "e2ewtq":  ["has_column", "shared_column_name"],
    "mmqa":    ["has_column", "comes_from", "similar_content", "shared_column_name"],
}

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
NUM_HARD_NEGATIVES = 3
EARLY_STOPPING_PATIENCE = 10


def get_key_fields(dataset):
    return ("sheet_name", "file_name") if dataset in ["ottqa", "feta", "e2ewtq"] else ("id",)


def make_key(item, kf):
    return "|".join(str(item.get(f, "")) for f in kf)


def filter_edges(data, keep_relations):
    filtered = HeteroData()
    for nt in data.node_types:
        for attr in data[nt].keys():
            filtered[nt][attr] = data[nt][attr]

    fwd_to_rev = {
        'has_column': 'rev_has_column', 'comes_from': 'rev_comes_from',
        'similar_table': 'similar_table', 'same_page': 'same_page',
        'similar_content': 'similar_content', 'shared_column_name': 'shared_column_name',
    }
    rev_to_fwd = {'rev_has_column': 'has_column', 'rev_comes_from': 'comes_from'}
    keep = set()
    for r in keep_relations:
        keep.add(r)
        if r in fwd_to_rev: keep.add(fwd_to_rev[r])

    for et, ei in data.edge_index_dict.items():
        _, rel, _ = et
        if rel in keep or rev_to_fwd.get(rel, rel) in keep_relations:
            filtered[et].edge_index = ei

    dest_types = {dst for _, _, dst in filtered.edge_types} if filtered.edge_types else set()
    for nt in filtered.node_types:
        if nt not in dest_types:
            filtered[nt, f'_self_loop_{nt}', nt].edge_index = torch.tensor([[0],[0]], dtype=torch.long)

    if hasattr(data, 'metadata_maps'):
        filtered.metadata_maps = data.metadata_maps
    return filtered


class DiffusionModel(nn.Module):
    def __init__(self, embed_dim, hidden_channels, metadata, dropout=0.1,
                 sage_aggr='min', hetero_aggr='max'):
        super().__init__()
        self.sage = GraphSAGE(in_channels=embed_dim, hidden_channels=hidden_channels,
                              num_layers=2, out_channels=hidden_channels, aggr=sage_aggr)
        self.hetero_sage = to_hetero(self.sage, metadata, aggr=hetero_aggr)
        self.norm = GraphNorm(hidden_channels)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), nn.Linear(hidden_channels, embed_dim))

    def forward(self, x_dict, edge_index_dict):
        out = self.hetero_sage(x_dict, edge_index_dict)
        x = self.norm(out['table'])
        return F.normalize(self.proj_head(x), p=2, dim=1)


def mine_hard_negatives(model, data, q_vecs, pos_lists, k, device):
    model.eval()
    with torch.no_grad():
        tbl = model.forward(data.x_dict, data.edge_index_dict)
        qn  = F.normalize(q_vecs, p=2, dim=1)
        all_negs = []
        for start in range(0, len(q_vecs), 1024):
            end = min(start + 1024, len(q_vecs))
            sim = torch.matmul(qn[start:end], tbl.T)
            for i in range(end - start):
                for pi in pos_lists[start + i]:
                    if 0 <= pi < tbl.size(0):
                        sim[i, pi] = -float('inf')
            _, topk = torch.topk(sim, k=min(k, tbl.size(0)-1), dim=1)
            all_negs.extend(topk.tolist())
    model.train()
    return all_negs


def train_twig(dataset, edges, device, embedder):
    key_fields = get_key_fields(dataset)
    hps = TWIG_HPS

    train_graph = torch.load(PROJECT_DIR / f'data/processed/train/{dataset}/graph.pt',
                             map_location='cpu', weights_only=False)
    dev_graph   = torch.load(PROJECT_DIR / f'data/processed/dev/{dataset}/graph.pt',
                             map_location='cpu', weights_only=False)

    train_data = filter_edges(train_graph, edges)
    dev_data   = filter_edges(dev_graph,   edges)
    del train_graph, dev_graph

    embed_dim = train_data['table'].x.size(1)

    id_to_idx = {}
    if hasattr(train_data, 'metadata_maps') and 'table_meta' in train_data.metadata_maps:
        for i, m in enumerate(train_data.metadata_maps['table_meta']):
            id_to_idx[make_key(m, key_fields)] = i

    texts, pos_lists = [], []
    with open(PROJECT_DIR / f'data/table/train/{dataset}/query.jsonl') as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get('question') or (obj.get('questions') or [''])[0]
            if not q: continue
            pos = [id_to_idx[make_key(g, key_fields)]
                   for g in obj.get('ground_truth_list', []) or []
                   if make_key(g, key_fields) in id_to_idx]
            if pos:
                texts.append(q.strip()); pos_lists.append(pos)

    print(f'  Embedding {len(texts)} train queries...')
    q_vecs = torch.tensor(embedder.encode(texts, show_progress_bar=False),
                          dtype=torch.float, device=device)

    val_texts, val_pos_lists = [], []
    val_id_to_idx = {}
    if hasattr(dev_data, 'metadata_maps') and 'table_meta' in dev_data.metadata_maps:
        for i, m in enumerate(dev_data.metadata_maps['table_meta']):
            val_id_to_idx[make_key(m, key_fields)] = i
    with open(PROJECT_DIR / f'data/table/dev/{dataset}/query.jsonl') as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get('question') or (obj.get('questions') or [''])[0]
            if not q: continue
            pos = [val_id_to_idx[make_key(g, key_fields)]
                   for g in obj.get('ground_truth_list', []) or []
                   if make_key(g, key_fields) in val_id_to_idx]
            if pos:
                val_texts.append(q.strip()); val_pos_lists.append(pos)
    val_q_vecs = torch.tensor(embedder.encode(val_texts, show_progress_bar=False),
                              dtype=torch.float, device=device) if val_texts else None

    data_gpu = train_data.clone().to(device)
    model = DiffusionModel(embed_dim, hps['HIDDEN_CHANNELS'], data_gpu.metadata(),
                           hps['DROPOUT'], hps['SAGE_AGGR'], hps['HETERO_AGGR']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=hps['LEARNING_RATE'],
                            weight_decay=hps['WEIGHT_DECAY'])
    warmup = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e+1)/hps['WARMUP_EPOCHS']))
    cosine = CosineAnnealingLR(optimizer, T_max=hps['NUM_EPOCHS']-hps['WARMUP_EPOCHS'])
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[hps['WARMUP_EPOCHS']])
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    hard_negs = [[-1]*NUM_HARD_NEGATIVES for _ in pos_lists]
    best_val_r10, best_epoch, patience, best_state = -1.0, 0, 0, None

    for epoch in range(1, hps['NUM_EPOCHS']+1):
        hard_negs = mine_hard_negatives(model, data_gpu, q_vecs, pos_lists,
                                        NUM_HARD_NEGATIVES, device)
        model.train()
        progress   = epoch / hps['NUM_EPOCHS']
        curr_temp  = hps['TEMP_END'] if epoch > hps['NUM_EPOCHS']*0.7 else hps['TEMP_START']
        curr_smooth= hps['SMOOTH_START'] + (hps['SMOOTH_END']-hps['SMOOTH_START'])*progress
        indices = list(range(len(q_vecs))); random.shuffle(indices)
        total_loss = 0.0; n_batches = 0

        for start in range(0, len(indices), hps['BATCH_SIZE']):
            b = indices[start:start+hps['BATCH_SIZE']]
            qb = q_vecs[b]
            labels = torch.tensor([pos_lists[i][0] for i in b], dtype=torch.long, device=device)
            optimizer.zero_grad()
            with autocast(enabled=(device.type == 'cuda')):
                tbl = model.forward(data_gpu.x_dict, data_gpu.edge_index_dict)
                logits = torch.matmul(F.normalize(qb, p=2, dim=1), tbl.T) / curr_temp
                loss = F.cross_entropy(logits, labels, label_smoothing=curr_smooth)
                hn = torch.tensor([hard_negs[i] for i in b], device=device, dtype=torch.long)
                mask = (hn != -1)
                if mask.any():
                    safe = hn.clone(); safe[~mask] = 0
                    pos_sc = logits[range(len(b)), labels]
                    neg_sc = torch.gather(logits, 1, safe)
                    loss = loss + 0.5 * (F.relu(neg_sc - pos_sc.unsqueeze(1) + 0.2/curr_temp) * mask.float()).sum() / (mask.sum()+1e-9)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hps['CLIP_GRAD_NORM'])
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item(); n_batches += 1

        scheduler.step()

        if val_q_vecs is not None:
            val_data_gpu = dev_data.clone().to(device)
            model.eval()
            with torch.no_grad():
                val_tbl = model.forward(val_data_gpu.x_dict, val_data_gpu.edge_index_dict)
                val_scores = torch.matmul(F.normalize(val_q_vecs, p=2, dim=1), val_tbl.T).cpu()
                r10 = sum(
                    1 for qi in range(val_scores.size(0))
                    if any(idx.item() in set(val_pos_lists[qi])
                           for idx in torch.topk(val_scores[qi], k=min(10, val_scores.size(1))).indices)
                ) / max(1, val_scores.size(0))
            del val_data_gpu; torch.cuda.empty_cache()

            if r10 > best_val_r10:
                best_val_r10 = r10; best_epoch = epoch; patience = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f'    Epoch {epoch:2d} | Loss {total_loss/max(1,n_batches):.4f} | '
                      f'Val R@10 {r10:.4f} | Best {best_val_r10:.4f} (ep {best_epoch})')

            if patience >= EARLY_STOPPING_PATIENCE:
                print(f'    Early stopping at epoch {epoch}')
                break

    if best_state:
        model.load_state_dict(best_state)
    return copy.deepcopy(model.state_dict()), best_val_r10, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['feta','ottqa','mimo_en','mimo_ch','e2ewtq','mmqa'])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    embedder = SentenceTransformer(MODEL_NAME, device=str(device))

    for dataset in args.datasets:
        edges = QA_BEST_EDGE_CONFIGS[dataset]
        save_path = PROJECT_DIR / f'checkpoints/{dataset}/model_qa_best_edges.pt'

        print(f'\n{"="*60}')
        print(f'Training TWIG for {dataset} with QA-best edges')
        print(f'  Edges: {edges}')
        print(f'{"="*60}')

        state, best_r10, best_epoch = train_twig(dataset, edges, device, embedder)

        torch.save({
            'model_state_dict': state,
            'hps': TWIG_HPS,
            'best_edges': edges,
            'best_val_r10': best_r10,
            'best_epoch': best_epoch,
        }, save_path)
        print(f'  Saved → {save_path}  (val R@10={best_r10:.4f}, epoch={best_epoch})')

        del state; torch.cuda.empty_cache()

    del embedder
    print('\nDone.')


if __name__ == '__main__':
    main()
