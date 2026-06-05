#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用 QA ablation dev-best 邊配置重新訓練 TWIG + QA，儲存兩個 checkpoint：
  checkpoints/{dataset}/model_qa_best_edges.pt      (TWIG)
  checkpoints/{dataset}/model_qa_best_edges_qa.pt   (QA fine-tune)

供 evaluate_full_corpus.py 載入後對全語料庫評估用。
選邊原則：以 dev QA_R@10 選出（run_qa_edge_ablation.py 結果，不用 test）。

Usage:
  cd reports/src && python train_twig_qa_best.py
  cd reports/src && python train_twig_qa_best.py --datasets feta ottqa --gpu 0
"""

import argparse
import os
import sys

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument('--gpu', type=int, default=0)
_args, _ = _p.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)

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

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

# Import QA model and helpers from the ablation script (single source of truth)
from query_aware.run_qa_edge_ablation import (
    QueryAwareModel,
    get_canonical_metadata,
    load_pretrained_into_qa_model,
    build_subgraph,
    build_id_to_idx,
    load_queries,
    filter_edges,
    QA_QUERY_EDGE_MODE,
    QA_SUBGRAPH_K,
    QA_COARSE_K_EVAL,
    QA_NUM_EPOCHS,
    QA_WARMUP_EPOCHS,
    QA_BATCH_SIZE,
    QA_NUM_HARD_NEGATIVES,
    QA_EARLY_STOPPING_PATIENCE,
    QA_BASE_LR,
    QA_QUERY_LR,
    QA_WEIGHT_DECAY,
    QA_CLIP_GRAD_NORM,
    QA_TEMP,
    QA_LABEL_SMOOTH,
)

MODEL_NAME = 'BAAI/bge-m3'

# dev-set selected best edge configs (run_qa_edge_ablation.py, dev QA_R@10)
QA_BEST_EDGE_CONFIGS = {
    "feta":    ['has_column', 'same_page'],                                             # A20 dev=0.9760
    "ottqa":   ['similar_table', 'has_column', 'same_page', 'similar_content'],         # A54 dev=0.9982
    "mimo_en": ['has_column', 'comes_from', 'same_page', 'similar_content',
                'shared_column_name'],                                                   # A31 dev=0.7448
    "mimo_ch": ['has_column', 'comes_from', 'same_page'],                               # A28 dev=0.7194
    "e2ewtq":  ['has_column', 'comes_from'],                                            # A24 dev=0.9500
    "mmqa":    ['has_column', 'shared_column_name'],                                    # A17 dev=0.8273
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

    print(f'  [TWIG] Embedding {len(texts)} train queries...')
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
                print(f'  [TWIG] Epoch {epoch:2d} | Loss {total_loss/max(1,n_batches):.4f} | '
                      f'Val R@10 {r10:.4f} | Best {best_val_r10:.4f} (ep {best_epoch})')

            if patience >= EARLY_STOPPING_PATIENCE:
                print(f'  [TWIG] Early stopping at epoch {epoch}')
                break

    if best_state:
        model.load_state_dict(best_state)
    return copy.deepcopy(model.state_dict()), best_val_r10, best_epoch


def train_qa(twig_state, dataset, edges, device, embedder):
    """Fine-tune QA model from TWIG checkpoint. Delegates to ablation's train_qa_v2 logic."""
    key_fields = get_key_fields(dataset)
    train_query_path = str(PROJECT_DIR / f"data/table/train/{dataset}/query.jsonl")
    val_graph_path   = str(PROJECT_DIR / f"data/processed/dev/{dataset}/graph.pt")
    val_query_path   = str(PROJECT_DIR / f"data/table/dev/{dataset}/query.jsonl")

    data_cpu = torch.load(PROJECT_DIR / f'data/processed/train/{dataset}/graph.pt',
                          map_location='cpu', weights_only=False)
    data_cpu = filter_edges(data_cpu, edges)
    id_to_idx = build_id_to_idx(data_cpu, key_fields)
    embed_dim = data_cpu['table'].x.size(1)

    metadata = get_canonical_metadata(QA_QUERY_EDGE_MODE)
    model = QueryAwareModel(embed_dim=embed_dim, hidden_channels=768, metadata=metadata,
                            dropout=0.10, sage_aggr='min', hetero_aggr='max').to(device)
    load_pretrained_into_qa_model(model, twig_state)

    query_keywords = ['queries', 'rev_queries', 'queries_page', 'rev_queries_page',
                      'queries_column', 'rev_queries_column']
    base_params, query_params = [], []
    for name, param in model.named_parameters():
        (query_params if any(kw in name for kw in query_keywords) else base_params).append(param)

    optimizer = optim.AdamW([{'params': base_params, 'lr': QA_BASE_LR},
                              {'params': query_params, 'lr': QA_QUERY_LR}],
                             weight_decay=QA_WEIGHT_DECAY)
    warmup = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e+1)/float(QA_WARMUP_EPOCHS)))
    cosine = CosineAnnealingLR(optimizer, T_max=QA_NUM_EPOCHS - QA_WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[QA_WARMUP_EPOCHS])
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    texts, pos_lists = load_queries(train_query_path, id_to_idx, key_fields)
    if not texts:
        print(f'  [QA] No valid training queries for {dataset}'); return None

    print(f'  [QA] Embedding {len(texts)} train queries...')
    query_vecs = torch.tensor(embedder.encode(texts, show_progress_bar=False),
                              dtype=torch.float, device=device)

    val_data_cpu, val_query_vecs, val_pos_lists = None, None, None
    if Path(val_graph_path).exists():
        val_data_cpu = torch.load(val_graph_path, map_location='cpu', weights_only=False)
        val_data_cpu = filter_edges(val_data_cpu, edges)
        val_id_to_idx = build_id_to_idx(val_data_cpu, key_fields)
        val_texts, val_pos_lists = load_queries(val_query_path, val_id_to_idx, key_fields)
        if val_texts:
            val_query_vecs = torch.tensor(embedder.encode(val_texts, show_progress_bar=False),
                                          dtype=torch.float, device=device)

    data = data_cpu.clone().to(device)
    hard_neg_indices = None
    best_val_recall, best_model_state, best_epoch, patience = -1.0, None, 0, 0

    for epoch in range(1, QA_NUM_EPOCHS + 1):
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
                    _, topk = torch.topk(sim, k=min(QA_NUM_HARD_NEGATIVES, table_emb.size(0)-1), dim=1)
                    all_negs.extend(topk.tolist())
                hard_neg_indices = all_negs

        model.train()
        total_loss = 0.0; n_batches = 0
        indices = list(range(len(query_vecs))); random.shuffle(indices)

        with torch.no_grad():
            base_table_emb = model.forward(data.x_dict, data.edge_index_dict)

        for start in range(0, len(indices), QA_BATCH_SIZE):
            end = min(start + QA_BATCH_SIZE, len(indices))
            batch_idx = indices[start:end]
            optimizer.zero_grad()
            batch_loss = 0.0; batch_count = 0

            with autocast(enabled=(device.type == 'cuda')):
                for sample_idx in batch_idx:
                    q_vec = query_vecs[sample_idx:sample_idx+1]
                    pos_list = pos_lists[sample_idx]
                    hard_negs_s = hard_neg_indices[sample_idx] if hard_neg_indices else []

                    with torch.no_grad():
                        scores = torch.matmul(F.normalize(q_vec, p=2, dim=1), base_table_emb.T).squeeze(0)
                        _, top_k = torch.topk(scores, k=min(QA_SUBGRAPH_K, scores.size(0)))
                        candidate_set = set(top_k.cpu().tolist())
                    for idx in pos_list: candidate_set.add(idx)
                    for idx in hard_negs_s:
                        if idx >= 0: candidate_set.add(idx)

                    subgraph, table_mapping = build_subgraph(
                        data, q_vec, sorted(candidate_set),
                        edge_mode=QA_QUERY_EDGE_MODE, device=device)
                    sub_emb = model.forward(subgraph.x_dict, subgraph.edge_index_dict)
                    pos_new_idx = table_mapping.get(pos_list[0], -1)
                    if pos_new_idx == -1: continue

                    logits = torch.matmul(F.normalize(q_vec, p=2, dim=1), sub_emb.T).squeeze(0) / QA_TEMP
                    label = torch.tensor([pos_new_idx], dtype=torch.long, device=device)
                    batch_loss += F.cross_entropy(logits.unsqueeze(0), label, label_smoothing=QA_LABEL_SMOOTH)
                    batch_count += 1

                if batch_count == 0: continue
                loss = batch_loss / batch_count

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=QA_CLIP_GRAD_NORM)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item(); n_batches += 1

            if n_batches % 20 == 0:
                with torch.no_grad():
                    base_table_emb = model.forward(data.x_dict, data.edge_index_dict)

        scheduler.step()

        if val_data_cpu is not None and val_query_vecs is not None:
            val_data = val_data_cpu.clone().to(device)
            model.eval()
            with torch.no_grad():
                val_base_emb = model.forward(val_data.x_dict, val_data.edge_index_dict)
                val_coarse = torch.matmul(F.normalize(val_query_vecs, p=2, dim=1), val_base_emb.T)
                hits10 = 0
                for qi in range(len(val_query_vecs)):
                    pos_set = set(val_pos_lists[qi])
                    q_vec = val_query_vecs[qi:qi+1]
                    _, top_k = torch.topk(val_coarse[qi], k=min(QA_COARSE_K_EVAL, val_coarse.size(1)))
                    candidates = top_k.cpu().tolist()
                    for pi in pos_set:
                        if pi not in set(candidates): candidates.append(pi)
                    subgraph, table_mapping = build_subgraph(
                        val_data, q_vec, candidates, edge_mode=QA_QUERY_EDGE_MODE, device=device)
                    sub_emb = model.forward(subgraph.x_dict, subgraph.edge_index_dict)
                    rerank = torch.matmul(F.normalize(q_vec, p=2, dim=1), sub_emb.T).squeeze(0)
                    _, reranked = torch.topk(rerank, k=min(10, rerank.size(0)))
                    new_to_old = {v: k for k, v in table_mapping.items()}
                    if any(new_to_old[i.item()] in pos_set for i in reranked):
                        hits10 += 1
            del val_data; torch.cuda.empty_cache()
            val_r10 = hits10 / max(1, len(val_query_vecs))

            if val_r10 > best_val_recall:
                best_val_recall = val_r10; best_epoch = epoch; patience = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience += 1

            if epoch % 3 == 0 or epoch == 1:
                print(f'  [QA] Epoch {epoch:2d}/{QA_NUM_EPOCHS} | Loss {total_loss/max(1,n_batches):.4f} | '
                      f'Val R@10 {val_r10:.4f} | Best {best_val_recall:.4f} (ep {best_epoch})')

            if patience >= QA_EARLY_STOPPING_PATIENCE:
                print(f'  [QA] Early stopping at epoch {epoch}'); break
        else:
            if epoch % 3 == 0 or epoch == 1:
                print(f'  [QA] Epoch {epoch:2d}/{QA_NUM_EPOCHS} | Loss {total_loss/max(1,n_batches):.4f}')
            best_model_state = copy.deepcopy(model.state_dict()); best_epoch = epoch

    if best_model_state:
        model.load_state_dict(best_model_state)
    result = copy.deepcopy(model.state_dict())
    del model, data, query_vecs
    torch.cuda.empty_cache()
    return result, best_val_recall, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['feta', 'ottqa', 'mimo_en', 'mimo_ch', 'e2ewtq', 'mmqa'])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    embedder = SentenceTransformer(MODEL_NAME, device=str(device))

    for dataset in args.datasets:
        edges = QA_BEST_EDGE_CONFIGS[dataset]
        ckpt_dir = PROJECT_DIR / f'checkpoints/{dataset}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        twig_path = ckpt_dir / 'model_qa_best_edges.pt'
        qa_path   = ckpt_dir / 'model_qa_best_edges_qa.pt'

        print(f'\n{"="*60}')
        print(f'Dataset: {dataset.upper()}  edges: {edges}')
        print(f'{"="*60}')

        # Phase 1: TWIG
        print(f'\n  Phase 1: Training TWIG...')
        twig_state, twig_r10, twig_ep = train_twig(dataset, edges, device, embedder)
        torch.save({
            'model_state_dict': twig_state,
            'hps': TWIG_HPS,
            'best_edges': edges,
            'best_val_r10': twig_r10,
            'best_epoch': twig_ep,
        }, twig_path)
        print(f'  TWIG saved → {twig_path}  (val R@10={twig_r10:.4f}, epoch={twig_ep})')

        # Phase 2: QA fine-tune
        print(f'\n  Phase 2: Fine-tuning QA...')
        result = train_qa(twig_state, dataset, edges, device, embedder)
        if result is not None:
            qa_state, qa_r10, qa_ep = result
            torch.save({
                'model_state_dict': qa_state,
                'hps': TWIG_HPS,
                'best_edges': edges,
                'edge_mode': QA_QUERY_EDGE_MODE,
                'best_val_r10': qa_r10,
                'best_epoch': qa_ep,
            }, qa_path)
            print(f'  QA saved  → {qa_path}  (val R@10={qa_r10:.4f}, epoch={qa_ep})')
        else:
            print(f'  QA training failed for {dataset}')

        del twig_state; torch.cuda.empty_cache()

    del embedder
    print('\nAll done.')


if __name__ == '__main__':
    main()
