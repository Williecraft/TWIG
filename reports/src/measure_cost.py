#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Computational Cost Measurement

Online:  bge-m3 baseline vs TWIG vs TWIG-QA，固定硬體 + query 數
Offline: 量 graph construction + TWIG training + QA fine-tuning 時間

Usage:
  cd reports/src && python measure_cost.py --dataset ottqa --gpu 0
  cd reports/src && python measure_cost.py --dataset ottqa --gpu 0 --online-only
"""

import argparse
import os
import sys
import time
import json
import platform
import subprocess
from pathlib import Path

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=int, default=0)
_temp_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_temp_args.gpu)

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'query_aware'))

RESULT_DIR = PROJECT_DIR / "results" / "cost_measurement"
MODEL_NAME = 'BAAI/bge-m3'
COARSE_K = 100   # QA rerank candidate pool size


# ═══════════════════════════════════════════════════════
# Hardware info
# ═══════════════════════════════════════════════════════

def get_hardware_info():
    info = {}
    # GPU
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info['gpu_vram_gb'] = round(props.total_memory / 1024**3, 1)
        info['gpu_count'] = torch.cuda.device_count()
    else:
        info['gpu_name'] = 'CPU only'

    # CPU
    try:
        cpu_out = subprocess.check_output(['lscpu'], text=True)
        for line in cpu_out.splitlines():
            if 'Model name' in line:
                info['cpu'] = line.split(':')[1].strip()
                break
    except Exception:
        info['cpu'] = platform.processor()

    # RAM
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    kb = int(line.split()[1])
                    info['ram_gb'] = round(kb / 1024**2, 1)
                    break
    except Exception:
        pass

    return info


def print_hardware(info):
    print(f"\n{'='*60}")
    print("Hardware")
    print(f"{'='*60}")
    print(f"  GPU : {info.get('gpu_name', 'N/A')}  ({info.get('gpu_vram_gb', '?')} GB VRAM)")
    print(f"  CPU : {info.get('cpu', 'N/A')}")
    print(f"  RAM : {info.get('ram_gb', '?')} GB")


# ═══════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════

def get_key_fields(dataset):
    if dataset in ["ottqa", "feta", "e2ewtq"]:
        return ("sheet_name", "file_name")
    return ("id",)


def make_key(item, key_fields):
    return "|".join(str(item.get(f, "")) for f in key_fields)


def load_question_texts(query_file):
    texts = []
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get('question') or (obj.get('questions') or [''])[0]
            if q and q.strip():
                texts.append(q.strip())
    return texts


def fmt(s):
    return f"{s:.2f}s" if s < 60 else f"{s/60:.2f}min"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════
# Online measurement
# ═══════════════════════════════════════════════════════

def measure_online(dataset, device, split='train'):
    graph_path = PROJECT_DIR / f"data/processed/{split}/{dataset}/graph.pt"
    query_file = PROJECT_DIR / f"data/table/{split}/{dataset}/query.jsonl"
    twig_ckpt_path = PROJECT_DIR / f"checkpoints/{dataset}/model.pt"
    qa_ckpt_path = PROJECT_DIR / f"checkpoints/{dataset}/model_query_aware_v2.pt"

    print(f"\n{'='*60}")
    print(f"Online Cost — {dataset} ({split} split)")
    print(f"{'='*60}")

    data = torch.load(graph_path, map_location=device, weights_only=False)
    queries = load_question_texts(query_file)
    n_queries = len(queries)
    n_tables  = data['table'].x.size(0)
    embed_dim = data['table'].x.size(1)
    print(f"  Queries: {n_queries} | Tables: {n_tables} | Embed dim: {embed_dim}")

    # ── Load embedder (shared by all methods) ───────────────
    print("\n  Loading bge-m3 embedder...")
    embedder = SentenceTransformer(MODEL_NAME, device=str(device))
    _ = embedder.encode(queries[:4], show_progress_bar=False)   # warm-up
    sync()

    results = {
        'dataset': dataset, 'split': split,
        'n_queries': n_queries, 'n_tables': n_tables,
    }

    # ════════════════════════════════════════════════════════
    # [1] bge-m3 baseline
    # ════════════════════════════════════════════════════════
    print("\n  [bge-m3] embedding + cosine similarity")
    sync(); t0 = time.perf_counter()

    q_vecs = torch.tensor(
        embedder.encode(queries, show_progress_bar=False),
        dtype=torch.float, device=device)
    q_vecs = F.normalize(q_vecs, p=2, dim=1)
    tbl_raw = F.normalize(data['table'].x.to(device), p=2, dim=1)
    scores  = torch.matmul(q_vecs, tbl_raw.T)
    _       = torch.topk(scores, k=10, dim=1)

    sync(); t_bgem3 = time.perf_counter() - t0
    print(f"    total={fmt(t_bgem3)} | per-query={t_bgem3/n_queries*1000:.2f}ms")
    results['bge_m3'] = {'total_s': round(t_bgem3,3), 'per_query_ms': round(t_bgem3/n_queries*1000,3)}
    del q_vecs, tbl_raw, scores; torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════
    # [2] TWIG (bge-m3 + GNN forward)
    # ════════════════════════════════════════════════════════
    if not twig_ckpt_path.exists():
        print(f"\n  [TWIG] skipping — checkpoint not found")
    else:
        print("\n  [TWIG] bge-m3 + GNN forward + cosine similarity")
        from train_model import DiffusionModel

        ckpt = torch.load(twig_ckpt_path, map_location=device, weights_only=False)
        hps  = ckpt.get('hps', {})
        twig = DiffusionModel(
            embed_dim=embed_dim,
            hidden_channels=hps.get('HIDDEN_CHANNELS', 768),
            metadata=data.metadata(),
            dropout=hps.get('DROPOUT', 0.1),
            sage_aggr=hps.get('SAGE_AGGR', 'min'),
            hetero_aggr=hps.get('HETERO_AGGR', 'max'),
        ).to(device)
        twig.load_state_dict(ckpt['model_state_dict'], strict=False)
        twig.eval()
        data_gpu = data.to(device)

        with torch.no_grad(): _ = twig.forward(data_gpu.x_dict, data_gpu.edge_index_dict)  # warm-up
        sync(); t0 = time.perf_counter()

        q_vecs = torch.tensor(
            embedder.encode(queries, show_progress_bar=False),
            dtype=torch.float, device=device)
        q_vecs = F.normalize(q_vecs, p=2, dim=1)
        with torch.no_grad():
            tbl_emb = twig.forward(data_gpu.x_dict, data_gpu.edge_index_dict)
        scores = torch.matmul(q_vecs, tbl_emb.T)
        _ = torch.topk(scores, k=10, dim=1)

        sync(); t_twig = time.perf_counter() - t0
        print(f"    total={fmt(t_twig)} | per-query={t_twig/n_queries*1000:.2f}ms "
              f"(+{(t_twig-t_bgem3)/t_bgem3*100:.1f}% vs bge-m3)")
        results['twig'] = {'total_s': round(t_twig,3), 'per_query_ms': round(t_twig/n_queries*1000,3)}
        del q_vecs, scores; torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════
    # [3] TWIG-QA (bge-m3 + TWIG coarse + subgraph + QA rerank)
    # ════════════════════════════════════════════════════════
    if not qa_ckpt_path.exists():
        print(f"\n  [TWIG-QA] skipping — checkpoint not found")
    else:
        print(f"\n  [TWIG-QA] bge-m3 + TWIG coarse(k={COARSE_K}) + subgraph + QA rerank")
        from train_query_aware_v2 import (
            QueryAwareModel, get_canonical_metadata,
            build_subgraph, filter_edges, BEST_EDGE_CONFIGS,
        )

        qa_ckpt  = torch.load(qa_ckpt_path, map_location=device, weights_only=False)
        qa_hps   = qa_ckpt.get('hps', hps)
        edge_mode = qa_ckpt.get('edge_mode', 'E4')
        metadata  = get_canonical_metadata(edge_mode)
        qa_model  = QueryAwareModel(
            embed_dim=embed_dim,
            hidden_channels=qa_hps.get('HIDDEN_CHANNELS', 768),
            metadata=metadata,
            dropout=qa_hps.get('DROPOUT', 0.1),
            sage_aggr=qa_hps.get('SAGE_AGGR', 'min'),
            hetero_aggr=qa_hps.get('HETERO_AGGR', 'max'),
        ).to(device)
        qa_model.load_state_dict(qa_ckpt['model_state_dict'], strict=False)
        qa_model.eval()

        best_edges = BEST_EDGE_CONFIGS.get(dataset, [])
        data_filt  = filter_edges(data_gpu, best_edges).to(device) if best_edges else data_gpu

        # warm-up one query
        with torch.no_grad():
            sub, tm = build_subgraph(data_filt, tbl_emb[0:1], list(range(min(COARSE_K, n_tables))),
                                     edge_mode=edge_mode, device=device)
            _ = qa_model.forward(sub.x_dict, sub.edge_index_dict)
        sync(); t0 = time.perf_counter()

        q_vecs = torch.tensor(
            embedder.encode(queries, show_progress_bar=False),
            dtype=torch.float, device=device)
        q_vecs = F.normalize(q_vecs, p=2, dim=1)

        # coarse scores already computed from twig forward above (reuse tbl_emb)
        coarse = torch.matmul(q_vecs, tbl_emb.T)

        with torch.no_grad():
            for qi in range(n_queries):
                q_vec = q_vecs[qi:qi+1]
                _, top_idx = torch.topk(coarse[qi], k=min(COARSE_K, n_tables))
                candidates = top_idx.cpu().tolist()
                sub, table_map = build_subgraph(data_filt, q_vec, candidates,
                                                edge_mode=edge_mode, device=device)
                sub_emb = qa_model.forward(sub.x_dict, sub.edge_index_dict)
                rerank  = torch.matmul(q_vec, sub_emb.T).squeeze(0)
                _ = torch.topk(rerank, k=min(10, rerank.size(0)))

        sync(); t_qa = time.perf_counter() - t0
        print(f"    total={fmt(t_qa)} | per-query={t_qa/n_queries*1000:.2f}ms "
              f"(+{(t_qa-t_bgem3)/t_bgem3*100:.1f}% vs bge-m3)")
        results['twig_qa'] = {'total_s': round(t_qa,3), 'per_query_ms': round(t_qa/n_queries*1000,3)}
        del q_vecs, coarse, tbl_emb, twig, qa_model, data_gpu, data_filt
        torch.cuda.empty_cache()

    del embedder
    return results


# ═══════════════════════════════════════════════════════
# Offline measurement
# ═══════════════════════════════════════════════════════

def measure_offline(dataset, device):
    print(f"\n{'='*60}")
    print(f"Offline Cost — {dataset}")
    print(f"{'='*60}")

    results = {'dataset': dataset}
    key_fields = get_key_fields(dataset)

    # ── Graph file size (proxy for construction cost) ─────
    for split in ['train', 'dev', 'test']:
        gp = PROJECT_DIR / f"data/processed/{split}/{dataset}/graph.pt"
        if gp.exists():
            results[f'graph_{split}_mb'] = round(gp.stat().st_size / 1024**2, 1)

    # ── Time one TWIG epoch, extrapolate ─────────────────
    import random, copy
    import torch.optim as optim
    from torch.cuda.amp import GradScaler, autocast

    train_graph = PROJECT_DIR / f"data/processed/train/{dataset}/graph.pt"
    train_query = PROJECT_DIR / f"data/table/train/{dataset}/query.jsonl"

    if not train_graph.exists():
        print(f"  Skipping: {train_graph} not found")
        return results

    from train_model import DiffusionModel
    embedder = SentenceTransformer(MODEL_NAME, device=str(device))
    data = torch.load(train_graph, map_location=device, weights_only=False)
    embed_dim = data['table'].x.size(1)

    id_to_idx = {}
    if hasattr(data, 'metadata_maps') and 'table_meta' in data.metadata_maps:
        for i, m in enumerate(data.metadata_maps['table_meta']):
            id_to_idx[make_key(m, key_fields)] = i

    texts, pos_lists = [], []
    with open(train_query) as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get('question') or (obj.get('questions') or [''])[0]
            if not q: continue
            pos = [id_to_idx[make_key(g, key_fields)]
                   for g in obj.get('ground_truth_list', []) or []
                   if make_key(g, key_fields) in id_to_idx]
            if pos:
                texts.append(q.strip())
                pos_lists.append(pos)

    n_train = len(texts)
    print(f"\n  TWIG training: {n_train} queries, {data['table'].x.size(0)} tables")

    q_vecs = torch.tensor(
        embedder.encode(texts, show_progress_bar=True, batch_size=256),
        dtype=torch.float, device=device)

    HPS = dict(HIDDEN_CHANNELS=768, DROPOUT=0.10, SAGE_AGGR='min', HETERO_AGGR='max',
               LEARNING_RATE=5.54e-4, WEIGHT_DECAY=0.032, BATCH_SIZE=128,
               TEMP_START=0.05, NUM_EPOCHS=30)

    model = DiffusionModel(embed_dim, HPS['HIDDEN_CHANNELS'], data.metadata(),
                           HPS['DROPOUT'], HPS['SAGE_AGGR'], HPS['HETERO_AGGR']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=HPS['LEARNING_RATE'],
                            weight_decay=HPS['WEIGHT_DECAY'])
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # time one full epoch
    model.train()
    idx = list(range(n_train)); random.shuffle(idx)
    sync(); t0 = time.perf_counter()

    for start in range(0, n_train, HPS['BATCH_SIZE']):
        b = idx[start:start+HPS['BATCH_SIZE']]
        qb = q_vecs[b]
        labels = torch.tensor([pos_lists[i][0] for i in b], dtype=torch.long, device=device)
        optimizer.zero_grad()
        with autocast(enabled=(device.type == 'cuda')):
            tbl = model.forward(data.x_dict, data.edge_index_dict)
            logits = torch.matmul(F.normalize(qb, p=2, dim=1), tbl.T) / HPS['TEMP_START']
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

    sync(); t_epoch = time.perf_counter() - t0
    t_twig_est = t_epoch * HPS['NUM_EPOCHS']

    print(f"  1 epoch: {fmt(t_epoch)} → estimated {HPS['NUM_EPOCHS']} epochs: {fmt(t_twig_est)}")
    results['twig_one_epoch_s'] = round(t_epoch, 1)
    results['twig_estimated_total_s'] = round(t_twig_est, 1)
    results['twig_epochs'] = HPS['NUM_EPOCHS']

    # QA fine-tuning: 15 epochs, assume ~0.5× per epoch of TWIG (subgraph, smaller)
    qa_epochs = 15
    t_qa_est = t_epoch * 0.5 * qa_epochs
    results['qa_finetune_estimated_s'] = round(t_qa_est, 1)
    results['qa_finetune_epochs'] = qa_epochs
    print(f"  QA fine-tune estimate ({qa_epochs} epochs, ~0.5× TWIG/epoch): {fmt(t_qa_est)}")

    results['qgpt_note'] = (
        "QGPT requires O(|tables|) LLM pseudo-query generation calls. "
        f"~{data['table'].x.size(0)} tables × 1-3s/call ≈ "
        f"{data['table'].x.size(0)*2//3600:.0f}-{data['table'].x.size(0)*3//3600:.0f} GPU-hours for generation alone."
    )

    del model, data, q_vecs, embedder; torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ottqa')
    parser.add_argument('--split', default='train',
                        help='Split for online measurement (train has more tables)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--online-only', action='store_true')
    parser.add_argument('--offline-only', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hw = get_hardware_info()
    print_hardware(hw)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {'hardware': hw}

    if not args.offline_only:
        all_results['online'] = measure_online(args.dataset, device, split=args.split)

    if not args.online_only:
        all_results['offline'] = measure_offline(args.dataset, device)

    # ── Summary table ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  GPU : {hw.get('gpu_name')}  ({hw.get('gpu_vram_gb')}GB)")
    print(f"  CPU : {hw.get('cpu','N/A')}")
    print(f"  RAM : {hw.get('ram_gb','?')}GB\n")

    if 'online' in all_results:
        o = all_results['online']
        print(f"  [Online — {o['n_queries']} queries / {o['n_tables']} tables]")
        print(f"  {'Method':<18} {'Total':>10}  {'Per-query':>12}")
        print(f"  {'-'*44}")
        for key, label in [('bge_m3','bge-m3'), ('twig','TWIG'), ('twig_qa','TWIG-QA')]:
            if key in o:
                v = o[key]
                print(f"  {label:<18} {fmt(v['total_s']):>10}  {v['per_query_ms']:>9.2f} ms")

    if 'offline' in all_results:
        o = all_results['offline']
        print(f"\n  [Offline]")
        if 'twig_one_epoch_s' in o:
            print(f"  TWIG training  : {fmt(o['twig_estimated_total_s'])}  "
                  f"({o['twig_epochs']} epochs, {fmt(o['twig_one_epoch_s'])}/epoch)")
        if 'qa_finetune_estimated_s' in o:
            print(f"  QA fine-tuning : {fmt(o['qa_finetune_estimated_s'])}  "
                  f"({o['qa_finetune_epochs']} epochs)")
        if 'qgpt_note' in o:
            print(f"  QGPT note      : {o['qgpt_note']}")

    out = RESULT_DIR / f"{args.dataset}_{args.split}.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved → {out}")


if __name__ == '__main__':
    main()
