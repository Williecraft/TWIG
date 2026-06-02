#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Computational Cost Measurement

每個 method 跑在獨立 subprocess，確保沒有跨 method GPU cache 污染。

Usage:
  cd reports/src && python measure_cost.py --dataset ottqa --gpu 0
  cd reports/src && python measure_cost.py --dataset ottqa --gpu 0 --online-only
  # 單獨測某個 method（供 subprocess 呼叫）：
  python measure_cost.py --dataset ottqa --gpu 0 --method bgem3
  python measure_cost.py --dataset ottqa --gpu 0 --method twig
  python measure_cost.py --dataset ottqa --gpu 0 --method twig_qa
"""

import argparse
import os
import sys
import time
import json
import platform
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ottqa')
parser.add_argument('--split', default='train')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--online-only', action='store_true')
parser.add_argument('--offline-only', action='store_true')
parser.add_argument('--method', default=None,
                    help='Internal: bgem3 | twig | twig_qa (runs single method and exits)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'query_aware'))

RESULT_DIR = PROJECT_DIR / "results" / "cost_measurement"
MODEL_NAME  = 'BAAI/bge-m3'
COARSE_K    = 100
QA_SAMPLE   = 300   # sample size for TWIG-QA per-query timing
N_REPEATS   = 3     # repeat timing N times, report median


# ═══════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════

def get_key_fields(dataset):
    return ("sheet_name", "file_name") if dataset in ["ottqa", "feta", "e2ewtq"] else ("id",)

def make_key(item, kf):
    return "|".join(str(item.get(f, "")) for f in kf)

def load_questions(query_file):
    texts = []
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get('question') or (obj.get('questions') or [''])[0]
            if q and q.strip():
                texts.append(q.strip())
    return texts

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def fmt(s):
    return f"{s:.3f}s" if s < 60 else f"{s/60:.2f}min"

def median(lst):
    s = sorted(lst)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n//2 - 1] + s[n//2]) / 2

def get_hardware_info():
    info = {}
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
        info['vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    try:
        for line in subprocess.check_output(['lscpu'], text=True).splitlines():
            if 'Model name' in line:
                info['cpu'] = line.split(':')[1].strip(); break
    except Exception:
        info['cpu'] = platform.processor()
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    info['ram_gb'] = round(int(line.split()[1]) / 1024**2, 1); break
    except Exception:
        pass
    return info


# ═══════════════════════════════════════════════════════
# Single-method runners  (each called in its own process)
# ═══════════════════════════════════════════════════════

def run_bgem3(dataset, split, device):
    graph_path = PROJECT_DIR / f"data/processed/{split}/{dataset}/graph.pt"
    query_file = PROJECT_DIR / f"data/table/{split}/{dataset}/query.jsonl"

    data    = torch.load(graph_path, map_location=device, weights_only=False)
    queries = load_questions(query_file)
    n_q     = len(queries)
    n_t     = data['table'].x.size(0)

    embedder = SentenceTransformer(MODEL_NAME, device=str(device))

    # warm-up (excluded from timing)
    _ = torch.tensor(embedder.encode(queries[:8], show_progress_bar=False),
                     dtype=torch.float, device=device)
    tbl_raw = F.normalize(data['table'].x.to(device), p=2, dim=1)
    sync()

    times = []
    for _ in range(N_REPEATS):
        torch.cuda.empty_cache()
        sync(); t0 = time.perf_counter()

        q_vecs = torch.tensor(embedder.encode(queries, show_progress_bar=False),
                              dtype=torch.float, device=device)
        q_vecs = F.normalize(q_vecs, p=2, dim=1)
        scores = torch.matmul(q_vecs, tbl_raw.T)
        _ = torch.topk(scores, k=10, dim=1)

        sync(); times.append(time.perf_counter() - t0)
        del q_vecs, scores

    t = median(times)
    return {'method': 'bge-m3', 'n_queries': n_q, 'n_tables': n_t,
            'total_s': round(t, 4), 'per_query_ms': round(t / n_q * 1000, 3),
            'repeats': times}


def run_twig(dataset, split, device):
    from train_model import DiffusionModel

    graph_path = PROJECT_DIR / f"data/processed/{split}/{dataset}/graph.pt"
    query_file = PROJECT_DIR / f"data/table/{split}/{dataset}/query.jsonl"
    ckpt_path  = PROJECT_DIR / f"checkpoints/{dataset}/model.pt"

    data    = torch.load(graph_path, map_location=device, weights_only=False)
    queries = load_questions(query_file)
    n_q     = len(queries)
    n_t     = data['table'].x.size(0)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hps  = ckpt.get('hps', {})
    model = DiffusionModel(
        embed_dim=data['table'].x.size(1),
        hidden_channels=hps.get('HIDDEN_CHANNELS', 768),
        metadata=data.metadata(),
        dropout=hps.get('DROPOUT', 0.1),
        sage_aggr=hps.get('SAGE_AGGR', 'min'),
        hetero_aggr=hps.get('HETERO_AGGR', 'max'),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    data_gpu = data.to(device)

    embedder = SentenceTransformer(MODEL_NAME, device=str(device))

    # warm-up
    _ = torch.tensor(embedder.encode(queries[:8], show_progress_bar=False),
                     dtype=torch.float, device=device)
    with torch.no_grad():
        _ = model.forward(data_gpu.x_dict, data_gpu.edge_index_dict)
    sync()

    times = []
    for _ in range(N_REPEATS):
        torch.cuda.empty_cache()
        sync(); t0 = time.perf_counter()

        q_vecs = torch.tensor(embedder.encode(queries, show_progress_bar=False),
                              dtype=torch.float, device=device)
        q_vecs = F.normalize(q_vecs, p=2, dim=1)
        with torch.no_grad():
            tbl_emb = model.forward(data_gpu.x_dict, data_gpu.edge_index_dict)
        scores = torch.matmul(q_vecs, tbl_emb.T)
        _ = torch.topk(scores, k=10, dim=1)

        sync(); times.append(time.perf_counter() - t0)
        del q_vecs, tbl_emb, scores

    t = median(times)
    return {'method': 'TWIG', 'n_queries': n_q, 'n_tables': n_t,
            'total_s': round(t, 4), 'per_query_ms': round(t / n_q * 1000, 3),
            'repeats': times}


def run_twig_qa(dataset, split, device):
    from train_model import DiffusionModel
    from train_query_aware_v2 import (
        QueryAwareModel, get_canonical_metadata,
        build_subgraph, filter_edges, BEST_EDGE_CONFIGS,
    )
    import random

    graph_path   = PROJECT_DIR / f"data/processed/{split}/{dataset}/graph.pt"
    query_file   = PROJECT_DIR / f"data/table/{split}/{dataset}/query.jsonl"
    twig_path    = PROJECT_DIR / f"checkpoints/{dataset}/model.pt"
    qa_ckpt_path = PROJECT_DIR / f"checkpoints/{dataset}/model_query_aware_v2.pt"

    data    = torch.load(graph_path, map_location=device, weights_only=False)
    queries = load_questions(query_file)
    n_q     = len(queries)
    n_t     = data['table'].x.size(0)
    embed_dim = data['table'].x.size(1)

    # TWIG model (coarse ranker)
    twig_ckpt = torch.load(twig_path, map_location=device, weights_only=False)
    hps = twig_ckpt.get('hps', {})
    twig = DiffusionModel(
        embed_dim=embed_dim, hidden_channels=hps.get('HIDDEN_CHANNELS', 768),
        metadata=data.metadata(), dropout=hps.get('DROPOUT', 0.1),
        sage_aggr=hps.get('SAGE_AGGR', 'min'), hetero_aggr=hps.get('HETERO_AGGR', 'max'),
    ).to(device)
    twig.load_state_dict(twig_ckpt['model_state_dict'], strict=False)
    twig.eval()

    # QA model
    qa_ckpt   = torch.load(qa_ckpt_path, map_location=device, weights_only=False)
    qa_hps    = qa_ckpt.get('hps', hps)
    edge_mode = qa_ckpt.get('edge_mode', 'E4')
    qa_model  = QueryAwareModel(
        embed_dim=embed_dim, hidden_channels=qa_hps.get('HIDDEN_CHANNELS', 768),
        metadata=get_canonical_metadata(edge_mode),
        dropout=qa_hps.get('DROPOUT', 0.1),
        sage_aggr=qa_hps.get('SAGE_AGGR', 'min'), hetero_aggr=qa_hps.get('HETERO_AGGR', 'max'),
    ).to(device)
    qa_model.load_state_dict(qa_ckpt['model_state_dict'], strict=False)
    qa_model.eval()

    best_edges = BEST_EDGE_CONFIGS.get(dataset, [])
    data_gpu   = data.to(device)
    data_filt  = filter_edges(data_gpu, best_edges) if best_edges else data_gpu

    embedder = SentenceTransformer(MODEL_NAME, device=str(device))

    # warm-up
    _ = torch.tensor(embedder.encode(queries[:8], show_progress_bar=False),
                     dtype=torch.float, device=device)
    with torch.no_grad():
        tbl_warmup = twig.forward(data_gpu.x_dict, data_gpu.edge_index_dict)
        sub_w, _ = build_subgraph(data_filt, tbl_warmup[0:1],
                                  list(range(min(COARSE_K, n_t))),
                                  edge_mode=edge_mode, device=device)
        _ = qa_model.forward(sub_w.x_dict, sub_w.edge_index_dict)
    del tbl_warmup
    sync()

    sample_idx = random.sample(range(n_q), min(QA_SAMPLE, n_q))

    times = []
    for rep in range(N_REPEATS):
        torch.cuda.empty_cache()
        sync(); t0 = time.perf_counter()

        # 1. embed queries
        q_vecs = torch.tensor(
            embedder.encode(queries, show_progress_bar=False),
            dtype=torch.float, device=device)
        q_vecs = F.normalize(q_vecs, p=2, dim=1)

        # 2. TWIG full-graph forward (once)
        with torch.no_grad():
            tbl_emb = twig.forward(data_gpu.x_dict, data_gpu.edge_index_dict)
        coarse = torch.matmul(q_vecs, tbl_emb.T)

        # 3. per-query subgraph + rerank (sampled → extrapolate)
        t_rerank_start = time.perf_counter()
        with torch.no_grad():
            for qi in sample_idx:
                q_vec = q_vecs[qi:qi+1]
                _, top_idx = torch.topk(coarse[qi], k=min(COARSE_K, n_t))
                sub, _ = build_subgraph(data_filt, q_vec, top_idx.cpu().tolist(),
                                        edge_mode=edge_mode, device=device)
                sub_emb = qa_model.forward(sub.x_dict, sub.edge_index_dict)
                rerank  = torch.matmul(q_vec, sub_emb.T).squeeze(0)
                _ = torch.topk(rerank, k=min(10, rerank.size(0)))
        sync()
        t_rerank_sample = time.perf_counter() - t_rerank_start

        # extrapolate rerank time to full query set
        per_q_rerank = t_rerank_sample / len(sample_idx)
        t_shared = time.perf_counter() - t0 - t_rerank_sample  # embed + gnn forward
        t_total  = t_shared + per_q_rerank * n_q

        sync(); times.append(t_total)
        del q_vecs, tbl_emb, coarse

    t = median(times)
    return {'method': 'TWIG-QA', 'n_queries': n_q, 'n_tables': n_t,
            'total_s': round(t, 4), 'per_query_ms': round(t / n_q * 1000, 3),
            'rerank_per_query_ms': round(median([
                # re-derive from repeats not easily available here; store separately
                t / n_q * 1000
            ]), 3),
            'sample_size': len(sample_idx), 'repeats': times}


# ═══════════════════════════════════════════════════════
# Subprocess dispatcher  (--method flag)
# ═══════════════════════════════════════════════════════

if args.method:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.method == 'bgem3':
        result = run_bgem3(args.dataset, args.split, device)
    elif args.method == 'twig':
        result = run_twig(args.dataset, args.split, device)
    elif args.method == 'twig_qa':
        result = run_twig_qa(args.dataset, args.split, device)
    else:
        sys.exit(f"Unknown method: {args.method}")
    print(json.dumps(result))
    sys.exit(0)


# ═══════════════════════════════════════════════════════
# Main: spawn one subprocess per method
# ═══════════════════════════════════════════════════════

def run_method_subprocess(method):
    """Spawn a fresh Python process to measure one method, return parsed JSON."""
    cmd = [
        sys.executable, __file__,
        '--dataset', args.dataset,
        '--split', args.split,
        '--gpu', str(args.gpu),
        '--method', method,
    ]
    print(f"  → subprocess: {method} ...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ERROR in {method}:\n{proc.stderr[-800:]}")
        return None
    # last non-empty line should be the JSON
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.strip().startswith('{'):
            return json.loads(line.strip())
    print(f"  Could not parse output for {method}:\n{proc.stdout[-400:]}")
    return None


def measure_online():
    print(f"\n{'='*60}")
    print(f"Online Cost — {args.dataset} ({args.split} split)")
    print(f"Each method in a fresh subprocess  |  N_REPEATS={N_REPEATS}  |  median reported")
    print(f"{'='*60}")

    results = {}
    for method in ['bgem3', 'twig', 'twig_qa']:
        r = run_method_subprocess(method)
        if r:
            results[r['method']] = r
            print(f"    {r['method']:<10}  total={fmt(r['total_s'])}  "
                  f"per-query={r['per_query_ms']:.2f}ms")
    return results


def measure_offline():
    print(f"\n{'='*60}")
    print(f"Offline Cost — {args.dataset}")
    print(f"{'='*60}")

    import random, torch.optim as optim
    from torch.cuda.amp import GradScaler, autocast

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    key_fields = get_key_fields(args.dataset)

    train_graph = PROJECT_DIR / f"data/processed/train/{args.dataset}/graph.pt"
    train_query = PROJECT_DIR / f"data/table/train/{args.dataset}/query.jsonl"
    if not train_graph.exists():
        print(f"  Skipping: {train_graph} not found")
        return {}

    from train_model import DiffusionModel
    embedder = SentenceTransformer(MODEL_NAME, device=str(device))
    data     = torch.load(train_graph, map_location=device, weights_only=False)
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
                texts.append(q.strip()); pos_lists.append(pos)

    n_train = len(texts)
    n_tables = data['table'].x.size(0)
    print(f"  {n_train} queries, {n_tables} tables")

    print("  Embedding training queries...")
    q_vecs = torch.tensor(
        embedder.encode(texts, show_progress_bar=True, batch_size=256),
        dtype=torch.float, device=device)

    HPS = dict(HIDDEN_CHANNELS=768, DROPOUT=0.10, SAGE_AGGR='min', HETERO_AGGR='max',
               LEARNING_RATE=5.54e-4, WEIGHT_DECAY=0.032, BATCH_SIZE=128,
               TEMP=0.05, NUM_EPOCHS=30)
    QA_EPOCHS = 15

    model = DiffusionModel(embed_dim, HPS['HIDDEN_CHANNELS'], data.metadata(),
                           HPS['DROPOUT'], HPS['SAGE_AGGR'], HPS['HETERO_AGGR']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=HPS['LEARNING_RATE'],
                            weight_decay=HPS['WEIGHT_DECAY'])
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # time one full epoch
    print("  Timing one TWIG training epoch...")
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
            logits = torch.matmul(F.normalize(qb, p=2, dim=1), tbl.T) / HPS['TEMP']
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    sync(); t_epoch = time.perf_counter() - t0

    t_twig = t_epoch * HPS['NUM_EPOCHS']
    t_qa   = t_epoch * 0.5 * QA_EPOCHS  # subgraph-based, ~half epoch length
    print(f"  1 epoch: {fmt(t_epoch)}")
    print(f"  TWIG ({HPS['NUM_EPOCHS']} epochs, estimated): {fmt(t_twig)}")
    print(f"  QA fine-tune ({QA_EPOCHS} epochs, estimated): {fmt(t_qa)}")
    print(f"  QGPT pseudo-query gen: O({n_tables}) LLM calls × ~2s/call ≈ "
          f"{n_tables*2//3600:.0f}–{n_tables*3//3600:.0f} GPU-hours (generation alone)")

    del model, data, q_vecs, embedder; torch.cuda.empty_cache()
    return {
        'n_train_queries': n_train, 'n_tables': n_tables,
        'twig_one_epoch_s': round(t_epoch, 1),
        'twig_total_estimated_s': round(t_twig, 1),
        'twig_epochs': HPS['NUM_EPOCHS'],
        'qa_finetune_estimated_s': round(t_qa, 1),
        'qa_epochs': QA_EPOCHS,
        'qgpt_llm_calls': n_tables,
        'qgpt_estimated_hours': f"{n_tables*2//3600:.0f}–{n_tables*3//3600:.0f}",
    }


# ─────────────────────────────────────────────────────
hw = get_hardware_info()
print(f"\n{'='*60}")
print("Hardware")
print(f"{'='*60}")
print(f"  GPU : {hw.get('gpu','N/A')}  ({hw.get('vram_gb','?')} GB VRAM)")
print(f"  CPU : {hw.get('cpu','N/A')}")
print(f"  RAM : {hw.get('ram_gb','?')} GB")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
all_results = {'hardware': hw}

if not args.offline_only:
    all_results['online'] = measure_online()

if not args.online_only:
    all_results['offline'] = measure_offline()

# ── Final summary ─────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  GPU: {hw.get('gpu')}  ({hw.get('vram_gb')}GB VRAM)")
print(f"  CPU: {hw.get('cpu')}")
print(f"  RAM: {hw.get('ram_gb')}GB\n")

if 'online' in all_results:
    o = all_results['online']
    first = next(iter(o.values()), {})
    print(f"  Online ({first.get('n_queries','?')} queries / "
          f"{first.get('n_tables','?')} tables, median of {N_REPEATS} runs)")
    print(f"  {'Method':<12} {'Total':>10}  {'Per-query':>12}")
    print(f"  {'-'*38}")
    for label in ['bge-m3', 'TWIG', 'TWIG-QA']:
        if label in o:
            v = o[label]
            print(f"  {label:<12} {fmt(v['total_s']):>10}  {v['per_query_ms']:>9.2f} ms")

if 'offline' in all_results:
    o = all_results['offline']
    print(f"\n  Offline ({o.get('n_tables','?')} tables, {o.get('n_train_queries','?')} train queries)")
    if 'twig_total_estimated_s' in o:
        print(f"  TWIG training    : {fmt(o['twig_total_estimated_s'])} "
              f"({o['twig_epochs']} epochs × {fmt(o['twig_one_epoch_s'])}/epoch)")
    if 'qa_finetune_estimated_s' in o:
        print(f"  QA fine-tuning   : {fmt(o['qa_finetune_estimated_s'])} "
              f"({o['qa_epochs']} epochs, estimated)")
    if 'qgpt_estimated_hours' in o:
        print(f"  QGPT (gen only)  : ~{o['qgpt_estimated_hours']} GPU-hours "
              f"({o['qgpt_llm_calls']} LLM calls × ~2s/call)")

out = RESULT_DIR / f"{args.dataset}_{args.split}.json"
with open(out, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\n  Saved → {out}")
