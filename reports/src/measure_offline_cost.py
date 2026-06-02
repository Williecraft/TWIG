#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline Cost Measurement: TWIG vs QGPT on FeTaQA

TWIG offline (完整流程，與 build_graph.py + train_model.py 一致):
  1. Graph construction  → 呼叫 build_graph.main('train/feta')
  2. TWIG training       → 完整訓練迴圈，含 hard neg mining + validation

QGPT offline:
  1. LLM pseudo-query generation (LLaMA-3.1-8B via Ollama, 1 call/table)
  2. Embed (table snippet + generated questions) with bge-m3

Usage:
  cd reports/src && python measure_offline_cost.py [--gpu 0] [--qgpt-sample 50]
  cd reports/src && python measure_offline_cost.py --twig-only
  cd reports/src && python measure_offline_cost.py --qgpt-only
"""

import argparse
import os
import sys
import time
import json
import random
import copy
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--qgpt-sample', type=int, default=50)
parser.add_argument('--twig-only', action='store_true')
parser.add_argument('--qgpt-only', action='store_true')
parser.add_argument('--twig-epochs', type=int, default=2,
                    help='Epochs to actually run for training timing (then extrapolate)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

DATASET    = 'feta'
SOURCE     = f'train/{DATASET}'
MODEL_NAME = 'BAAI/bge-m3'
RESULT_DIR = PROJECT_DIR / 'results' / 'cost_measurement'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_TABLE_FILE = PROJECT_DIR / f'data/table/{SOURCE}/table.jsonl'
TRAIN_GRAPH_FILE = PROJECT_DIR / f'data/processed/{SOURCE}/graph.pt'
TRAIN_QUERY_FILE = PROJECT_DIR / f'data/table/{SOURCE}/query.jsonl'
VAL_GRAPH_FILE   = PROJECT_DIR / f'data/processed/dev/{DATASET}/graph.pt'
VAL_QUERY_FILE   = PROJECT_DIR / f'data/table/dev/{DATASET}/query.jsonl'

# 與 train_model.py BEST_PARAMS 一致
TWIG_HPS = dict(
    LEARNING_RATE=0.0005543418807199451,
    HIDDEN_CHANNELS=768,
    DROPOUT=0.10028905529000982,
    WEIGHT_DECAY=0.03217253330215496,
    SAGE_AGGR='min',
    HETERO_AGGR='max',
    CLIP_GRAD_NORM=0.60,
    TEMP_START=0.05,
    TEMP_END=0.03,
    SMOOTH_START=0.120,
    SMOOTH_END=0.060,
    NUM_EPOCHS=30,
    WARMUP_EPOCHS=2,
    BATCH_SIZE=128,
    CHUNK_SIZE=1024,
)
NUM_HARD_NEGATIVES = 3
REMINING_INTERVAL  = 1   # 每 epoch 都重新挖掘（與 train_model.py 一致）


def fmt(s):
    if s < 60:   return f'{s:.1f}s'
    if s < 3600: return f'{s/60:.1f}min'
    return f'{s/3600:.2f}h'


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_key(item, key_fields):
    return '|'.join(str(item.get(f, '')) for f in key_fields)


KEY_FIELDS = ('sheet_name', 'file_name')


# ══════════════════════════════════════════════════════
# TWIG Offline
# ══════════════════════════════════════════════════════

def measure_twig_offline(device):
    print(f"\n{'='*60}")
    print(f'TWIG Offline Cost — {SOURCE}')
    print(f'  build_graph.main()  +  full training loop')
    print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print('='*60)

    results = {}

    # ── 1. Graph Construction ─────────────────────────────────
    print(f'\n[1/2] Graph Construction — calling build_graph.main("{SOURCE}")')
    print('  (will overwrite existing graph.pt for fresh timing)')

    torch.cuda.empty_cache()
    import build_graph as bg

    sync(); t0 = time.perf_counter()
    bg.main(source=SOURCE)
    sync(); t_build = time.perf_counter() - t0

    print(f'\n  ✓ Graph build: {fmt(t_build)}')
    results['graph_build_s'] = round(t_build, 1)

    # ── 2. TWIG Training ─────────────────────────────────────
    print(f'\n[2/2] TWIG Training — full loop ({args.twig_epochs} epochs timed, '
          f'extrapolating to {TWIG_HPS["NUM_EPOCHS"]})')
    print('  Includes: hard-neg mining + forward + backward + validation per epoch')

    from train_model import DiffusionModel, mine_hard_negatives_topk

    # Load graph (freshly built above)
    torch.cuda.empty_cache()
    data_cpu = torch.load(TRAIN_GRAPH_FILE, map_location='cpu', weights_only=False)
    embed_dim = data_cpu['table'].x.size(1)
    n_tables  = data_cpu['table'].x.size(0)

    # Build id_to_idx
    id_to_idx = {}
    if hasattr(data_cpu, 'metadata_maps') and 'table_meta' in data_cpu.metadata_maps:
        for i, m in enumerate(data_cpu.metadata_maps['table_meta']):
            id_to_idx[make_key(m, KEY_FIELDS)] = i

    # Load train queries
    texts, pos_lists = [], []
    hard_neg_init = []
    with open(TRAIN_QUERY_FILE) as f:
        for line in f:
            obj = json.loads(line)
            qs = obj.get('questions', []) or ([obj['question']] if 'question' in obj else [])
            pos = [id_to_idx[make_key(g, KEY_FIELDS)]
                   for g in obj.get('ground_truth_list', []) or []
                   if make_key(g, KEY_FIELDS) in id_to_idx]
            if pos:
                for q in qs:
                    if q and q.strip():
                        texts.append(q.strip())
                        pos_lists.append(pos)
                        hard_neg_init.append([-1] * NUM_HARD_NEGATIVES)

    n_queries = len(texts)
    print(f'  Tables: {n_tables} | Train queries: {n_queries}')

    # Load val queries
    val_data_cpu = None
    val_q_vecs   = None
    val_pos_lists = None
    if VAL_GRAPH_FILE.exists() and VAL_QUERY_FILE.exists():
        val_data_cpu = torch.load(VAL_GRAPH_FILE, map_location='cpu', weights_only=False)
        val_id_to_idx = {}
        if hasattr(val_data_cpu, 'metadata_maps') and 'table_meta' in val_data_cpu.metadata_maps:
            for i, m in enumerate(val_data_cpu.metadata_maps['table_meta']):
                val_id_to_idx[make_key(m, KEY_FIELDS)] = i
        val_texts, val_pos_lists = [], []
        with open(VAL_QUERY_FILE) as f:
            for line in f:
                obj = json.loads(line)
                qs = obj.get('questions', []) or ([obj['question']] if 'question' in obj else [])
                pos = [val_id_to_idx[make_key(g, KEY_FIELDS)]
                       for g in obj.get('ground_truth_list', []) or []
                       if make_key(g, KEY_FIELDS) in val_id_to_idx]
                if pos:
                    for q in qs:
                        if q and q.strip():
                            val_texts.append(q.strip())
                            val_pos_lists.append(pos)

    # Embed queries (part of offline prep)
    embedder = SentenceTransformer(MODEL_NAME, device=str(device))
    print('  Embedding train queries...')
    t_embed0 = time.perf_counter()
    q_vecs = torch.tensor(
        embedder.encode(texts, show_progress_bar=True, batch_size=256),
        dtype=torch.float, device=device)
    t_embed = time.perf_counter() - t_embed0
    print(f'  Train query embedding: {fmt(t_embed)}')

    if val_texts:
        val_q_vecs = torch.tensor(
            embedder.encode(val_texts, show_progress_bar=False, batch_size=256),
            dtype=torch.float, device=device)
    del embedder; torch.cuda.empty_cache()

    # Setup model
    data = data_cpu.clone().to(device)
    model = DiffusionModel(
        embed_dim, TWIG_HPS['HIDDEN_CHANNELS'], data.metadata(),
        TWIG_HPS['DROPOUT'], TWIG_HPS['SAGE_AGGR'], TWIG_HPS['HETERO_AGGR']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(),
                            lr=TWIG_HPS['LEARNING_RATE'],
                            weight_decay=TWIG_HPS['WEIGHT_DECAY'])
    warmup_sched = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e+1)/TWIG_HPS['WARMUP_EPOCHS']))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=TWIG_HPS['NUM_EPOCHS']-TWIG_HPS['WARMUP_EPOCHS'])
    scheduler    = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                                milestones=[TWIG_HPS['WARMUP_EPOCHS']])
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    hard_negs = hard_neg_init
    epoch_times = []

    # Actually run N epochs (full loop, matching train_model.py exactly)
    for epoch in range(1, args.twig_epochs + 1):
        sync(); t_ep0 = time.perf_counter()

        # Step A: Hard negative re-mining (full GNN forward, every epoch)
        hard_negs = mine_hard_negatives_topk(
            model, data, q_vecs, pos_lists,
            num_hard_negatives=NUM_HARD_NEGATIVES, device=str(device))

        # Step B: Train one epoch
        model.train()
        progress  = epoch / TWIG_HPS['NUM_EPOCHS']
        curr_temp   = TWIG_HPS['TEMP_END'] if epoch > TWIG_HPS['NUM_EPOCHS'] * 0.7 else TWIG_HPS['TEMP_START']
        curr_smooth = TWIG_HPS['SMOOTH_START'] + (TWIG_HPS['SMOOTH_END']-TWIG_HPS['SMOOTH_START']) * progress
        indices = list(range(n_queries)); random.shuffle(indices)

        for start in range(0, n_queries, TWIG_HPS['BATCH_SIZE']):
            b = indices[start:start+TWIG_HPS['BATCH_SIZE']]
            qb = q_vecs[b]
            labels = torch.tensor([pos_lists[i][0] for i in b], dtype=torch.long, device=device)
            optimizer.zero_grad()
            with autocast(enabled=(device.type == 'cuda')):
                tbl = model.forward(data.x_dict, data.edge_index_dict)
                qn  = F.normalize(qb, p=2, dim=1)
                logits = torch.matmul(qn, tbl.T) / curr_temp
                loss = F.cross_entropy(logits, labels, label_smoothing=curr_smooth)
                # Hard negative margin loss
                batch_hn = torch.tensor([hard_negs[i] for i in b], device=device, dtype=torch.long)
                mask = (batch_hn != -1)
                if mask.any():
                    safe_hn = batch_hn.clone(); safe_hn[~mask] = 0
                    pos_sc = logits[range(len(b)), labels]
                    neg_sc = torch.gather(logits, 1, safe_hn)
                    margin_loss = F.relu(neg_sc - pos_sc.unsqueeze(1) + 0.2/curr_temp) * mask.float()
                    loss = loss + 0.5 * margin_loss.sum() / (mask.sum() + 1e-9)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TWIG_HPS['CLIP_GRAD_NORM'])
            scaler.step(optimizer); scaler.update()

        scheduler.step()

        # Step C: Validation evaluation (full GNN forward on val graph)
        if val_data_cpu is not None and val_q_vecs is not None:
            val_data = val_data_cpu.clone().to(device)
            model.eval()
            with torch.no_grad():
                val_tbl = model.forward(val_data.x_dict, val_data.edge_index_dict)
                val_qn  = F.normalize(val_q_vecs, p=2, dim=1)
                val_scores = torch.matmul(val_qn, val_tbl.T).cpu()
                r10 = sum(
                    1 for qi in range(val_scores.size(0))
                    if any(idx.item() in set(val_pos_lists[qi])
                           for idx in torch.topk(val_scores[qi], k=min(10, val_scores.size(1))).indices)
                ) / max(1, val_scores.size(0))
            del val_data; torch.cuda.empty_cache()

        sync(); t_ep = time.perf_counter() - t_ep0
        epoch_times.append(t_ep)
        val_str = f' | Val R@10={r10:.4f}' if val_data_cpu is not None else ''
        print(f'  Epoch {epoch}/{args.twig_epochs}: {fmt(t_ep)}{val_str}')

    t_epoch_avg  = sum(epoch_times) / len(epoch_times)
    t_train_est  = t_epoch_avg * TWIG_HPS['NUM_EPOCHS']
    t_total_est  = t_build + t_embed + t_train_est

    print(f'\n  Avg epoch time: {fmt(t_epoch_avg)}')
    print(f'  Training estimate ({TWIG_HPS["NUM_EPOCHS"]} epochs): {fmt(t_train_est)}')
    print(f'  Total TWIG offline: {fmt(t_total_est)}')
    print(f'    = graph build ({fmt(t_build)}) + query embed ({fmt(t_embed)}) + train ({fmt(t_train_est)})')

    results.update({
        'n_tables': int(n_tables),
        'n_train_queries': n_queries,
        'graph_build_s': round(t_build, 1),
        'query_embed_s': round(t_embed, 1),
        'epoch_times_s': [round(t, 2) for t in epoch_times],
        'avg_epoch_s': round(t_epoch_avg, 1),
        'training_estimated_s': round(t_train_est, 1),
        'total_estimated_s': round(t_total_est, 1),
        'epochs_measured': args.twig_epochs,
        'epochs_total': TWIG_HPS['NUM_EPOCHS'],
    })

    del model, data, q_vecs; torch.cuda.empty_cache()
    return results


# ══════════════════════════════════════════════════════
# QGPT Offline
# ══════════════════════════════════════════════════════

QGPT_PROMPT = """\
You are an expert in table data analysis.
Given a table with its file name, sheet name, and a portion of its content (first ten rows), \
your task is to extract key headers and generate questions based on the table & headers.

Important: ignore nan/Unnamed values. Generate questions answerable only from this table, \
each involving 1-3 headers. Total questions > half the number of headers.

Output (strictly JSON, no other text):
{{"headers": ["header1", ...], "questions": ["question1", ...]}}

Table Meta
- File: {file_name}
- Sheet: {sheet_name}

Table Preview
{table_csv}
"""


def table_to_csv_preview(table: dict, top: int = 10) -> str:
    import io
    header = table.get('header', [])
    if isinstance(header, list) and len(header) == 1 and isinstance(header[0], str):
        header = next(csv.reader([header[0]]))
    instances = table.get('instances', [])[:top]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for row in instances:
        if isinstance(row, str):
            w.writerow(next(csv.reader([row])))
        else:
            w.writerow(['' if x is None else str(x) for x in row])
    return buf.getvalue()


def measure_qgpt_offline(device, sample_n: int):
    print(f"\n{'='*60}")
    print(f'QGPT Offline Cost — {SOURCE}')
    print(f'  LLM: LLaMA-3.1-8B (Ollama)  |  sample={sample_n} tables')
    print('='*60)

    with open(PROJECT_DIR / 'config/api_keys.json') as f:
        api_key = json.load(f)['Ollama']
    from OllamaAgent import Agent
    agent = Agent(api_key=api_key)

    with open(TRAIN_TABLE_FILE) as f:
        all_tables = [json.loads(line) for line in f]
    n_tables = len(all_tables)
    print(f'  Total tables: {n_tables}')

    sample = random.sample(all_tables, min(sample_n, n_tables))
    random.shuffle(sample)

    # ── LLM generation ────────────────────────────────────────
    print(f'\n[1/2] LLM pseudo-query generation ({len(sample)} tables sampled)')
    llm_times = []
    generated_texts = []

    for i, table in enumerate(sample):
        prompt = QGPT_PROMPT.format(
            file_name=table.get('file_name', ''),
            sheet_name=table.get('sheet_name', ''),
            table_csv=table_to_csv_preview(table, top=10),
        )
        t0 = time.perf_counter()
        response = agent.query(prompt)
        elapsed = time.perf_counter() - t0
        llm_times.append(elapsed)

        try:
            obj = json.loads(response.replace('```json','').replace('```','').strip())
            questions = obj.get('questions', [])
        except Exception:
            questions = []
        generated_texts.append((table_to_csv_preview(table, top=10), ' '.join(questions)))

        if (i+1) % 10 == 0 or i == 0:
            avg = sum(llm_times) / len(llm_times)
            print(f'  [{i+1:>3}/{len(sample)}] avg={avg:.2f}s/table  est. total={fmt(avg*n_tables)}')

    avg_llm   = sum(llm_times) / len(llm_times)
    med_llm   = sorted(llm_times)[len(llm_times)//2]
    t_llm_est = avg_llm * n_tables
    print(f'\n  LLM avg={avg_llm:.2f}s  median={med_llm:.2f}s  → est. total={fmt(t_llm_est)}')

    # ── Embedding ─────────────────────────────────────────────
    print(f'\n[2/2] Embedding table snippets + generated questions')
    torch.cuda.empty_cache()
    embedder = SentenceTransformer(MODEL_NAME, device=str(device))
    embed_texts = [f'{t[0]}\n{t[1]}' for t in generated_texts]

    sync(); t0 = time.perf_counter()
    _ = torch.tensor(
        embedder.encode(embed_texts, show_progress_bar=True, batch_size=256),
        dtype=torch.float, device=device)
    sync(); t_embed_sample = time.perf_counter() - t0

    t_embed_est = t_embed_sample / len(sample) * n_tables
    print(f'  Sample embed ({len(sample)} tables): {fmt(t_embed_sample)}  → est. total={fmt(t_embed_est)}')
    del embedder; torch.cuda.empty_cache()

    t_total_est = t_llm_est + t_embed_est
    print(f'\n  Total QGPT offline estimate: {fmt(t_total_est)}')

    return {
        'n_tables': n_tables,
        'sample_n': len(sample),
        'llm_model': 'LLaMA-3.1-8B-instruct (Ollama)',
        'llm_avg_s': round(avg_llm, 3),
        'llm_median_s': round(med_llm, 3),
        'llm_total_estimated_s': round(t_llm_est, 1),
        'embed_sample_s': round(t_embed_sample, 1),
        'embed_total_estimated_s': round(t_embed_est, 1),
        'total_estimated_s': round(t_total_est, 1),
        'llm_times_sample': [round(t, 3) for t in llm_times],
    }


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory//1024**3}GB)')

    all_results = {}

    if not args.qgpt_only:
        all_results['twig'] = measure_twig_offline(device)

    if not args.twig_only:
        all_results['qgpt'] = measure_qgpt_offline(device, sample_n=args.qgpt_sample)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print('OFFLINE COST SUMMARY')
    print('='*60)

    if 'twig' in all_results:
        t = all_results['twig']
        print(f'\nTWIG ({t["n_tables"]} tables, {t["n_train_queries"]} train queries):')
        print(f'  Graph build        : {fmt(t["graph_build_s"])}')
        print(f'  Query embedding    : {fmt(t["query_embed_s"])}')
        print(f'  Training estimate  : {fmt(t["training_estimated_s"])}  '
              f'({t["epochs_total"]} epochs × {fmt(t["avg_epoch_s"])}/epoch)')
        print(f'  ─────────────────────────────────')
        print(f'  Total (estimated)  : {fmt(t["total_estimated_s"])}')

    if 'qgpt' in all_results:
        q = all_results['qgpt']
        print(f'\nQGPT (LLaMA-3.1-8B, {q["sample_n"]} tables sampled):')
        print(f'  LLM generation     : {fmt(q["llm_total_estimated_s"])}  '
              f'(avg {q["llm_avg_s"]:.2f}s/table × {q["n_tables"]} tables)')
        print(f'  Embedding (est.)   : {fmt(q["embed_total_estimated_s"])}')
        print(f'  ─────────────────────────────────')
        print(f'  Total (estimated)  : {fmt(q["total_estimated_s"])}')

    out = RESULT_DIR / f'{DATASET}_offline.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f'\n  Saved → {out}')


if __name__ == '__main__':
    main()
