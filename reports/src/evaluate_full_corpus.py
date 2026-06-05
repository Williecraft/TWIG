#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-corpus evaluation：用 train-best TWIG + QA checkpoints，
在全語料庫（train+dev+test 合併）上評估 test 查詢。

前置條件：
  1. python build_full_corpus_graph.py  （建全語料庫圖）
  2. python train_twig_qa_best.py       （訓練並儲存 TWIG + QA checkpoints）

結果輸出至：
  results/full_corpus_eval/{dataset}/results.json

用法：
  cd reports/src && python evaluate_full_corpus.py
  cd reports/src && python evaluate_full_corpus.py --datasets feta --gpu 0
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument('--gpu', type=int, default=0)
_args, _ = _p.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from query_aware.run_qa_edge_ablation import (
    QueryAwareModel, DiffusionModel,
    get_canonical_metadata, load_pretrained_into_qa_model,
    build_subgraph, filter_edges,
    QA_QUERY_EDGE_MODE, EVAL_COARSE_K, EVAL_TOP_K,
)

MODEL_NAME = 'BAAI/bge-m3'
DATASETS = ['feta', 'ottqa', 'mimo_en', 'mimo_ch', 'e2ewtq', 'mmqa']
RESULTS_DIR = PROJECT_DIR / 'results' / 'full_corpus_eval'


def full_recall_at_k(retrieved, relevant, k):
    if not relevant: return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def reciprocal_rank(retrieved, relevant):
    for i, r in enumerate(retrieved, 1):
        if r in relevant: return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved, relevant, k):
    dcg = sum(1.0 / math.log2(i+1) for i, r in enumerate(retrieved[:k], 1) if r in relevant)
    idcg = sum(1.0 / math.log2(i+1) for i in range(1, min(len(relevant), k)+1))
    return dcg / idcg if idcg > 0 else 0.0


def load_test_queries(dataset):
    """Load test queries with global-index GT from build_full_corpus_graph output."""
    query_path = PROJECT_DIR / f'data/table/full/{dataset}/query.jsonl'
    if not query_path.exists():
        print(f"  [WARN] {query_path} not found. Run build_full_corpus_graph.py first.")
        return []
    queries = []
    for line in open(query_path, encoding='utf-8'):
        obj = json.loads(line)
        q = obj.get('questions', [None])[0] if 'questions' in obj else obj.get('question')
        if not q or not q.strip():
            continue
        gt_global = {entry['_global_idx'] for entry in obj.get('_global_ground_truth', [])}
        queries.append((q.strip(), gt_global))
    return queries


def evaluate_dataset(dataset, device, embedder):
    print(f"\n{'='*60}")
    print(f"  {dataset.upper()} — full corpus evaluation")
    print(f"{'='*60}")

    # Load full corpus graph
    graph_path = PROJECT_DIR / f'data/processed/full/{dataset}/graph.pt'
    if not graph_path.exists():
        print(f"  [SKIP] Full corpus graph not found. Run build_full_corpus_graph.py first.")
        return None

    twig_path = PROJECT_DIR / f'checkpoints/{dataset}/model_qa_best_edges.pt'
    qa_path   = PROJECT_DIR / f'checkpoints/{dataset}/model_qa_best_edges_qa.pt'
    if not twig_path.exists():
        print(f"  [SKIP] TWIG checkpoint not found: {twig_path}")
        return None
    if not qa_path.exists():
        print(f"  [SKIP] QA checkpoint not found: {qa_path}")
        return None

    data_full = torch.load(str(graph_path), map_location=device, weights_only=False)
    embed_dim = data_full['table'].x.size(0)  # will fix below
    embed_dim = data_full['table'].x.size(1)
    n_tables  = data_full['table'].x.size(0)
    print(f"  Full corpus: {n_tables} tables")

    # Load TWIG checkpoint and filter edges
    twig_ckpt = torch.load(str(twig_path), map_location=device, weights_only=False)
    best_edges = twig_ckpt['best_edges']
    twig_hps   = twig_ckpt.get('hps', {})

    data_filtered = filter_edges(data_full, best_edges)
    data_filtered = data_filtered.to(device)

    twig_model = DiffusionModel(
        embed_dim=embed_dim,
        hidden_channels=twig_hps.get('HIDDEN_CHANNELS', 768),
        metadata=data_filtered.metadata(),
        dropout=twig_hps.get('DROPOUT', 0.1),
        sage_aggr=twig_hps.get('SAGE_AGGR', 'min'),
        hetero_aggr=twig_hps.get('HETERO_AGGR', 'max'),
    ).to(device)
    twig_model.load_state_dict(twig_ckpt['model_state_dict'], strict=True)
    twig_model.eval()
    print(f"  TWIG loaded (edges: {best_edges})")

    # Load QA checkpoint
    qa_ckpt  = torch.load(str(qa_path), map_location=device, weights_only=False)
    qa_metadata = get_canonical_metadata(QA_QUERY_EDGE_MODE)
    qa_model = QueryAwareModel(
        embed_dim=embed_dim,
        hidden_channels=768,
        metadata=qa_metadata,
        dropout=0.10,
        sage_aggr='min',
        hetero_aggr='max',
    ).to(device)
    qa_model.load_state_dict(qa_ckpt['model_state_dict'], strict=False)
    qa_model.eval()
    print(f"  QA loaded")

    # Load test queries (global idx GT)
    queries = load_test_queries(dataset)
    if not queries:
        print(f"  [SKIP] No test queries.")
        return None

    questions = [q for q, _ in queries]
    relevants  = [gt for _, gt in queries]
    total      = len(queries)
    eval_count = sum(1 for gt in relevants if gt)
    print(f"  {total} queries, {eval_count} with GT")

    # Embed queries
    query_vecs = torch.tensor(
        embedder.encode(questions, show_progress_bar=False),
        dtype=torch.float, device=device)
    query_vecs = F.normalize(query_vecs, p=2, dim=1)

    # TWIG coarse retrieval over full corpus
    with torch.no_grad():
        tbl_emb = twig_model.forward(data_filtered.x_dict, data_filtered.edge_index_dict)
        coarse_scores = torch.matmul(query_vecs, tbl_emb.T)
    del twig_model; torch.cuda.empty_cache()

    # TWIG-only metrics
    e0 = {k: 0.0 for k in ['r1','r2','r5','r10','r50','mrr']}
    for qi in range(total):
        gt = relevants[qi]
        if not gt: continue
        _, ti = torch.topk(coarse_scores[qi], k=min(EVAL_TOP_K, coarse_scores.size(1)))
        ret = ti.cpu().tolist()
        e0['r1']  += full_recall_at_k(ret, gt, 1)
        e0['r2']  += full_recall_at_k(ret, gt, 2)
        e0['r5']  += full_recall_at_k(ret, gt, 5)
        e0['r10'] += full_recall_at_k(ret, gt, 10)
        e0['r50'] += full_recall_at_k(ret, gt, 50)
        e0['mrr'] += reciprocal_rank(ret, gt)

    # QA rerank
    qa = {k: 0.0 for k in ['r1','r2','r5','r10','r50','mrr','ndcg10']}
    with torch.no_grad():
        for qi in tqdm(range(total), desc=f"  QA Rerank", leave=False):
            gt = relevants[qi]
            if not gt: continue
            q_vec = query_vecs[qi:qi+1]
            _, top_k_idx = torch.topk(coarse_scores[qi],
                                      k=min(EVAL_COARSE_K, coarse_scores.size(1)))
            candidates = top_k_idx.cpu().tolist()
            subgraph, table_mapping = build_subgraph(
                data_filtered, q_vec, candidates,
                edge_mode=QA_QUERY_EDGE_MODE, device=device)
            sub_emb = qa_model.forward(subgraph.x_dict, subgraph.edge_index_dict)
            final_scores = torch.matmul(q_vec, sub_emb.T).squeeze(0)
            _, reranked = torch.topk(final_scores, k=min(EVAL_TOP_K, final_scores.size(0)))
            new_to_old = {v: k for k, v in table_mapping.items()}
            ret = [new_to_old.get(i.item(), -1) for i in reranked]
            qa['r1']     += full_recall_at_k(ret, gt, 1)
            qa['r2']     += full_recall_at_k(ret, gt, 2)
            qa['r5']     += full_recall_at_k(ret, gt, 5)
            qa['r10']    += full_recall_at_k(ret, gt, 10)
            qa['r50']    += full_recall_at_k(ret, gt, 50)
            qa['mrr']    += reciprocal_rank(ret, gt)
            qa['ndcg10'] += ndcg_at_k(ret, gt, 10)

    ec = max(1, eval_count)
    result = {
        'dataset': dataset,
        'corpus': 'full (train+dev+test)',
        'n_tables': n_tables,
        'eval_count': eval_count,
        'best_edges': best_edges,
        'TWIG_R@1':   e0['r1']  / ec,
        'TWIG_R@2':   e0['r2']  / ec,
        'TWIG_R@5':   e0['r5']  / ec,
        'TWIG_R@10':  e0['r10'] / ec,
        'TWIG_R@50':  e0['r50'] / ec,
        'TWIG_MRR':   e0['mrr'] / ec,
        'QA_R@1':     qa['r1']  / ec,
        'QA_R@2':     qa['r2']  / ec,
        'QA_R@5':     qa['r5']  / ec,
        'QA_R@10':    qa['r10'] / ec,
        'QA_R@50':    qa['r50'] / ec,
        'QA_MRR':     qa['mrr'] / ec,
        'QA_nDCG@10': qa['ndcg10'] / ec,
    }

    print(f"  TWIG  R@1={result['TWIG_R@1']:.4f}  R@10={result['TWIG_R@10']:.4f}  MRR={result['TWIG_MRR']:.4f}")
    print(f"  QA    R@1={result['QA_R@1']:.4f}  R@10={result['QA_R@10']:.4f}  MRR={result['QA_MRR']:.4f}")

    del qa_model, data_full, data_filtered, query_vecs; torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    embedder = SentenceTransformer(MODEL_NAME, device=str(device))

    all_results = []
    for dataset in args.datasets:
        r = evaluate_dataset(dataset, device, embedder)
        if r is None: continue
        all_results.append(r)
        out_dir = RESULTS_DIR / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        json.dump(r, open(out_dir / 'results.json', 'w'), indent=2, ensure_ascii=False)

    del embedder

    if all_results:
        print(f"\n{'='*80}")
        print("FULL CORPUS EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"{'Dataset':<10} {'N_tables':>8} | {'TWIG R@1':>8} {'TWIG R@10':>9} {'TWIG MRR':>9} | {'QA R@1':>7} {'QA R@10':>8} {'QA MRR':>8}")
        print(f"{'-'*10} {'-'*8}-+-{'-'*8}-{'-'*9}-{'-'*9}-+-{'-'*7}-{'-'*8}-{'-'*8}")
        for r in all_results:
            print(f"{r['dataset']:<10} {r['n_tables']:>8} | "
                  f"{r['TWIG_R@1']:>8.4f} {r['TWIG_R@10']:>9.4f} {r['TWIG_MRR']:>9.4f} | "
                  f"{r['QA_R@1']:>7.4f} {r['QA_R@10']:>8.4f} {r['QA_MRR']:>8.4f}")

        # Save combined
        json.dump({'results': all_results},
                  open(RESULTS_DIR / 'all_results.json', 'w'), indent=2, ensure_ascii=False)
        print(f"\nSaved → {RESULTS_DIR}/all_results.json")

    print("\nDone.")


if __name__ == '__main__':
    main()
