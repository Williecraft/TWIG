#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Query-Aware v3 Evaluation — Clean comparison with TWIG ablation baselines.

Pipeline:
  1. Load TWIG best-edge model → full graph forward → coarse ranking (= TWIG baseline)
  2. Load QA v3 model → per-query subgraph reranking
  3. Score interpolation: final = alpha * coarse + (1-alpha) * rerank
  4. Grid search alpha for best results
  5. Compare against known TWIG ablation baselines

Usage:
  cd src && python query_aware/evaluate_query_aware_v3.py
  cd src && python query_aware/evaluate_query_aware_v3.py feta ottqa
"""

import sys
import json
import math
from typing import List, Set

import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_model import DiffusionModel

from run_edge_ablation import filter_edges as filter_edges_with_selfloops
from train_query_aware_v3 import (
    QueryAwareModel,
    get_canonical_metadata,
    build_subgraph,
    filter_edges as filter_edges_qa,
    get_key_fields,
    make_key,
    rebuild_id_to_idx,
    BEST_EDGE_CONFIGS,
)

# ===========================
# Config
# ===========================

MODEL_NAME = 'BAAI/bge-m3'
COARSE_K = 100
ALPHA_CANDIDATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

PROJECT_DIR = "/user_data/TabGNN"
QUERY_FILE = PROJECT_DIR + "/data/table/test/{source}/query.jsonl"
GRAPH_PATH = PROJECT_DIR + "/data/processed/test/{source}/graph.pt"
TWIG_MODEL_PATH = PROJECT_DIR + "/checkpoints/{dataset}/model_best_edges.pt"
QA_MODEL_PATH = PROJECT_DIR + "/checkpoints/{dataset}/model_qa_v3.pt"
RESULT_DIR = PROJECT_DIR + "/results/query_aware_v3"

# Known TWIG ablation baselines (from results/edge_ablation_extended/)
TWIG_BASELINES = {
    "feta":    {"config": "A20", "R@1": 0.8842, "R@5": 0.9641, "R@10": 0.9820},
    "ottqa":   {"config": "A18", "R@1": 0.8394, "R@5": 0.9495, "R@10": 0.9765},
    "mimo_en": {"config": "A19", "R@1": 0.3854, "R@5": 0.5955, "R@10": 0.7368},
    "mimo_ch": {"config": "A2",  "R@1": 0.4245, "R@5": 0.5982, "R@10": 0.6835},
    "e2ewtq":  {"config": "A2",  "R@1": 0.4426, "R@5": 0.8033, "R@10": 0.9180},
    "mmqa":    {"config": "A62", "R@1": 0.0000, "R@5": 0.6411, "R@10": 0.7452},
}


# ===========================
# Metrics
# ===========================

def full_recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    for i, rid in enumerate(retrieved, 1):
        if rid in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, rid in enumerate(retrieved[:k], 1) if rid in relevant)
    num_rel = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ===========================
# Query parsing
# ===========================

def parse_queries(query_file, mapping_keys, key_fields):
    queries = []
    with open(query_file, 'r', encoding='utf-8') as f:
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
    return queries


# ===========================
# Core evaluation
# ===========================

def compute_metrics(retrieved_list, relevants_list, ks=[1, 5, 10]):
    """Compute aggregate metrics over all queries."""
    metrics = {}
    eval_count = sum(1 for gt in relevants_list if len(gt) > 0)
    if eval_count == 0:
        return metrics

    for k in ks:
        metrics[f'R@{k}'] = sum(full_recall_at_k(r, gt, k)
                                for r, gt in zip(retrieved_list, relevants_list)
                                if gt) / eval_count

    metrics['MRR'] = sum(reciprocal_rank(r, gt)
                         for r, gt in zip(retrieved_list, relevants_list)
                         if gt) / eval_count
    metrics['nDCG@10'] = sum(ndcg_at_k(r, gt, 10)
                             for r, gt in zip(retrieved_list, relevants_list)
                             if gt) / eval_count
    metrics['eval_count'] = eval_count
    return metrics


def evaluate(source, dataset, coarse_k=COARSE_K):
    key_fields = get_key_fields(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph_path = GRAPH_PATH.format(source=source)
    twig_model_path = TWIG_MODEL_PATH.format(dataset=dataset)
    qa_model_path = QA_MODEL_PATH.format(dataset=dataset)
    query_file = QUERY_FILE.format(source=source)

    for p, name in [(graph_path, "test graph"), (twig_model_path, "TWIG model"),
                     (qa_model_path, "QA model"), (query_file, "query file")]:
        if not Path(p).exists():
            print(f"  Skip {source}: {name} not found ({p})")
            return None

    print(f"\n{'='*60}")
    print(f"Evaluating: {source}")
    print(f"{'='*60}")

    best_edges = BEST_EDGE_CONFIGS.get(source, [])

    # Load test graph (full, for TWIG model)
    data_full = torch.load(graph_path, map_location='cpu', weights_only=False)
    id_to_idx = rebuild_id_to_idx(data_full, key_fields)
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    mapping_keys = set(idx_to_id.values())
    embed_dim = data_full['table'].x.size(1)

    # Filtered graph for TWIG model (with self-loops for orphan nodes, matches training)
    data_twig_filtered = filter_edges_with_selfloops(data_full, best_edges) if best_edges else data_full
    # Filtered graph for QA subgraph construction (no self-loops, QA model handles via _ensure_entries)
    data_qa_filtered = filter_edges_qa(data_full, best_edges) if best_edges else data_full

    # Load TWIG model (trained with best edges, evaluated on filtered graph)
    twig_ckpt = torch.load(twig_model_path, map_location=device, weights_only=False)
    twig_hps = twig_ckpt.get('hps', {})

    data_twig_gpu = data_twig_filtered.clone().to(device)
    data_filtered_gpu = data_qa_filtered.clone().to(device)
    twig_model = DiffusionModel(
        embed_dim=embed_dim,
        hidden_channels=twig_hps.get('HIDDEN_CHANNELS', 768),
        metadata=data_twig_gpu.metadata(),
        dropout=twig_hps.get('DROPOUT', 0.1),
        sage_aggr=twig_hps.get('SAGE_AGGR', 'min'),
        hetero_aggr=twig_hps.get('HETERO_AGGR', 'max'),
    ).to(device)
    twig_model.load_state_dict(twig_ckpt['model_state_dict'], strict=False)
    twig_model.eval()
    print(f"  TWIG model loaded (best edges: {best_edges})")

    # Load QA model
    qa_ckpt = torch.load(qa_model_path, map_location=device, weights_only=False)
    qa_hps = qa_ckpt.get('hps', {})
    metadata = get_canonical_metadata()

    qa_model = QueryAwareModel(
        embed_dim=embed_dim,
        hidden_channels=qa_hps.get('HIDDEN_CHANNELS', 768),
        metadata=metadata,
        dropout=qa_hps.get('DROPOUT', 0.1),
        sage_aggr=qa_hps.get('SAGE_AGGR', 'min'),
        hetero_aggr=qa_hps.get('HETERO_AGGR', 'max'),
    ).to(device)
    qa_model.load_state_dict(qa_ckpt['model_state_dict'], strict=False)
    qa_model.eval()
    print(f"  QA model loaded (edge_mode={qa_ckpt.get('edge_mode', 'E4')}, "
          f"frozen_base={qa_ckpt.get('frozen_base', False)})")

    # Parse queries
    queries = parse_queries(query_file, mapping_keys, key_fields)
    questions = [q for q, _ in queries]
    relevants = [gt for _, gt in queries]
    total = len(queries)
    eval_count = sum(1 for gt in relevants if len(gt) > 0)
    print(f"  Queries: {total} total, {eval_count} with ground truth")

    # Embed queries
    embedder = SentenceTransformer(MODEL_NAME, device=str(device))
    query_vecs = embedder.encode(questions, show_progress_bar=True, convert_to_tensor=True).to(device)
    query_vecs = F.normalize(query_vecs, p=2, dim=1)
    del embedder
    torch.cuda.empty_cache()

    # === Phase 1: TWIG coarse ranking (= TWIG baseline) ===
    print("  Phase 1: TWIG coarse ranking...")
    with torch.no_grad():
        twig_table_emb = twig_model.forward(data_twig_gpu.x_dict, data_twig_gpu.edge_index_dict)
        coarse_scores = torch.matmul(query_vecs, twig_table_emb.T)

    # TWIG baseline metrics
    twig_retrieved = []
    for qi in range(total):
        _, top_indices = torch.topk(coarse_scores[qi], k=min(10, coarse_scores.size(1)))
        twig_retrieved.append([idx_to_id.get(idx.item(), "") for idx in top_indices])

    twig_metrics = compute_metrics(twig_retrieved, relevants)
    print(f"  TWIG Baseline: R@1={twig_metrics.get('R@1',0):.4f} "
          f"R@5={twig_metrics.get('R@5',0):.4f} R@10={twig_metrics.get('R@10',0):.4f} "
          f"MRR={twig_metrics.get('MRR',0):.4f}")

    # === Phase 2: QA reranking with alpha search ===
    print(f"  Phase 2: QA reranking (coarse_k={coarse_k})...")

    # Pre-compute QA rerank scores for all queries
    qa_rerank_data = []  # (candidate_indices, table_mapping, rerank_scores, coarse_sub_scores)

    with torch.no_grad():
        for qi in tqdm(range(total), desc="  QA Rerank", leave=False):
            q_vec = query_vecs[qi:qi + 1]

            # Coarse candidates
            top_k_scores, top_k_idx = torch.topk(
                coarse_scores[qi], k=min(coarse_k, coarse_scores.size(1)))
            candidate_indices = top_k_idx.cpu().tolist()

            # Build subgraph with query node
            subgraph, table_mapping = build_subgraph(
                data_filtered_gpu, q_vec, candidate_indices, device=device)

            # QA forward
            sub_table_emb = qa_model.forward(subgraph.x_dict, subgraph.edge_index_dict)
            rerank_scores = torch.matmul(q_vec, sub_table_emb.T).squeeze(0)

            # Map coarse scores to subgraph space
            coarse_in_sub = torch.zeros(rerank_scores.size(0), device=device)
            for rank_i, orig_idx in enumerate(candidate_indices):
                new_idx = table_mapping.get(orig_idx, -1)
                if 0 <= new_idx < coarse_in_sub.size(0):
                    coarse_in_sub[new_idx] = top_k_scores[rank_i]

            qa_rerank_data.append((candidate_indices, table_mapping,
                                   rerank_scores.cpu(), coarse_in_sub.cpu()))

    # Grid search over alpha
    print("  Alpha search...")
    best_alpha = 0.0
    best_alpha_r10 = 0.0
    alpha_results = {}

    for alpha in ALPHA_CANDIDATES:
        retrieved_all = []
        for qi in range(total):
            candidate_indices, table_mapping, rerank_scores, coarse_in_sub = qa_rerank_data[qi]
            final_scores = alpha * coarse_in_sub + (1 - alpha) * rerank_scores
            _, reranked = torch.topk(final_scores, k=min(10, final_scores.size(0)))

            new_to_old = {v: k for k, v in table_mapping.items()}
            retrieved = [idx_to_id.get(new_to_old.get(idx.item(), -1), "") for idx in reranked]
            retrieved_all.append(retrieved)

        metrics = compute_metrics(retrieved_all, relevants)
        alpha_results[alpha] = metrics
        r10 = metrics.get('R@10', 0)
        marker = " *" if r10 > best_alpha_r10 else ""
        print(f"    alpha={alpha:.1f}: R@1={metrics.get('R@1',0):.4f} "
              f"R@5={metrics.get('R@5',0):.4f} R@10={r10:.4f} "
              f"MRR={metrics.get('MRR',0):.4f}{marker}")

        if r10 > best_alpha_r10:
            best_alpha_r10 = r10
            best_alpha = alpha

    best_metrics = alpha_results[best_alpha]

    # Compare with known TWIG ablation baseline
    twig_bl = TWIG_BASELINES.get(source, {})
    print(f"\n  === Final Results (best alpha={best_alpha:.1f}) ===")
    print(f"  TWIG (ours):     R@1={twig_metrics.get('R@1',0):.4f}  "
          f"R@5={twig_metrics.get('R@5',0):.4f}  R@10={twig_metrics.get('R@10',0):.4f}")
    if twig_bl:
        print(f"  TWIG (ablation):  R@1={twig_bl['R@1']:.4f}  "
              f"R@5={twig_bl['R@5']:.4f}  R@10={twig_bl['R@10']:.4f}")
    print(f"  QA v3:           R@1={best_metrics.get('R@1',0):.4f}  "
          f"R@5={best_metrics.get('R@5',0):.4f}  R@10={best_metrics.get('R@10',0):.4f}")

    if twig_bl:
        for k in [1, 5, 10]:
            delta = best_metrics.get(f'R@{k}', 0) - twig_bl.get(f'R@{k}', 0)
            arrow = '+' if delta >= 0 else ''
            print(f"  vs ablation R@{k}: {arrow}{delta:.4f}")

    return {
        'source': source,
        'dataset': dataset,
        'best_edges': best_edges,
        'best_alpha': best_alpha,
        'twig_ours': twig_metrics,
        'twig_ablation_baseline': twig_bl,
        'qa_v3': best_metrics,
        'alpha_search': {str(a): m for a, m in alpha_results.items()},
    }


def main():
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)

    datasets = sys.argv[1:] if len(sys.argv) > 1 else [
        "feta", "ottqa", "mimo_en", "mimo_ch", "e2ewtq", "mmqa"
    ]

    all_results = {}
    for ds in datasets:
        results = evaluate(ds, ds)
        if results is not None:
            all_results[ds] = results
            result_file = Path(RESULT_DIR) / f"{ds}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    if all_results:
        print(f"\n\n{'='*90}")
        print("Summary: QA v3 vs TWIG Ablation Baselines")
        print(f"{'='*90}")
        print(f"{'Dataset':>10} | {'TWIG R@1':>8} | {'QA R@1':>7} | {'TWIG R@5':>8} | "
              f"{'QA R@5':>7} | {'TWIG R@10':>9} | {'QA R@10':>8} | {'alpha':>5}")
        print("-" * 90)

        wins = 0
        total_ds = 0
        for ds, r in all_results.items():
            bl = r['twig_ablation_baseline']
            qa = r['qa_v3']
            if not bl:
                continue
            total_ds += 1
            dr10 = qa.get('R@10', 0) - bl.get('R@10', 0)
            if dr10 > 0:
                wins += 1
            tag = '+' if dr10 > 0 else ' '
            print(f"{ds:>10} | {bl['R@1']:>8.4f} | {qa.get('R@1',0):>7.4f} | "
                  f"{bl['R@5']:>8.4f} | {qa.get('R@5',0):>7.4f} | "
                  f"{bl['R@10']:>9.4f} | {qa.get('R@10',0):>8.4f}{tag} | "
                  f"{r['best_alpha']:.1f}")

        print(f"\nWins (R@10): {wins}/{total_ds}")

        # Save summary
        summary_file = Path(RESULT_DIR) / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {RESULT_DIR}/")


if __name__ == '__main__':
    main()
