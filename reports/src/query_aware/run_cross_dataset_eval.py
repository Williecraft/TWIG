#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-Dataset Generalization Evaluation

Tests whether a model trained on dataset A can generalize to dataset B's graph.
Since GraphSAGE produces inductive embeddings, we can apply a trained model
to a completely different graph (different tables, different structure).

Pipeline for each (train_dataset, test_dataset) pair:
  1. Load the trained QA v2 model (from train_dataset)
  2. Load the test graph (from test_dataset)
  3. Filter edges using the best edge config (from ablation results or config)
  4. Run TWIG coarse ranking + QA reranking on the test graph
  5. Report Recall@1/5/10, MRR, nDCG@10

Usage:
  cd ~/TabGNN/src/query_aware
  python run_cross_dataset_eval.py --gpu 0
  python run_cross_dataset_eval.py --gpu 0 --best-config '{"feta": 18, "ottqa": 2}'
  python run_cross_dataset_eval.py --gpu 0 --train-datasets feta ottqa --test-datasets mimo_en mimo_ch
"""

# ========= Parse GPU before importing torch =========
import argparse
import sys
import os

_temp_parser = argparse.ArgumentParser(add_help=False)
_temp_parser.add_argument('--gpu', type=int, default=0)
_temp_args, _ = _temp_parser.parse_known_args()
if _temp_args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_temp_args.gpu)

import json
import math
import csv
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Import from the ablation script (same directory)
from run_qa_edge_ablation import (
    DiffusionModel,
    QueryAwareModel,
    get_canonical_metadata,
    build_subgraph,
    filter_edges,
    build_id_to_idx,
    load_queries,
    make_key,
    get_key_fields,
    binary_to_edges,
    get_config_label,
    full_recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    QA_QUERY_EDGE_MODE,
    ALL_EDGE_RELATIONS,
)

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_NAME = 'BAAI/bge-m3'
ALL_DATASETS = ["feta", "ottqa", "mimo_en", "mimo_ch", "e2ewtq", "mmqa"]

EVAL_COARSE_K = 100
EVAL_TOP_K = 10
EVAL_ALPHA = 0.3

# Default best edge configs from the original TWIG ablation.
# These will be overridden by --best-config or by reading ablation results.
DEFAULT_BEST_EDGE_CONFIGS = {
    "feta": ["has_column", "same_page"],
    "ottqa": ["has_column", "similar_content"],
    "mimo_en": ["has_column", "similar_content", "shared_column_name"],
    "mimo_ch": ["similar_content"],
    "e2ewtq": ["similar_content"],
    "mmqa": ["similar_table", "has_column", "comes_from", "same_page", "similar_content"],
}

# Paths
QA_MODEL_PATH = str(PROJECT_DIR / "checkpoints/{dataset}/model_query_aware_v2.pt")
TWIG_MODEL_PATH = str(PROJECT_DIR / "checkpoints/{dataset}/model.pt")
TEST_GRAPH_PATH = str(PROJECT_DIR / "data/processed/test/{dataset}/graph.pt")
TEST_QUERY_PATH = str(PROJECT_DIR / "data/table/test/{dataset}/query.jsonl")
RESULTS_DIR = str(PROJECT_DIR / "results" / "cross_dataset_eval")


def load_best_config_from_ablation(dataset):
    """Try to load the best QA edge config from ablation results."""
    ablation_json = PROJECT_DIR / "results" / "qa_edge_ablation" / dataset / "results.json"
    if not ablation_json.exists():
        return None

    try:
        with open(ablation_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        results = data.get('results', [])
        if not results:
            return None

        # Find the config with the highest QA_R@10
        best = max(results, key=lambda r: r.get('QA_R@10', 0))
        best_code = best['code']
        best_edges = binary_to_edges(best_code)
        print(f"  Best QA edge config for {dataset}: A{best_code} = {best_edges} "
              f"(R@10={best.get('QA_R@10', 0):.4f})")
        return best_edges
    except Exception as e:
        print(f"  Warning: could not load ablation results for {dataset}: {e}")
        return None


def evaluate_cross(train_dataset, test_dataset, best_edges, device, embedder):
    """
    Evaluate a model trained on train_dataset against test_dataset's test graph.

    Args:
        train_dataset: dataset whose trained model we use
        test_dataset: dataset whose test graph + queries we evaluate on
        best_edges: list of edge types to keep
        device: torch device
        embedder: SentenceTransformer instance

    Returns:
        dict with evaluation metrics, or None if data missing
    """
    key_fields_test = get_key_fields(test_dataset)
    key_fields_train = get_key_fields(train_dataset)

    # Paths for the model (from training dataset)
    qa_model_path = QA_MODEL_PATH.format(dataset=train_dataset)
    twig_model_path = TWIG_MODEL_PATH.format(dataset=train_dataset)

    # Paths for test data (from test dataset)
    test_graph_path = TEST_GRAPH_PATH.format(dataset=test_dataset)
    test_query_path = TEST_QUERY_PATH.format(dataset=test_dataset)

    # Check files exist
    for path, desc in [(qa_model_path, "QA model"), (twig_model_path, "TWIG model"),
                       (test_graph_path, "Test graph"), (test_query_path, "Test queries")]:
        if not Path(path).exists():
            print(f"    Skipping: {desc} not found at {path}")
            return None

    print(f"  {train_dataset} -> {test_dataset}")

    # Load test graph
    data_full = torch.load(test_graph_path, map_location=device, weights_only=False)
    id_to_idx = build_id_to_idx(data_full, key_fields_test)
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    mapping_keys = set(idx_to_id.values())
    embed_dim = data_full['table'].x.size(1)

    # Filter edges for QA subgraphs
    if best_edges:
        data_filtered = filter_edges(data_full, best_edges)
    else:
        data_filtered = data_full

    # Load QA model (trained on train_dataset, using canonical metadata)
    ckpt = torch.load(qa_model_path, map_location=device, weights_only=False)
    hps = ckpt.get('hps', {})
    ckpt_edge_mode = ckpt.get('edge_mode', QA_QUERY_EDGE_MODE)
    metadata = get_canonical_metadata(ckpt_edge_mode)

    qa_model = QueryAwareModel(
        embed_dim=embed_dim,
        hidden_channels=hps.get('HIDDEN_CHANNELS', 768),
        metadata=metadata,
        dropout=hps.get('DROPOUT', 0.1),
        sage_aggr=hps.get('SAGE_AGGR', 'min'),
        hetero_aggr=hps.get('HETERO_AGGR', 'max'),
    ).to(device)
    qa_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    qa_model.eval()

    # Load TWIG model (for coarse ranking on full graph)
    twig_ckpt = torch.load(twig_model_path, map_location=device, weights_only=False)
    twig_hps = twig_ckpt.get('hps', hps)
    twig_metadata = data_full.metadata()

    twig_model = DiffusionModel(
        embed_dim=embed_dim,
        hidden_channels=twig_hps.get('HIDDEN_CHANNELS', 768),
        metadata=twig_metadata,
        dropout=twig_hps.get('DROPOUT', 0.1),
        sage_aggr=twig_hps.get('SAGE_AGGR', 'min'),
        hetero_aggr=twig_hps.get('HETERO_AGGR', 'max'),
    ).to(device)
    twig_model.load_state_dict(twig_ckpt['model_state_dict'], strict=False)
    twig_model.eval()

    # Parse test queries
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
                if all(gt.get(field) is not None for field in key_fields_test):
                    key = make_key(gt, key_fields_test)
                    if key in mapping_keys:
                        gt_keys.add(key)
            queries.append((question, gt_keys))

    questions = [q for q, _ in queries]
    relevants = [gt for _, gt in queries]
    total = len(queries)
    eval_count = sum(1 for gt in relevants if len(gt) > 0)

    if eval_count == 0:
        print(f"    No evaluable queries for {test_dataset}")
        del qa_model, twig_model, data_full, data_filtered
        torch.cuda.empty_cache()
        return None

    # Embed queries
    query_vecs = torch.tensor(
        embedder.encode(questions, show_progress_bar=False),
        dtype=torch.float, device=device)
    query_vecs = F.normalize(query_vecs, p=2, dim=1)

    # Coarse ranking with TWIG
    data_full = data_full.to(device)
    data_filtered = data_filtered.to(device)
    with torch.no_grad():
        table_emb = twig_model.forward(data_full.x_dict, data_full.edge_index_dict)
        coarse_scores = torch.matmul(query_vecs, table_emb.T)

    del twig_model
    torch.cuda.empty_cache()

    # TWIG baseline
    twig_r1 = twig_r5 = twig_r10 = twig_mrr = 0.0
    for qi in range(total):
        gt = relevants[qi]
        if not gt:
            continue
        _, top_idx = torch.topk(coarse_scores[qi], k=min(EVAL_TOP_K, coarse_scores.size(1)))
        retrieved = [idx_to_id.get(idx.item(), "") for idx in top_idx]
        twig_r1 += full_recall_at_k(retrieved, gt, 1)
        twig_r5 += full_recall_at_k(retrieved, gt, 5)
        twig_r10 += full_recall_at_k(retrieved, gt, 10)
        twig_mrr += reciprocal_rank(retrieved, gt)

    # QA reranking
    qa_r1 = qa_r5 = qa_r10 = qa_mrr = qa_ndcg10 = 0.0
    with torch.no_grad():
        for qi in tqdm(range(total), desc=f"    {train_dataset}->{test_dataset}", leave=False):
            gt = relevants[qi]
            if not gt:
                continue

            q_vec = query_vecs[qi:qi + 1]
            top_k_scores, top_k_idx = torch.topk(
                coarse_scores[qi], k=min(EVAL_COARSE_K, coarse_scores.size(1)))
            candidate_indices = top_k_idx.cpu().tolist()

            subgraph, table_mapping = build_subgraph(
                data_filtered, q_vec, candidate_indices,
                edge_mode=ckpt_edge_mode, device=device)
            sub_emb = qa_model.forward(subgraph.x_dict, subgraph.edge_index_dict)

            rerank_scores = torch.matmul(q_vec, sub_emb.T).squeeze(0)
            coarse_in_sub = torch.zeros(rerank_scores.size(0), device=device)
            for orig_rank, orig_idx in enumerate(candidate_indices):
                new_idx = table_mapping.get(orig_idx, -1)
                if 0 <= new_idx < coarse_in_sub.size(0):
                    coarse_in_sub[new_idx] = top_k_scores[orig_rank]

            final_scores = EVAL_ALPHA * coarse_in_sub + (1 - EVAL_ALPHA) * rerank_scores
            _, reranked = torch.topk(final_scores, k=min(EVAL_TOP_K, final_scores.size(0)))

            new_to_old = {v: k for k, v in table_mapping.items()}
            retrieved = [idx_to_id.get(new_to_old.get(idx.item(), -1), "") for idx in reranked]

            qa_r1 += full_recall_at_k(retrieved, gt, 1)
            qa_r5 += full_recall_at_k(retrieved, gt, 5)
            qa_r10 += full_recall_at_k(retrieved, gt, 10)
            qa_mrr += reciprocal_rank(retrieved, gt)
            qa_ndcg10 += ndcg_at_k(retrieved, gt, 10)

    del qa_model, data_full, data_filtered, query_vecs
    torch.cuda.empty_cache()

    results = {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'eval_count': eval_count,
        'total_queries': total,
        'best_edges': best_edges,
        'TWIG_R@1': twig_r1 / eval_count,
        'TWIG_R@5': twig_r5 / eval_count,
        'TWIG_R@10': twig_r10 / eval_count,
        'TWIG_MRR': twig_mrr / eval_count,
        'QA_R@1': qa_r1 / eval_count,
        'QA_R@5': qa_r5 / eval_count,
        'QA_R@10': qa_r10 / eval_count,
        'QA_MRR': qa_mrr / eval_count,
        'QA_nDCG@10': qa_ndcg10 / eval_count,
    }

    is_self = train_dataset == test_dataset
    label = "(self)" if is_self else "(cross)"
    print(f"    {label} TWIG R@10={results['TWIG_R@10']:.4f}  QA R@10={results['QA_R@10']:.4f}  "
          f"MRR={results['QA_MRR']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Dataset Generalization Evaluation for QA v2')

    parser.add_argument('--train-datasets', nargs='+', default=None,
                        help='Datasets whose trained models to use. Default: all')
    parser.add_argument('--test-datasets', nargs='+', default=None,
                        help='Datasets to test on. Default: all')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--best-config', type=str, default=None,
                        help='JSON dict mapping dataset -> ablation code (int). '
                             'E.g. \'{"feta": 18, "ottqa": 2}\'')
    parser.add_argument('--use-ablation-results', action='store_true', default=True,
                        help='Auto-load best edge config from QA ablation results (default: True)')

    args = parser.parse_args()

    train_datasets = args.train_datasets or ALL_DATASETS
    test_datasets = args.test_datasets or ALL_DATASETS

    print(f"Using GPU: {args.gpu}")
    print(f"Train datasets (model source): {train_datasets}")
    print(f"Test datasets (evaluation target): {test_datasets}")
    print(f"Total pairs: {len(train_datasets) * len(test_datasets)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Determine best edge configs
    best_edge_configs = dict(DEFAULT_BEST_EDGE_CONFIGS)

    if args.best_config:
        manual = json.loads(args.best_config)
        for ds, code in manual.items():
            best_edge_configs[ds] = binary_to_edges(int(code))
            print(f"  Manual config for {ds}: A{code} = {best_edge_configs[ds]}")
    elif args.use_ablation_results:
        print("\nLoading best configs from QA ablation results...")
        for ds in set(train_datasets + test_datasets):
            ablation_edges = load_best_config_from_ablation(ds)
            if ablation_edges is not None:
                best_edge_configs[ds] = ablation_edges

    print("\nBest edge configurations:")
    for ds in sorted(best_edge_configs.keys()):
        print(f"  {ds}: {best_edge_configs[ds]}")

    # Load embedder
    print("\nLoading sentence embedder...")
    embedder = SentenceTransformer(MODEL_NAME, device=str(device))

    # Run all pairs
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    all_results = []

    for train_ds in train_datasets:
        edges = best_edge_configs.get(train_ds, [])
        for test_ds in test_datasets:
            try:
                result = evaluate_cross(train_ds, test_ds, edges, device, embedder)
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                print(f"    ERROR {train_ds}->{test_ds}: {e}")
                import traceback
                traceback.print_exc()

    del embedder
    torch.cuda.empty_cache()

    if not all_results:
        print("\nNo results to report.")
        return

    # ── Summary Table ──
    print(f"\n{'='*100}")
    print(f"  CROSS-DATASET GENERALIZATION RESULTS (QA R@10)")
    print(f"{'='*100}")

    # Build matrix
    train_test_label = 'Train \\ Test'
    header = f"{train_test_label:>12}"
    for test_ds in test_datasets:
        header += f" | {test_ds:>8}"
    print(header)
    print("-" * len(header))

    results_lookup = {}
    for r in all_results:
        results_lookup[(r['train_dataset'], r['test_dataset'])] = r

    for train_ds in train_datasets:
        row = f"{train_ds:>12}"
        for test_ds in test_datasets:
            r = results_lookup.get((train_ds, test_ds))
            if r is not None:
                val = r['QA_R@10']
                marker = "*" if train_ds == test_ds else " "
                row += f" | {val:>7.4f}{marker}"
            else:
                row += f" |     N/A "
        print(row)

    print(f"\n  * = self-evaluation (same train and test dataset)")

    # TWIG baseline matrix
    print(f"\n{'='*100}")
    print(f"  CROSS-DATASET GENERALIZATION RESULTS (TWIG R@10 baseline)")
    print(f"{'='*100}")
    print(header)
    print("-" * len(header))

    for train_ds in train_datasets:
        row = f"{train_ds:>12}"
        for test_ds in test_datasets:
            r = results_lookup.get((train_ds, test_ds))
            if r is not None:
                val = r['TWIG_R@10']
                marker = "*" if train_ds == test_ds else " "
                row += f" | {val:>7.4f}{marker}"
            else:
                row += f" |     N/A "
        print(row)

    # Save results
    # JSON
    output_json = Path(RESULTS_DIR) / "cross_dataset_results.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Cross-Dataset Generalization Evaluation',
            'timestamp': datetime.now().isoformat(),
            'best_edge_configs': {k: v for k, v in best_edge_configs.items()},
            'results': all_results,
        }, f, ensure_ascii=False, indent=2)

    # CSV
    output_csv = Path(RESULTS_DIR) / "cross_dataset_results.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['train_dataset', 'test_dataset', 'is_self',
                      'QA_R@1', 'QA_R@5', 'QA_R@10', 'QA_MRR', 'QA_nDCG@10',
                      'TWIG_R@1', 'TWIG_R@5', 'TWIG_R@10', 'TWIG_MRR',
                      'eval_count']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            r['is_self'] = r['train_dataset'] == r['test_dataset']
            writer.writerow(r)

    # Summary statistics
    self_results = [r for r in all_results if r['train_dataset'] == r['test_dataset']]
    cross_results = [r for r in all_results if r['train_dataset'] != r['test_dataset']]

    if self_results:
        avg_self = sum(r['QA_R@10'] for r in self_results) / len(self_results)
        print(f"\n  Avg self R@10: {avg_self:.4f} ({len(self_results)} pairs)")
    if cross_results:
        avg_cross = sum(r['QA_R@10'] for r in cross_results) / len(cross_results)
        print(f"  Avg cross R@10: {avg_cross:.4f} ({len(cross_results)} pairs)")
        if self_results:
            drop = avg_self - avg_cross
            print(f"  Generalization gap: {drop:.4f}")

    print(f"\n  Results saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
