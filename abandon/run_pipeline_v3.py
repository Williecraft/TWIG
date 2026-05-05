#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Query-Aware v3 Complete Pipeline

Phase 1: Retrain TWIG with best edge config per dataset (saves model_best_edges.pt)
Phase 2: Frozen-base QA fine-tuning (saves model_qa_v3.pt)
Phase 3: Evaluate and compare against TWIG ablation baselines

Usage:
  cd src && python query_aware/run_pipeline_v3.py                     # all datasets
  cd src && python query_aware/run_pipeline_v3.py feta ottqa          # specific datasets
  cd src && python query_aware/run_pipeline_v3.py --skip-twig feta    # skip TWIG retraining
  cd src && python query_aware/run_pipeline_v3.py --eval-only feta    # evaluation only
"""

import sys
from pathlib import Path

PROJECT_DIR = "/user_data/TabGNN"


def main():
    skip_twig = '--skip-twig' in sys.argv
    eval_only = '--eval-only' in sys.argv
    datasets = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not datasets:
        datasets = ["feta", "ottqa", "mimo_en", "mimo_ch", "e2ewtq", "mmqa"]

    # Phase 1: Retrain TWIG with best edge configs
    if not eval_only and not skip_twig:
        print("=" * 60)
        print("Phase 1: Training TWIG with Best Edge Configs")
        print("=" * 60)
        from train_twig_best import train_twig_best
        for ds in datasets:
            graph_path = Path(f'{PROJECT_DIR}/data/processed/train/{ds}/graph.pt')
            if not graph_path.exists():
                print(f"  Skip {ds}: train graph not found")
                continue
            train_twig_best(ds)

    # Phase 2: QA fine-tuning with frozen base
    if not eval_only:
        print("\n" + "=" * 60)
        print("Phase 2: Query-Aware v3 Training (Frozen Base)")
        print("=" * 60)
        from train_query_aware_v3 import main as qa_train_main
        for ds in datasets:
            pretrained = Path(f'{PROJECT_DIR}/checkpoints/{ds}/model_best_edges.pt')
            if not pretrained.exists():
                print(f"  Skip {ds}: model_best_edges.pt not found")
                continue
            qa_train_main(ds)

    # Phase 3: Evaluate
    print("\n" + "=" * 60)
    print("Phase 3: Evaluation")
    print("=" * 60)
    from evaluate_query_aware_v3 import evaluate, RESULT_DIR
    import json

    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    all_results = {}

    for ds in datasets:
        results = evaluate(ds, ds)
        if results is not None:
            all_results[ds] = results
            result_file = Path(RESULT_DIR) / f"{ds}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # Final summary
    if all_results:
        print(f"\n\n{'='*90}")
        print("FINAL: QA v3 vs TWIG Ablation Baselines")
        print(f"{'='*90}")
        print(f"{'Dataset':>10} | {'TWIG R@1':>8} | {'QA R@1':>7} | "
              f"{'TWIG R@10':>9} | {'QA R@10':>8} | {'Delta':>7} | {'alpha':>5}")
        print("-" * 90)

        for ds, r in all_results.items():
            bl = r.get('twig_ablation_baseline', {})
            qa = r.get('qa_v3', {})
            if not bl:
                continue
            dr10 = qa.get('R@10', 0) - bl.get('R@10', 0)
            tag = '+' if dr10 > 0 else ''
            print(f"{ds:>10} | {bl.get('R@1',0):>8.4f} | {qa.get('R@1',0):>7.4f} | "
                  f"{bl.get('R@10',0):>9.4f} | {qa.get('R@10',0):>8.4f} | "
                  f"{tag}{dr10:.4f} | {r.get('best_alpha',0):.1f}")

        summary_file = Path(RESULT_DIR) / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {RESULT_DIR}/")


if __name__ == '__main__':
    main()
