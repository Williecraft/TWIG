#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Query-Aware v2 完整 Pipeline

用法:
  python run_pipeline_v2.py                    # 全部資料集
  python run_pipeline_v2.py feta ottqa         # 指定資料集
  python run_pipeline_v2.py --eval-only feta   # 只評估（不訓練）
"""

import sys
from pathlib import Path


def main():
    eval_only = '--eval-only' in sys.argv
    datasets = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not datasets:
        datasets = ["feta", "ottqa", "mimo_en", "mimo_ch", "e2ewtq", "mmqa"]

    # 訓練
    if not eval_only:
        print("=" * 60)
        print("Phase 1: Training Query-Aware v2 Models")
        print("=" * 60)
        from train_query_aware_v2 import main as train_main
        for ds in datasets:
            graph_path = Path(f'/user_data/TabGNN/data/processed/train/{ds}/graph.pt')
            if not graph_path.exists():
                print(f"跳過 {ds}: 找不到訓練圖 {graph_path}")
                continue
            train_main(ds)

    # 評估
    print("\n" + "=" * 60)
    print("Phase 2: Evaluating Query-Aware v2 Models")
    print("=" * 60)
    from evaluate_query_aware_v2 import evaluate, RESULT_DIR, BEST_EDGE_CONFIGS
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)

    import json
    all_results = {}
    for ds in datasets:
        results = evaluate(ds, ds)
        if results is not None:
            all_results[ds] = results
            result_file = Path(RESULT_DIR) / f"{ds}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'source': ds,
                    'dataset': ds,
                    'best_edges': BEST_EDGE_CONFIGS.get(ds, []),
                    'results': results,
                }, f, ensure_ascii=False, indent=2)

    # 匯總
    if all_results:
        print(f"\n\n{'='*80}")
        print("Final Summary: Query-Aware v2 vs TWIG Baseline")
        print(f"{'='*80}")
        print(f"{'Dataset':>10} | {'E0 R@1':>7} | {'QA R@1':>7} | {'E0 R@10':>8} | {'QA R@10':>8} | {'ΔR@10':>8} | {'E0 MRR':>7} | {'QA MRR':>7}")
        print("-" * 80)
        for ds, r in all_results.items():
            dr10 = r['Recall@10'] - r['E0_Recall@10']
            a = '↑' if dr10 > 0 else '↓'
            print(f"{ds:>10} | {r['E0_Recall@1']:>7.4f} | {r['Recall@1']:>7.4f} | "
                  f"{r['E0_Recall@10']:>8.4f} | {r['Recall@10']:>8.4f} | "
                  f"{a}{abs(dr10):.4f} | {r['E0_MRR']:>7.4f} | {r['MRR@k']:>7.4f}")

        summary_file = Path(RESULT_DIR) / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n結果已儲存至 {RESULT_DIR}/")


if __name__ == '__main__':
    main()
