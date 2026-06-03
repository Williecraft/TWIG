#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融實驗結果摘要

讀取 results/qa_edge_ablation/{dataset}/results.json，
以 dev set QA_R@10 選出每個 dataset 的最佳邊配置，
輸出 test set Stage1（TWIG 粗排）和 Stage2（QA rerank）完整指標表格。

R@1, R@5, R@10, R@50；MMQA 從 R@2 開始。

Usage:
  cd reports/src && python summarize_ablation.py
"""

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
ABLATION_DIR = PROJECT_DIR / 'results' / 'qa_edge_ablation'

DATASETS = ['feta', 'ottqa', 'mimo_en', 'mimo_ch', 'e2ewtq', 'mmqa']

# MMQA 從 R@2 開始（reviewer 要求）
MMQA_START_R2 = True


def load_best_config(dataset):
    """讀取 results.json，以 dev QA_R@10 選最佳 config，回傳該 config 的 test 分數。"""
    result_file = ABLATION_DIR / dataset / 'results.json'
    if not result_file.exists():
        print(f"  [{dataset}] results.json not found")
        return None

    with open(result_file) as f:
        data = json.load(f)

    results = [r for r in data.get('results', []) if not r.get('skipped')]
    if not results:
        print(f"  [{dataset}] no valid results")
        return None

    # 選最佳：以 dev QA_R@10（即 QA_R@10 欄位，在修正後的 ablation 裡是 dev 分數）
    best = max(results, key=lambda r: r.get('QA_R@10', 0))

    # test 分數存在 test_* 欄位（修正後的 ablation）
    # 若沒有 test_* 欄位（舊格式），退回用 QA_* 欄位（test，舊版本的洩漏版）
    has_test = 'test_QA_R@10' in best

    def get(r, key):
        test_key = f'test_{key}'
        if has_test and test_key in r:
            return r[test_key]
        return r.get(key, None)

    return {
        'dataset': dataset,
        'config': best['config'],
        'binary': best['binary'],
        'label': best['label'],
        'edges': best.get('kept_edges', ''),
        'dev_QA_R@10': best.get('QA_R@10'),        # dev score（選邊依據）
        'eval_source': 'dev+test' if has_test else 'test_only (old, leakage)',
        # Stage 1 (TWIG coarse)
        'S1_R@1':  get(best, 'TWIG_R@1'),
        'S1_R@2':  get(best, 'TWIG_R@2'),
        'S1_R@5':  get(best, 'TWIG_R@5'),
        'S1_R@10': get(best, 'TWIG_R@10'),
        'S1_R@50': get(best, 'TWIG_R@50'),
        'S1_MRR':  get(best, 'TWIG_MRR'),
        # Stage 2 (QA rerank)
        'S2_R@1':  get(best, 'QA_R@1'),
        'S2_R@2':  get(best, 'QA_R@2'),
        'S2_R@5':  get(best, 'QA_R@5'),
        'S2_R@10': get(best, 'QA_R@10') if not has_test else get(best, 'test_QA_R@10'),
        'S2_R@50': get(best, 'QA_R@50'),
        'S2_MRR':  get(best, 'QA_MRR'),
    }


def fmt(v):
    if v is None: return '  N/A '
    return f'{v:.4f}'


def print_table(rows):
    # Header
    print(f"\n{'='*100}")
    print('QA Edge Ablation — Best Config per Dataset (dev-set selection, alpha=0)')
    print(f'{"="*100}')

    # Stage 1
    print(f"\n【Stage 1 — TWIG Coarse Ranking】")
    print(f"{'Dataset':<10} | {'Config':<6} | {'Edges':<50} | {'R@1':>6} | {'R@5':>6} | {'R@10':>6} | {'R@50':>6} | {'MRR':>6}")
    print(f"{'─'*10}-+-{'─'*6}-+-{'─'*50}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}")
    for r in rows:
        if r is None: continue
        ds = r['dataset']
        # MMQA: R@2 instead of R@1
        r1_col = fmt(r['S1_R@2']) if (ds == 'mmqa' and MMQA_START_R2) else fmt(r['S1_R@1'])
        r1_label = 'R@2 ' if (ds == 'mmqa' and MMQA_START_R2) else 'R@1 '
        print(f"{ds:<10} | {r['config']:<6} | {r['edges'][:50]:<50} | "
              f"{r1_col} | {fmt(r['S1_R@5'])} | {fmt(r['S1_R@10'])} | {fmt(r['S1_R@50'])} | {fmt(r['S1_MRR'])}")

    # Stage 2
    print(f"\n【Stage 2 — QA Reranking (test set)】")
    print(f"{'Dataset':<10} | {'Config':<6} | {'dev R@10':>8} | {'R@1':>6} | {'R@5':>6} | {'R@10':>6} | {'R@50':>6} | {'MRR':>6} | Source")
    print(f"{'─'*10}-+-{'─'*6}-+-{'─'*8}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}-+-{'─'*10}")
    for r in rows:
        if r is None: continue
        ds = r['dataset']
        r1_col = fmt(r['S2_R@2']) if (ds == 'mmqa' and MMQA_START_R2) else fmt(r['S2_R@1'])
        dev_r10 = fmt(r['dev_QA_R@10'])
        print(f"{ds:<10} | {r['config']:<6} | {dev_r10} | "
              f"{r1_col} | {fmt(r['S2_R@5'])} | {fmt(r['S2_R@10'])} | {fmt(r['S2_R@50'])} | {fmt(r['S2_MRR'])} | {r['eval_source']}")

    # Best configs summary
    print(f"\n【Best Edge Configs（供 BEST_EDGE_CONFIGS 使用）】")
    print("QA_BEST_EDGE_CONFIGS = {")
    for r in rows:
        if r is None: continue
        edges_list = [e.strip() for e in r['edges'].split(',') if e.strip() and e.strip() != 'none']
        print(f"    \"{r['dataset']}\": {edges_list},  # {r['config']} | dev R@10={fmt(r['dev_QA_R@10'])}")
    print("}")


def main():
    print("Loading ablation results...")
    rows = []
    for ds in DATASETS:
        r = load_best_config(ds)
        rows.append(r)
        if r:
            n_done = len([x for x in (ABLATION_DIR / ds / 'results.json').open().__iter__()]) if (ABLATION_DIR / ds / 'results.json').exists() else 0
            src = r['eval_source']
            print(f"  {ds}: best={r['config']} dev_R@10={fmt(r['dev_QA_R@10'])}  [{src}]")
        else:
            print(f"  {ds}: not available yet")

    print_table(rows)


if __name__ == '__main__':
    main()
