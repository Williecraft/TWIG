#!/usr/bin/env python3
"""
合併 6 個 GPU worker 的 QA edge ablation 分片結果，並做覆蓋率檢查。

每個 worker 用 QA_ABLATION_RESULTS_DIR 寫到獨立目錄
results/qa_edge_ablation_shards/shard{0..5}/{dataset}/results.json
（避免並行 read-modify-write 互相覆蓋）。本腳本把同一 dataset 的
所有分片 union 起來，寫回 summarize_ablation.py 預期的標準路徑
results/qa_edge_ablation/{dataset}/results.json，並列出缺漏的 config。

合法跳過（該 dataset 缺某種邊，例如 e2ewtq 沒有 same_page）不算缺漏；
只有「該跑卻沒結果」（例如 OOM/crash）才標記為需補跑。

用法：
  cd reports/src && python merge_shards.py
  # 只看報告不寫檔： python merge_shards.py --dry-run
"""
import argparse
import csv
import json
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
SHARDS_DIR = PROJECT_DIR / "results" / "qa_edge_ablation_shards"
MERGED_DIR = PROJECT_DIR / "results" / "qa_edge_ablation"

DATASETS = ["e2ewtq", "feta", "ottqa", "mimo_en", "mimo_ch", "mmqa"]

# bit 5..0 -> tt tc tp sp cc sc（與 run_qa_edge_ablation.py 一致）
ALL_EDGE_RELATIONS = [
    "similar_table", "has_column", "comes_from",
    "same_page", "similar_content", "shared_column_name",
]


def edges_for_code(code: int):
    """code 的 6-bit 對應選了哪些邊（bit5=tt ... bit0=sc）。"""
    return {rel for i, rel in enumerate(ALL_EDGE_RELATIONS)
            if (code >> (5 - i)) & 1}


def available_edges(dataset: str):
    g = PROJECT_DIR / f"data/processed/train/{dataset}/graph.pt"
    if not g.exists():
        return None
    d = torch.load(str(g), weights_only=False)
    return {rel for _, rel, _ in d.edge_types if not rel.startswith("rev_")}


def runnable_codes(dataset: str):
    """該 dataset 在 0..63 中、所有選用邊都存在、因而應該跑出結果的 config。"""
    avail = available_edges(dataset)
    if avail is None:
        return None
    return {c for c in range(64) if edges_for_code(c) <= avail}


def load_shard_results(dataset: str):
    """union 所有 shard 的該 dataset 結果，回傳 {code: result_dict}。"""
    merged = {}
    for shard in sorted(SHARDS_DIR.glob("shard*")):
        rj = shard / dataset / "results.json"
        if not rj.exists():
            continue
        try:
            data = json.load(open(rj, encoding="utf-8"))
        except Exception as e:
            print(f"  [warn] 無法讀取 {rj}: {e}")
            continue
        for r in data.get("results", []):
            code = r.get("code")
            if code is None:
                continue
            # 同一 code 若出現在多個 shard（理論上不該），保留欄位較完整者
            if code not in merged or len(r) > len(merged[code]):
                merged[code] = r
    return merged


CSV_FIELDS = ["config", "code", "binary", "label", "num_edges", "kept_edges",
              "QA_R@1", "QA_R@5", "QA_R@10", "QA_MRR", "QA_nDCG@10",
              "TWIG_R@1", "TWIG_R@5", "TWIG_R@10", "TWIG_MRR",
              "twig_best_epoch", "qa_best_epoch", "elapsed_min"]


def write_merged(dataset: str, merged: dict):
    out_dir = MERGED_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [merged[c] for c in sorted(merged)]
    json.dump({"results": results}, open(out_dir / "results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    json.dump(sorted(merged), open(out_dir / "completed_configs.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只報告，不寫入合併結果")
    args = ap.parse_args()

    print(f"Shards dir : {SHARDS_DIR}")
    print(f"Merged dir : {MERGED_DIR}")
    print(f"{'='*70}")

    rerun = {}
    all_ok = True
    for ds in DATASETS:
        merged = load_shard_results(ds)
        runnable = runnable_codes(ds)
        present = set(merged)
        if runnable is None:
            print(f"{ds:8s} | 找不到 graph，跳過")
            continue
        missing = sorted(runnable - present)
        extra = sorted(present - runnable)  # 不該出現（例如 e2ewtq 卻有 sp config）
        skipped_legit = 64 - len(runnable)
        status = "OK" if not missing else f"缺 {len(missing)}"
        print(f"{ds:8s} | 應跑 {len(runnable):2d} | 有 {len(present):2d} | "
              f"合法跳過 {skipped_legit:2d} | {status}")
        if missing:
            all_ok = False
            rerun[ds] = missing
            print(f"           需補跑 configs: {missing}")
        if extra:
            print(f"           [warn] 多出非預期 configs（邊不該存在）: {extra}")
        if not args.dry_run:
            write_merged(ds, merged)

    print(f"{'='*70}")
    if args.dry_run:
        print("dry-run：未寫入任何檔案。")
    else:
        print(f"已寫入 {MERGED_DIR}/<dataset>/results.json")

    if all_ok:
        print("\n✅ 全部覆蓋完整，可直接跑 summarize_ablation.py")
    else:
        print("\n⚠️  有缺漏，補跑指令（挑空閒 GPU，--gpu 自行替換）：")
        for ds, codes in rerun.items():
            seq = " ".join(str(c) for c in codes)
            print(f"  QA_ABLATION_RESULTS_DIR={SHARDS_DIR}/rerun \\")
            print(f"    python query_aware/run_qa_edge_ablation.py --datasets {ds} --gpu 0 --ablation {seq}")
        print("  （補跑寫到 shards/rerun，再重跑一次 merge_shards.py 即可併入）")


if __name__ == "__main__":
    main()
