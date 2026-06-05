#!/usr/bin/env python3
"""
單 GPU 多 process 加速 ottqa（或任一 dataset）邊消融。

ottqa 每個 config ~70–80 分鐘，但單 process 只吃滿 ~25% GPU（rerank 建子圖
是 CPU-bound）。本 launcher 把某個 config 範圍內「尚未完成」的 configs 平均切
給 (GPU 數 × 每 GPU process 數) 個 slot，每個 slot 用獨立的
QA_ABLATION_RESULTS_DIR（以 shard_ 開頭，讓 merge_shards.py 的 glob('shard*')
直接吃到）背景啟動 run_qa_edge_ablation.py，互不干擾。

完成的 config 由掃描既有 shard*/<ds>/completed_configs.json 判定，因此可重複執行
（補跑只會跑還沒完成的）。

用法（機器1，範圍 0-21、每卡 3 process、用 GPU 0,1）：
  cd reports/src && python run_ottqa_parallel.py --range 0 21 --procs-per-gpu 3 --gpus 0,1
  # 預覽不啟動：加 --dry-run
  # 換 dataset：--dataset feta
  # 機器2 GPU0 顯存緊，可分別指定：--gpus 0 --procs-per-gpu 2 跑一次，再 --gpus 1 --procs-per-gpu 3
"""
import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
SHARDS_DIR = PROJECT_DIR / "results" / "qa_edge_ablation_shards"
LOGS_DIR = PROJECT_DIR / "logs"
RUNNER = Path(__file__).resolve().parent / "query_aware" / "run_qa_edge_ablation.py"


def completed_codes(dataset: str) -> set:
    """掃描所有 shard*/<ds>/completed_configs.json，回傳已完成的 config codes。"""
    done = set()
    for shard in SHARDS_DIR.glob("shard*"):
        f = shard / dataset / "completed_configs.json"
        if f.exists():
            try:
                done |= set(json.load(open(f)))
            except Exception:
                pass
    return done


def split_even(items, n):
    """把 items 盡量平均切成 n 段（連續分配）。"""
    k, m = divmod(len(items), n)
    out, i = [], 0
    for s in range(n):
        size = k + (1 if s < m else 0)
        out.append(items[i:i + size])
        i += size
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", nargs=2, type=int, required=True, metavar=("LO", "HI"),
                    help="本機負責的 config 範圍（含兩端），例如 0 21")
    ap.add_argument("--procs-per-gpu", type=int, default=3)
    ap.add_argument("--gpus", type=str, default="0,1", help="逗號分隔，例如 0,1")
    ap.add_argument("--dataset", type=str, default="ottqa")
    ap.add_argument("--omp-threads", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lo, hi = args.range
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    n_slots = len(gpus) * args.procs_per_gpu

    done = completed_codes(args.dataset)
    remaining = [c for c in range(lo, hi + 1) if c not in done]

    print(f"dataset={args.dataset} range={lo}-{hi} gpus={gpus} procs/gpu={args.procs_per_gpu}")
    print(f"已完成 {sorted(c for c in done if lo <= c <= hi)}")
    print(f"待跑 {remaining}  (共 {len(remaining)} 個, 切成 {n_slots} 個 slot)")
    if not remaining:
        print("沒有待跑的 config，結束。")
        return

    chunks = split_even(remaining, n_slots)
    host = socket.gethostname()[:8]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    slot = 0
    launched = []
    for gpu in gpus:
        for p in range(args.procs_per_gpu):
            codes = chunks[slot]
            slot += 1
            if not codes:
                continue
            tag = f"{host}_g{gpu}_p{p}"
            out_dir = SHARDS_DIR / f"shard_{tag}"
            log = LOGS_DIR / f"ottqa_{tag}.log"
            env = dict(os.environ)
            env["QA_ABLATION_RESULTS_DIR"] = str(out_dir)
            env["OMP_NUM_THREADS"] = str(args.omp_threads)
            env["PYTHONUNBUFFERED"] = "1"  # 即時 flush log（避免 block buffering 看不到進度）
            # 用 sys.executable 而非 "python"，確保子程序用「跑 launcher 的同一個
            # Python」（venv），不依賴有沒有 source activate / PATH。
            cmd = [sys.executable, str(RUNNER), "--datasets", args.dataset,
                   "--gpu", gpu, "--ablation", *[str(c) for c in codes]]
            print(f"  slot {slot-1}: gpu{gpu} p{p} -> {codes}")
            print(f"           QA_ABLATION_RESULTS_DIR={out_dir}")
            print(f"           log={log}")
            if args.dry_run:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            lf = open(log, "a")
            subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True,
                             cwd=str(RUNNER.parent))
            launched.append((tag, gpu, codes))

    if args.dry_run:
        print("\n[dry-run] 未啟動任何 process。")
    else:
        print(f"\n已啟動 {len(launched)} 個 process。檢查：")
        print("  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader")
        print("  pgrep -af run_qa_edge_ablation.py | wc -l")
        print(f"  tail -f {LOGS_DIR}/ottqa_*.log")


if __name__ == "__main__":
    main()
