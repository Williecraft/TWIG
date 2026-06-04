#!/usr/bin/env python3
"""
BGE-M3 dense-retrieval baseline（無圖、無 GNN）。

純 dense retrieval：用 bge-m3 直接編碼 table 文字與 query，cosine 相似度取 top-K。
table 文字組成與 build_graph.py 的 table node 完全一致（Page/Sheet/Section/
Columns/前5列），確保與 TWIG 可直接比較。指標定義（Recall@k = 找回的 gold 比例）
與 evaluate_retrieval.py 一致。

順便量測效率：
  - 線下：編碼整個 table 語料庫的時間（offline corpus encoding）
  - 線上：每個 query 的「編碼 + 檢索」延遲（ms/query，batch=1，realistic online）

用法：
  cd reports/src && python bgem3_baseline.py --gpu 1
  # 指定資料集： --datasets feta ottqa
輸出：reports/bgem3_baseline.md
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_NAME = "BAAI/bge-m3"
DATASETS = ["feta", "ottqa", "mimo_en", "mimo_ch", "e2ewtq", "mmqa"]


def get_key_fields(dataset):
    if dataset in ("ottqa", "feta", "e2ewtq"):
        return ("sheet_name", "file_name")
    return ("id",)


def make_key(item, key_fields):
    return "|".join(str(item.get(f, "")) for f in key_fields)


def build_table_docs(table_path, key_fields):
    """完全照 build_graph.py 的 table node 文字組成，回傳 (docs, ids)（去重、跳過無 header）。"""
    docs, ids, seen = [], [], set()
    with open(table_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                header_list = item.get("header")
                if not header_list:
                    continue
                instance_rows = [next(csv.reader([row_str.replace("\n", " ").replace("\r", " ")]))
                                 for row_str in item.get("instances", [])]
                table_id = make_key(item, key_fields)
                if table_id in seen:
                    continue
                seen.add(table_id)
                metadata = item.get("metadata", {}) or {}
                page_title = (metadata.get("table_page_title") or metadata.get("title")
                              or metadata.get("table_section_title") or item.get("file_name")
                              or f"__UNKNOWN_PAGE_{table_id}__")
                table_doc = " ".join(filter(None, [
                    f"Page: {page_title}",
                    f"Sheet: {item.get('sheet_name', '')}",
                    f"Section: {metadata.get('table_section_title', '')}",
                    f"Columns: {', '.join(header_list)}",
                    f"Data: {'; '.join([', '.join(row) for row in instance_rows[:5]])}",
                ]))
                docs.append(table_doc)
                ids.append(table_id)
            except Exception:
                continue
    return docs, ids


def parse_queries(query_path, mapping_keys, key_fields):
    """照 evaluate_retrieval.py：回傳 [(question, gt_key_set)]，gt 只保留能對齊到語料庫的。"""
    queries = []
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "questions" in obj:
                q = (obj.get("questions") or [""])[0].strip()
            elif "question" in obj:
                q = (obj.get("question") or "").strip()
            else:
                continue
            gt_keys = set()
            for gt in (obj.get("ground_truth_list", []) or []):
                if all(gt.get(fld) is not None for fld in key_fields):
                    k = make_key(gt, key_fields)
                    if k in mapping_keys:
                        gt_keys.add(k)
            if q:
                queries.append((q, gt_keys))
    return queries


def full_recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved, relevant):
    for i, rid in enumerate(retrieved, 1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def evaluate_dataset(dataset, embedder, split, batch_size):
    import torch
    import torch.nn.functional as F

    key_fields = get_key_fields(dataset)
    table_path = PROJECT_DIR / f"data/table/{split}/{dataset}/table.jsonl"
    query_path = PROJECT_DIR / f"data/table/{split}/{dataset}/query.jsonl"
    if not table_path.exists() or not query_path.exists():
        print(f"  [{dataset}] 缺檔，跳過")
        return None

    docs, ids = build_table_docs(table_path, key_fields)
    mapping_keys = set(ids)
    queries = parse_queries(query_path, mapping_keys, key_fields)
    questions = [q for q, _ in queries]
    relevants = [gt for _, gt in queries]
    eval_count = sum(1 for gt in relevants if gt)
    device = embedder.device

    # --- 線下：編碼整個 table 語料庫並計時 ---
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    table_emb = embedder.encode(docs, batch_size=batch_size, show_progress_bar=False,
                                convert_to_tensor=True, normalize_embeddings=True).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    offline_sec = time.perf_counter() - t0

    # --- 線上：逐 query（batch=1）量「編碼 + 檢索」延遲 ---
    table_emb_T = table_emb.T.contiguous()
    # warmup
    _ = embedder.encode([questions[0]], show_progress_bar=False, convert_to_tensor=True,
                        normalize_embeddings=True)
    latencies = []
    r1 = r5 = r10 = r50 = mrr = 0.0
    for qi, q in enumerate(questions):
        gt = relevants[qi]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        qv = embedder.encode([q], show_progress_bar=False, convert_to_tensor=True,
                             normalize_embeddings=True).to(device)
        scores = torch.matmul(qv, table_emb_T).squeeze(0)
        topk = torch.topk(scores, k=min(50, scores.size(0))).indices.tolist()
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if not gt:
            continue
        ret = [ids[i] for i in topk]
        r1 += full_recall_at_k(ret, gt, 1)
        r5 += full_recall_at_k(ret, gt, 5)
        r10 += full_recall_at_k(ret, gt, 10)
        r50 += full_recall_at_k(ret, gt, 50)
        mrr += reciprocal_rank(ret, gt)

    latencies.sort()
    ec = max(1, eval_count)
    return {
        "dataset": dataset, "n_tables": len(docs), "n_queries": len(questions),
        "eval_count": eval_count,
        "R@1": r1 / ec, "R@5": r5 / ec, "R@10": r10 / ec, "R@50": r50 / ec, "MRR": mrr / ec,
        "offline_encode_sec": offline_sec,
        "offline_tables_per_sec": len(docs) / offline_sec if offline_sec > 0 else 0.0,
        "online_ms_mean": sum(latencies) / len(latencies),
        "online_ms_median": latencies[len(latencies) // 2],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output", default=str(PROJECT_DIR / "reports" / "bgem3_baseline.md"))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"Device={device} ({gpu_name}) | model={MODEL_NAME} | batch_size={args.batch_size}")
    embedder = SentenceTransformer(MODEL_NAME, device=device)

    rows = []
    for ds in args.datasets:
        print(f"=== {ds} ===")
        r = evaluate_dataset(ds, embedder, args.split, args.batch_size)
        if r:
            rows.append(r)
            print(f"  tables={r['n_tables']} queries={r['n_queries']}(eval {r['eval_count']}) "
                  f"R@1={r['R@1']:.4f} R@5={r['R@5']:.4f} R@10={r['R@10']:.4f} "
                  f"| offline={r['offline_encode_sec']:.1f}s online={r['online_ms_mean']:.1f}ms")

    write_markdown(rows, args, gpu_name, device)
    print(f"\n寫入 {args.output}")


def write_markdown(rows, args, gpu_name, device):
    lines = []
    lines.append("# BGE-M3 Dense Retrieval Baseline（無圖 / 無 GNN）\n")
    lines.append(f"> 產生時間：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 設定\n")
    lines.append(f"- Embedding model：`{MODEL_NAME}`")
    lines.append(f"- 方法：純 dense retrieval（cosine top-K），**不建圖、不跑 GNN**")
    lines.append(f"- table 文字組成：與 `build_graph.py` table node 完全一致（Page / Sheet / "
                 f"Section / Columns / 前 5 列）")
    lines.append(f"- 指標：Recall@k = 找回的 gold 表格比例（同 `evaluate_retrieval.py`），"
                 f"在有 gold 的 query 上平均")
    lines.append(f"- 硬體：{gpu_name}（device={device}）")
    lines.append(f"- batch_size（線下編碼）：{args.batch_size}；線上延遲量測：batch=1")
    lines.append(f"- split：{args.split}\n")

    lines.append("## Table 3 用：檢索品質（BGE-M3 no-graph 列）\n")
    lines.append("| 資料集 | #tables | #queries(eval) | R@1 | R@5 | R@10 | R@50 | MRR |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['n_tables']} | {r['n_queries']}({r['eval_count']}) | "
                     f"{r['R@1']:.4f} | {r['R@5']:.4f} | {r['R@10']:.4f} | {r['R@50']:.4f} | {r['MRR']:.4f} |")

    lines.append("\n## Table 4 用：效率（BGE-M3 列）\n")
    lines.append("| 資料集 | #tables | 線下編碼語料庫(s) | 編碼吞吐(tables/s) | 線上 ms/query(mean) | 線上 ms/query(median) |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['n_tables']} | {r['offline_encode_sec']:.1f} | "
                     f"{r['offline_tables_per_sec']:.1f} | {r['online_ms_mean']:.1f} | {r['online_ms_median']:.1f} |")

    lines.append("\n> 線上延遲 = 單一 query 的「bge-m3 編碼 + 與全語料庫 cosine + top-50」端到端時間"
                 "（batch=1，含 GPU 同步）。線下 = 一次編碼整個 table 語料庫的時間。")
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
