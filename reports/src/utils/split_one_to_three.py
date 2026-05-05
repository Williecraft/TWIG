#!/usr/bin/env python3
"""
Split dataset into train/dev/test = 8:1:1.

規則：
  - 以 query 為單位切分
  - 同一張 table 可以出現在多個 split（作為候選池）
  - 每個 split 的 table 和 query 都從 0 重新編號
  - 使用深拷貝避免跨 split 污染
"""

import json
import copy
import random
from pathlib import Path

# ========= 可調參數 =========
DATASET = "mimo_ch"

SEED = 42
SOURCE_DIR = Path(f"/user_data/TabGNN/data/downloads/{DATASET}")
OUTPUT_BASE = Path("/user_data/TabGNN/data/table")
DATASET_NAME = DATASET
SPLIT_RATIO = (0.8, 0.1, 0.1)  # train, dev, test
# ===========================


def read_jsonl(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def write_jsonl(items, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_split(name: str, queries_orig, table_map_orig):
    """將一個 split 的 query/table 深拷貝後重新編號並儲存"""
    queries = copy.deepcopy(queries_orig)

    # 收集此 split 引用的所有 table id
    referenced_ids = set()
    for q in queries:
        for gt in q.get("ground_truth_list", []):
            referenced_ids.add(gt["id"])

    # 建立 old_id -> new_id 映射，並深拷貝 table
    tables = []
    old_to_new = {}
    for old_id in sorted(referenced_ids):
        if old_id not in table_map_orig:
            continue
        new_id = len(tables)
        old_to_new[old_id] = new_id
        t = copy.deepcopy(table_map_orig[old_id])
        t["id"] = new_id
        tables.append(t)

    # 更新 query 的 id 和 ground_truth_list
    for i, q in enumerate(queries):
        q["id"] = i
        new_gt = []
        for gt in q.get("ground_truth_list", []):
            if gt["id"] in old_to_new:
                gt["id"] = old_to_new[gt["id"]]
                new_gt.append(gt)
        q["ground_truth_list"] = new_gt

    out_dir = OUTPUT_BASE / name / DATASET_NAME
    write_jsonl(queries, out_dir / "query.jsonl")
    write_jsonl(tables, out_dir / "table.jsonl")
    print(f"  {name}: {len(queries)} queries, {len(tables)} tables -> {out_dir}")


def main():
    random.seed(SEED)

    # 讀取資料
    queries = read_jsonl(SOURCE_DIR / "query.jsonl")
    tables = read_jsonl(SOURCE_DIR / "table.jsonl")
    table_map = {t["id"]: t for t in tables}

    print(f"來源: {SOURCE_DIR}")
    print(f"總 queries: {len(queries)}, 總 tables: {len(tables)}")

    # 隨機打亂並切分
    random.shuffle(queries)
    n = len(queries)
    n_train = int(n * SPLIT_RATIO[0])
    n_dev = int(n * SPLIT_RATIO[1])

    splits = {
        "train": queries[:n_train],
        "dev": queries[n_train:n_train + n_dev],
        "test": queries[n_train + n_dev:],
    }

    print(f"\n切分比例: {SPLIT_RATIO}")
    for name, qs in splits.items():
        print(f"  {name}: {len(qs)} queries")

    # 儲存各 split
    print("\n儲存中...")
    for name, qs in splits.items():
        save_split(name, qs, table_map)

    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
