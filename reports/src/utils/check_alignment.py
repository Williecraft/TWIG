"""檢查 query.jsonl 的 ground_truth 能否在 table.jsonl 中被找到"""
import json
from pathlib import Path

# ========= 可調參數 =========
DATASETS = [
    # "mimo_en",
    # "mimo_ch",
    # "mmqa",
    "ottqa",
    "feta",
    "e2ewtq",
]

# 要比對的欄位（可選: "id", "file_name", "sheet_name" 的任意組合）
KEY_FIELDS = ("file_name", "sheet_name")

BASE_DIR = Path("/user_data/TabGNN/data/downloads")
# ===========================


def make_key(item: dict) -> tuple:
    return tuple(item.get(f, "") for f in KEY_FIELDS)


def check_one(dataset: str):
    base = BASE_DIR / dataset
    table_path = base / "table.jsonl"
    query_path = base / "query.jsonl"

    if not table_path.exists() or not query_path.exists():
        print(f"  跳過：找不到 {base}")
        return

    # 建立 table 索引
    table_keys = set()
    with open(table_path, encoding="utf-8") as f:
        for line in f:
            table_keys.add(make_key(json.loads(line)))

    # 檢查 query
    total_queries = 0
    total_gt = 0
    missing_gt = 0
    missing_queries = 0
    missing_details = []

    with open(query_path, encoding="utf-8") as f:
        for line in f:
            total_queries += 1
            obj = json.loads(line)
            q = obj.get("question", "")
            gt_list = obj.get("ground_truth_list", []) or []
            query_has_miss = False

            for gt in gt_list:
                total_gt += 1
                key = make_key(gt)
                if key not in table_keys:
                    missing_gt += 1
                    query_has_miss = True
                    missing_details.append({
                        "query_id": obj.get("id"),
                        "question": q[:80],
                        "gt_key": key,
                    })

            if query_has_miss:
                missing_queries += 1

    print(f"  table.jsonl 表格數: {len(table_keys)}")
    print(f"  query.jsonl 查詢數: {total_queries}")
    print(f"  ground_truth 總數: {total_gt}")
    print(f"  找不到的 GT 數: {missing_gt} / {total_gt}")
    print(f"  有缺失 GT 的 query 數: {missing_queries} / {total_queries}")

    if missing_details:
        print(f"\n  缺失的 GT 詳情（共 {len(missing_details)} 筆）:")
        for d in missing_details[:15]:
            key_str = ", ".join(f"{f}={v}" for f, v in zip(KEY_FIELDS, d["gt_key"]))
            print(f"    Query {d['query_id']}: {d['question']}")
            print(f"      GT key: {key_str}")
        if len(missing_details) > 15:
            print(f"    ... 還有 {len(missing_details) - 15} 筆未顯示")
    else:
        print("  ✅ 所有 ground_truth 都能在 table.jsonl 中找到！")


def main():
    print(f"KEY_FIELDS = {KEY_FIELDS}\n")
    for ds in DATASETS:
        print(f"{'='*50}")
        print(f"[{ds}]")
        print(f"{'='*50}")
        check_one(ds)
        print()


if __name__ == "__main__":
    main()
