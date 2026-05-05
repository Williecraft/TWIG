"""去除 train/test/dev 之間重疊的 query（保留優先序：test > dev > train）

策略：
  - 以 question 文字為 key 判斷重疊
  - 優先保留 test，其次 dev，最後 train
  - 修改後直接覆寫原始 query.jsonl
"""
import json
from pathlib import Path

SOURCES = [
    "mimo_en",
    "mimo_ch",
    "ottqa",
    "feta",
    "e2ewtq",
    "mmqa",
]

# 保留優先序：越前面越優先（重疊時保留此 split 的 query）
PRIORITY = ["test", "dev", "train"]
BASE_DIR = Path("/user_data/TabGNN/data/table")


def get_question(item: dict) -> str:
    """從 query item 提取 question 文字"""
    q = item.get("question") or (item.get("questions", [None]) or [None])[0]
    return q.strip() if q else ""


def dedup_queries(source: str):
    print(f"\n{'─'*50}")
    print(f"Source: {source}")
    print(f"{'─'*50}")

    # 載入各 split 的 queries
    split_queries = {}
    for split in PRIORITY:
        query_file = BASE_DIR / split / source / "query.jsonl"
        if not query_file.exists():
            split_queries[split] = []
            continue
        with open(query_file, "r", encoding="utf-8") as f:
            split_queries[split] = [json.loads(line) for line in f]
        print(f"  {split}: {len(split_queries[split])} queries")

    # 按優先序收集已見 question，低優先的重疊 query 會被移除
    seen_questions = set()
    removed_total = 0

    for split in PRIORITY:
        original_count = len(split_queries[split])
        deduped = []
        removed = 0

        for item in split_queries[split]:
            q = get_question(item)
            if not q:
                deduped.append(item)  # 沒有 question 的保留
                continue
            if q in seen_questions:
                removed += 1
            else:
                seen_questions.add(q)
                deduped.append(item)

        if removed > 0:
            print(f"  ⚠ {split}: 移除 {removed} 筆重疊 query ({original_count} → {len(deduped)})")
            # 覆寫檔案
            query_file = BASE_DIR / split / source / "query.jsonl"
            with open(query_file, "w", encoding="utf-8") as f:
                for item in deduped:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"    已更新: {query_file}")
        else:
            print(f"  ✓ {split}: 無重疊")

        removed_total += removed

    return removed_total


def main():
    print("=" * 60)
    print("去除 train/test/dev 之間重疊的 query")
    print(f"保留優先序: {' > '.join(PRIORITY)}")
    print("=" * 60)

    total_removed = 0
    for source in SOURCES:
        total_removed += dedup_queries(source)

    print(f"\n{'='*60}")
    if total_removed > 0:
        print(f"共移除 {total_removed} 筆重疊 query。")
    else:
        print("✓ 所有資料集均無重疊 query。")
    print("=" * 60)


if __name__ == "__main__":
    main()
