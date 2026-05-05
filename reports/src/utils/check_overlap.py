"""檢查每個 source 中 train/test/dev 之間是否有交集的 table 或 query"""
import json
from pathlib import Path
from itertools import combinations

SOURCES = [
    # "mimo_en",
    # "mimo_ch",
    # "mmqa",
    "ottqa",
    "feta",
    "e2ewtq",
]
CHECK_TABLE = False   # 是否檢查 table 重疊
CHECK_QUERY = True   # 是否檢查 query 重疊

SPLITS = ["train", "test", "dev"]
BASE_DIR = Path("/user_data/TabGNN/data/table")


def load_table_keys(path: Path) -> set:
    """讀取 table.jsonl，回傳 file_name|sheet_name 的 set"""
    keys = set()
    if not path.exists():
        return keys
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            fn = item.get("file_name", "")
            sn = item.get("sheet_name", "")
            keys.add(f"{fn}|{sn}")
    return keys


def load_query_keys(path: Path) -> set:
    """讀取 query.jsonl，回傳 question 文字的 set"""
    keys = set()
    if not path.exists():
        return keys
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            q = item.get("question") or (item.get("questions", [None]) or [None])[0]
            if q:
                keys.add(q.strip())
    return keys


def check_overlap(source: str):
    found_any = False

    # 載入各 split 的 table 和 query
    split_tables = {}
    split_queries = {}
    for split in SPLITS:
        split_dir = BASE_DIR / split / source
        split_tables[split] = load_table_keys(split_dir / "table.jsonl")
        split_queries[split] = load_query_keys(split_dir / "query.jsonl")

    # 檢查任兩個 split 的交集
    for a, b in combinations(SPLITS, 2):
        # Table 交集
        if CHECK_TABLE:
            table_overlap = split_tables[a] & split_tables[b]
            if table_overlap:
                found_any = True
                print(f"  ⚠ TABLE 交集 [{a} ∩ {b}]: {len(table_overlap)} 筆")
                for t in sorted(list(table_overlap)[:10]):
                    print(f"    - {t}")
            if len(table_overlap) > 10:
                print(f"    ... 還有 {len(table_overlap) - 10} 筆")

        # Query 交集
        if CHECK_QUERY:
            query_overlap = split_queries[a] & split_queries[b]
            if query_overlap:
                found_any = True
                print(f"  ⚠ QUERY 交集 [{a} ∩ {b}]: {len(query_overlap)} 筆")
                for q in sorted(list(query_overlap)[:5]):
                    print(f"    - {q[:80]}...")
            if len(query_overlap) > 5:
                print(f"    ... 還有 {len(query_overlap) - 5} 筆")

    if not found_any:
        print("  ✓ 無交集")

    return found_any


def main():
    print("=" * 60)
    print("檢查 train/test/dev 之間的 table 與 query 交集")
    print("=" * 60)

    any_overlap = False
    for source in SOURCES:
        print(f"\n{'─'*40}")
        print(f"Source: {source}")
        # 印出各 split 大小
        for split in SPLITS:
            t_path = BASE_DIR / split / source / "table.jsonl"
            q_path = BASE_DIR / split / source / "query.jsonl"
            t_count = sum(1 for _ in open(t_path)) if t_path.exists() else 0
            q_count = sum(1 for _ in open(q_path)) if q_path.exists() else 0
            print(f"  {split}: {t_count} tables, {q_count} queries")
        print(f"{'─'*40}")
        if check_overlap(source):
            any_overlap = True

    print(f"\n{'='*60}")
    if any_overlap:
        print("⚠ 有發現交集，請檢查上方輸出。")
    else:
        print("✓ 所有資料集 train/test/dev 之間無交集。")
    print("=" * 60)


if __name__ == "__main__":
    main()
