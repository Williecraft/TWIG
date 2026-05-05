"""檢查每個 source 在 train/test/dev 同一資料集內是否有 file_name + sheet_name 重複"""
import json
from pathlib import Path
from collections import Counter

SOURCES = [
    # "mimo_en",
    # "mimo_ch",
    # "mmqa",
    "ottqa",
    "feta",
    "e2ewtq",
]

# 唯一鍵欄位（可選: "id", "file_name", "sheet_name" 的任意組合）
KEY_FIELDS = ("file_name", "sheet_name")

SPLITS = ["train", "test", "dev"]
BASE_DIR = Path("/user_data/TabGNN/data/table")


def check_duplicates(source: str):
    """檢查指定 source 在各 split 下的 table.jsonl 是否有重複的 file_name|sheet_name"""
    found_any = False

    for split in SPLITS:
        table_file = BASE_DIR / split / source / "table.jsonl"
        if not table_file.exists():
            continue

        keys = []
        with open(table_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                key = "|".join(str(item.get(f, "")) for f in KEY_FIELDS)
                keys.append(key)

        counter = Counter(keys)
        duplicates = {k: v for k, v in counter.items() if v > 1}

        if duplicates:
            found_any = True
            print(f"\n  [{split}/{source}] 共 {len(keys)} 筆，發現 {len(duplicates)} 組重複：")
            for key, count in sorted(duplicates.items(), key=lambda x: -x[1]):
                print(f"    {key}  x{count}")
        else:
            print(f"  [{split}/{source}] 共 {len(keys)} 筆，無重複 ✓")

    return found_any


def main():
    print("=" * 60)
    print(f"檢查 {KEY_FIELDS} 重複")
    print("=" * 60)

    any_dup = False
    for source in SOURCES:
        print(f"\n{'─'*40}")
        print(f"Source: {source}")
        print(f"{'─'*40}")
        if check_duplicates(source):
            any_dup = True

    print(f"\n{'='*60}")
    if any_dup:
        print("⚠ 有發現重複，請檢查上方輸出。")
    else:
        print("✓ 所有資料集均無重複。")
    print("=" * 60)


if __name__ == "__main__":
    main()
