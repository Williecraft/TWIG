import json
import shutil
import pandas as pd
from pathlib import Path


# =========================
# 全域設定
# =========================
BASE = Path("/user_data/TabGNN/results/edge_ablation_extended")   # <-- 改成你的根目錄
DATASETS = [
    # "mimo_en",
    # "mimo_ch",
    # "feta",
    # "ottqa",
    # "mmqa",
    "e2ewtq",
]


# =========================
# binary 正規化：轉字串 + 補齊 6 位
# =========================
def norm_binary(x) -> str:
    # x 可能是 int / float / str / NaN
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    s = str(x).strip()

    # 有些情況會變成 "101.0"（float），先處理掉
    if s.endswith(".0"):
        s = s[:-2]

    # 去掉可能的引號
    s = s.strip('"').strip("'")

    # 只保留 0/1（如果有怪字元）
    s = "".join(ch for ch in s if ch in "01")

    # 補齊 6 位
    return s.zfill(6) if s else ""


# =========================
# 刪除條件：包含 tp 或 sp
# edge_order: tt tc tp sp cc sc (bit 5 to 0)
# index:        0  1  2  3  4  5
# =========================
def has_tp_or_sp(binary_like) -> bool:
    b = norm_binary(binary_like)
    if len(b) != 6:
        return False
    return (b[2] == "1") or (b[3] == "1")


def process_dataset(dataset: str):
    print(f"\n=== Processing {dataset} ===")

    dataset_path = BASE / dataset
    csv_file = dataset_path / "results.csv"
    json_file = dataset_path / "results.json"

    csv_backup = dataset_path / "results_backup.csv"
    json_backup = dataset_path / "results_backup.json"

    if not csv_file.exists() or not json_file.exists():
        print(f"[WARNING] {dataset}: 缺少 result.csv 或 result.json，跳過")
        return

    # ---------- CSV ----------
    # 強制把 binary 當字串讀入，避免 000101 -> 101
    df = pd.read_csv(csv_file, dtype={"binary": "string"})

    if "binary" not in df.columns:
        print(f"[WARNING] {dataset}: CSV 沒有 binary 欄位，跳過 CSV")
    else:
        original_len_csv = len(df)

        # 先正規化 binary 欄位（保險起見）
        df["binary"] = df["binary"].apply(norm_binary)

        df_filtered = df[~df["binary"].apply(has_tp_or_sp)]
        removed_csv = original_len_csv - len(df_filtered)

        shutil.copy2(csv_file, csv_backup)
        df_filtered.to_csv(csv_file, index=False)

        print(f"CSV: removed {removed_csv} rows (kept {len(df_filtered)}/{original_len_csv})")

    # ---------- JSON ----------
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    original_len_json = len(results)

    filtered_results = []
    for item in results:
        b = item.get("binary", "")
        if not has_tp_or_sp(b):
            # 順便把 binary 正規化（確保是 6 位字串）
            item["binary"] = norm_binary(b)
            filtered_results.append(item)

    removed_json = original_len_json - len(filtered_results)

    shutil.copy2(json_file, json_backup)
    data["results"] = filtered_results

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"JSON: removed {removed_json} rows (kept {len(filtered_results)}/{original_len_json})")


def main():
    print("Start batch processing...")

    for dataset in DATASETS:
        process_dataset(dataset)

    print("\nAll datasets processed.")


if __name__ == "__main__":
    main()
