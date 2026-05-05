#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從 data.gov.tw 爬取開放資料表格，轉成 TabGNN 的 JSONL 格式。

用法:
    python scripts/crawl_datagov.py

輸出:
    data/table/train/datagov/table.jsonl
    data/table/test/datagov/table.jsonl
    data/table/dev/datagov/table.jsonl
"""

import csv
import io
import json
import os
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import requests

# ===========================
# 設定
# ===========================
PROJECT_DIR = Path(__file__).resolve().parent.parent
CATALOG_CACHE = PROJECT_DIR / "data" / "downloads" / "datagov" / "catalog.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "table"

TARGET_TOTAL = 5000           # 目標表格數
MAX_ROWS_PER_TABLE = 50       # 每張表最多保留幾列
MIN_ROWS = 3                  # 少於此列數的表格跳過
MIN_COLS = 2                  # 少於此欄位數的表格跳過
MAX_CSV_SIZE = 10 * 1024 * 1024  # 10MB，超過的 CSV 跳過
REQUEST_TIMEOUT = 30          # 秒
DELAY_BETWEEN_REQUESTS = 0.3  # 秒，避免被 ban

TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42
random.seed(SEED)

# ===========================
# 下載資料集目錄
# ===========================

def download_catalog():
    """下載 data.gov.tw 的完整資料集目錄 CSV"""
    CATALOG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CATALOG_CACHE.exists():
        print(f"目錄快取已存在: {CATALOG_CACHE}")
        return

    print("下載 data.gov.tw 資料集目錄...")
    url = "https://data.gov.tw/datasets/export/csv"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    CATALOG_CACHE.write_bytes(resp.content)
    print(f"已儲存目錄: {CATALOG_CACHE} ({len(resp.content) / 1024 / 1024:.1f} MB)")


def parse_catalog():
    """解析目錄 CSV，回傳按分類分組的資料集列表"""
    datasets_by_category = defaultdict(list)

    with open(CATALOG_CACHE, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        # 欄位: 資料集識別碼, 資料集名稱, 資料提供屬性, 服務分類, 品質檢測,
        #        檔案格式, 資料下載網址, 編碼格式, ...
        for row in reader:
            if len(row) < 10:
                continue
            dataset_id = row[0].strip()
            title = row[1].strip()
            category = row[3].strip()
            formats = row[5].strip()
            urls = row[6].strip()
            encoding = row[7].strip() if len(row) > 7 else ""
            field_desc = row[10].strip() if len(row) > 10 else ""

            if not dataset_id or not category:
                continue

            # 只保留含有 CSV 格式的資料集
            format_list = [f.strip().upper() for f in formats.split(";")]
            url_list = [u.strip() for u in urls.split(";")]

            csv_urls = []
            for fmt, url in zip(format_list, url_list):
                if fmt == "CSV" and url:
                    csv_urls.append(url)

            if not csv_urls:
                continue

            datasets_by_category[category].append({
                "id": dataset_id,
                "title": title,
                "category": category,
                "csv_urls": csv_urls,
                "encoding": encoding,
                "field_desc": field_desc,
            })

    return datasets_by_category


# ===========================
# 下載並解析 CSV
# ===========================

def try_decode(content_bytes, hint_encoding=""):
    """嘗試多種編碼解碼 CSV"""
    encodings = []
    if hint_encoding:
        enc = hint_encoding.lower().replace("-", "").replace("_", "")
        if "big5" in enc:
            encodings.append("big5")
        elif "utf8" in enc or "utf" in enc:
            encodings.append("utf-8-sig")
    encodings.extend(["utf-8-sig", "utf-8", "big5", "cp950", "latin-1"])

    for enc in encodings:
        try:
            return content_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def download_and_parse_csv(url, encoding_hint=""):
    """下載 CSV 並解析成 (header, rows) 格式"""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True,
                           headers={"User-Agent": "Mozilla/5.0 TabGNN-Research"})
        resp.raise_for_status()

        # 檢查大小
        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_CSV_SIZE:
            return None, None

        content = resp.content
        if len(content) > MAX_CSV_SIZE:
            return None, None

        text = try_decode(content, encoding_hint)
        if text is None:
            return None, None

        # 解析 CSV
        reader = csv.reader(io.StringIO(text))
        rows = []
        for row in reader:
            # 跳過完全空的列
            if any(cell.strip() for cell in row):
                rows.append(row)

        if len(rows) < 2:  # 至少要有 header + 1 row
            return None, None

        header = rows[0]
        data_rows = rows[1:]

        # 過濾
        if len(header) < MIN_COLS:
            return None, None
        if len(data_rows) < MIN_ROWS:
            return None, None

        return header, data_rows

    except Exception:
        return None, None


# ===========================
# 轉換成 TabGNN 格式
# ===========================

def make_table_entry(table_id, dataset_info, url, header, rows):
    """
    轉換成 TabGNN 的 JSONL 格式:
    {
        "id": int,
        "file_name": str,
        "sheet_name": str,
        "header": [comma-separated header string],
        "instances": [comma-separated row strings],
        "metadata": {
            "table_page_title": str,  # 資料集名稱
            "table_section_title": str,  # sheet_name
            "page_wikipedia_url": str,  # 原始 URL (改成 data.gov.tw URL)
        }
    }
    """
    # header: 合成一個逗號分隔的字串列表（與 feta 格式一致）
    header_str = ",".join(header)

    # instances: 每一列轉成逗號分隔字串
    truncated_rows = rows[:MAX_ROWS_PER_TABLE]
    instances = []
    for row in truncated_rows:
        # 確保每列的欄位數與 header 一致
        padded = row + [""] * max(0, len(header) - len(row))
        padded = padded[:len(header)]
        # 對含逗號的欄位加引號
        cells = []
        for cell in padded:
            cell = cell.strip()
            if "," in cell or '"' in cell:
                cell = '"' + cell.replace('"', '""') + '"'
            cells.append(cell)
        instances.append(",".join(cells))

    dataset_url = f"https://data.gov.tw/dataset/{dataset_info['id']}"

    entry = {
        "id": table_id,
        "file_name": f"datagov/{dataset_info['id']}.csv",
        "sheet_name": dataset_info["title"],
        "header": [header_str],
        "instances": instances,
        "metadata": {
            "table_page_title": dataset_info["category"],
            "table_section_title": dataset_info["title"],
            "page_url": dataset_url,
            "source": "data.gov.tw",
            "category": dataset_info["category"],
        }
    }
    return entry


# ===========================
# 主程式
# ===========================

def main():
    # 1. 下載並解析目錄
    download_catalog()
    datasets_by_category = parse_catalog()

    categories = sorted(datasets_by_category.keys())
    print(f"\n找到 {len(categories)} 個分類:")
    for cat in categories:
        print(f"  {cat}: {len(datasets_by_category[cat])} 個含 CSV 的資料集")

    total_datasets = sum(len(v) for v in datasets_by_category.values())
    print(f"共 {total_datasets} 個含 CSV 的資料集")

    # 2. 按分類均勻分配目標數量
    per_category_target = max(TARGET_TOTAL // len(categories), 50)
    print(f"\n每個分類目標: ~{per_category_target} 個表格")

    # 3. 對每個分類隨機取樣並下載
    all_tables = []
    table_id = 0
    progress_file = PROJECT_DIR / "data" / "downloads" / "datagov" / "progress.json"

    # 載入進度
    completed_ids = set()
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
            completed_ids = set(progress.get("completed", []))
            table_id = progress.get("next_id", 0)
            # 載入已完成的表格
            tables_file = PROJECT_DIR / "data" / "downloads" / "datagov" / "tables_partial.jsonl"
            if tables_file.exists():
                with open(tables_file) as tf:
                    for line in tf:
                        all_tables.append(json.loads(line))
        print(f"恢復進度: {len(all_tables)} 個表格已完成, {len(completed_ids)} 個資料集已處理")

    tables_file = PROJECT_DIR / "data" / "downloads" / "datagov" / "tables_partial.jsonl"
    tables_file.parent.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)  # 每分類已有多少表格
    for t in all_tables:
        stats[t["metadata"]["category"]] += 1

    for cat in categories:
        current_count = stats.get(cat, 0)
        if current_count >= per_category_target:
            print(f"\n[{cat}] 已有 {current_count} 個表格，跳過")
            continue

        datasets = datasets_by_category[cat]
        random.shuffle(datasets)

        remaining = per_category_target - current_count
        print(f"\n[{cat}] 需要再抓 {remaining} 個表格 (從 {len(datasets)} 個資料集)")

        success = 0
        fail = 0

        for ds in datasets:
            if success >= remaining:
                break
            if ds["id"] in completed_ids:
                continue

            for url in ds["csv_urls"]:
                if success >= remaining:
                    break

                header, rows = download_and_parse_csv(url, ds.get("encoding", ""))

                if header is None:
                    fail += 1
                    continue

                entry = make_table_entry(table_id, ds, url, header, rows)
                all_tables.append(entry)
                table_id += 1
                success += 1
                stats[cat] = stats.get(cat, 0) + 1

                # 即時寫入
                with open(tables_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                if success % 10 == 0:
                    print(f"  [{cat}] {current_count + success}/{per_category_target} "
                          f"(+{success} ok, {fail} fail)")

                time.sleep(DELAY_BETWEEN_REQUESTS)

            completed_ids.add(ds["id"])

            # 定期儲存進度
            if (success + fail) % 50 == 0:
                with open(progress_file, "w") as f:
                    json.dump({
                        "completed": list(completed_ids),
                        "next_id": table_id,
                    }, f)

        print(f"  [{cat}] 完成: +{success} ok, {fail} fail, 共 {stats.get(cat, 0)} 個表格")

        # 儲存進度
        with open(progress_file, "w") as f:
            json.dump({
                "completed": list(completed_ids),
                "next_id": table_id,
            }, f)

    # 4. 統計
    print(f"\n{'='*60}")
    print(f"共收集 {len(all_tables)} 個表格")
    print(f"各分類數量:")
    for cat in sorted(stats.keys()):
        print(f"  {cat}: {stats[cat]}")

    # 5. 分割 train/dev/test
    random.shuffle(all_tables)

    # 重新分配 ID
    for i, t in enumerate(all_tables):
        t["id"] = i

    n = len(all_tables)
    n_train = int(n * TRAIN_RATIO)
    n_dev = int(n * DEV_RATIO)

    train_tables = all_tables[:n_train]
    dev_tables = all_tables[n_train:n_train + n_dev]
    test_tables = all_tables[n_train + n_dev:]

    print(f"\n分割: train={len(train_tables)}, dev={len(dev_tables)}, test={len(test_tables)}")

    # 6. 寫入 JSONL
    for split, tables in [("train", train_tables), ("dev", dev_tables), ("test", test_tables)]:
        out_dir = OUTPUT_DIR / split / "datagov"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "table.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for t in tables:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"  寫入 {out_file} ({len(tables)} 筆)")

    # 清理暫存
    if tables_file.exists():
        tables_file.unlink()
    if progress_file.exists():
        progress_file.unlink()

    print("\n完成！")


if __name__ == "__main__":
    main()
