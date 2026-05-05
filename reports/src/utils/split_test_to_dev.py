#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""將 test 集的數據切一半到 dev 集"""

import json
import os
import random
from pathlib import Path

# 設定隨機種子以確保可重複性
SEED = 42
random.seed(SEED)

# 數據路徑
TEST_DIR = Path("/user_data/TabGNN/data/table/test")
DEV_DIR = Path("/user_data/TabGNN/data/table/dev")

def split_dataset(dataset_name: str):
    """將指定數據集從 test 切一半到 dev"""
    
    test_dataset_dir = TEST_DIR / dataset_name
    dev_dataset_dir = DEV_DIR / dataset_name
    
    # 檢查 test 目錄是否存在
    if not test_dataset_dir.exists():
        print(f"跳過 {dataset_name}：test 目錄不存在")
        return
    
    table_file = test_dataset_dir / "table.jsonl"
    query_file = test_dataset_dir / "query.jsonl"
    
    # 檢查文件是否存在
    if not table_file.exists() or not query_file.exists():
        print(f"跳過 {dataset_name}：缺少 table.jsonl 或 query.jsonl")
        return
    
    print(f"\n處理 {dataset_name}...")
    
    # 讀取 query.jsonl
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = [json.loads(line) for line in f]
    
    original_query_count = len(queries)
    print(f"  原始 query 數量: {original_query_count}")
    
    # 隨機打亂並切分
    random.shuffle(queries)
    mid_point = len(queries) // 2
    
    dev_queries = queries[:mid_point]
    test_queries = queries[mid_point:]
    
    print(f"  dev query 數量: {len(dev_queries)}")
    print(f"  test query 數量: {len(test_queries)}")
    
    # 收集 dev 和 test 需要的 table keys (file_name + sheet_name)
    dev_table_keys = set()
    test_table_keys = set()
    
    for q in dev_queries:
        gt_list = q.get('ground_truth_list', []) or []
        for gt in gt_list:
            fn = gt.get('file_name')
            sn = gt.get('sheet_name')
            if fn is not None and sn is not None:
                dev_table_keys.add((fn, sn))
    
    for q in test_queries:
        gt_list = q.get('ground_truth_list', []) or []
        for gt in gt_list:
            fn = gt.get('file_name')
            sn = gt.get('sheet_name')
            if fn is not None and sn is not None:
                test_table_keys.add((fn, sn))
    
    # 讀取 table.jsonl
    with open(table_file, 'r', encoding='utf-8') as f:
        all_tables = [json.loads(line) for line in f]
    
    original_table_count = len(all_tables)
    print(f"  原始 table 數量: {original_table_count}")
    
    # 為了保持完整性，dev 和 test 都包含各自需要的表格
    # 使用 (file_name, sheet_name) 作為 key 來匹配
    dev_tables = [t for t in all_tables if (t.get('file_name'), t.get('sheet_name')) in dev_table_keys]
    test_tables = [t for t in all_tables if (t.get('file_name'), t.get('sheet_name')) in test_table_keys]
    
    # 如果沒有 ID 匹配（舊格式），則平分所有表格
    if not dev_tables and not test_tables:
        print("  使用表格平分策略（無 file_name/sheet_name 匹配）")
        random.shuffle(all_tables)
        mid_table = len(all_tables) // 2
        dev_tables = all_tables[:mid_table]
        test_tables = all_tables[mid_table:]
    
    print(f"  dev table 數量: {len(dev_tables)}")
    print(f"  test table 數量: {len(test_tables)}")
    
    # 創建 dev 目錄
    dev_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # 寫入 dev 數據
    dev_table_file = dev_dataset_dir / "table.jsonl"
    dev_query_file = dev_dataset_dir / "query.jsonl"
    
    with open(dev_table_file, 'w', encoding='utf-8') as f:
        for table in dev_tables:
            f.write(json.dumps(table, ensure_ascii=False) + '\n')
    
    with open(dev_query_file, 'w', encoding='utf-8') as f:
        for query in dev_queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    
    print(f"  已寫入 dev: {dev_table_file}")
    print(f"  已寫入 dev: {dev_query_file}")
    
    # 覆寫 test 數據（保留另一半）
    with open(table_file, 'w', encoding='utf-8') as f:
        for table in test_tables:
            f.write(json.dumps(table, ensure_ascii=False) + '\n')
    
    with open(query_file, 'w', encoding='utf-8') as f:
        for query in test_queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    
    print(f"  已更新 test: {table_file}")
    print(f"  已更新 test: {query_file}")


def main():
    # 獲取所有 test 數據集
    datasets = [d.name for d in TEST_DIR.iterdir() if d.is_dir()]
    
    print(f"找到 {len(datasets)} 個數據集: {', '.join(datasets)}")
    print("=" * 60)
    
    for dataset in sorted(datasets):
        try:
            split_dataset(dataset)
        except Exception as e:
            print(f"處理 {dataset} 時出錯: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("完成！")


if __name__ == "__main__":
    main()
