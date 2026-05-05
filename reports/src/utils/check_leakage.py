"""檢查訓練/測試資料洩漏 & 候選池大小"""
import json
import torch
from pathlib import Path

DATASETS = [
    # "mimo_en", 
    # "mimo_ch", 
    # "mmqa",
    "ottqa", 
    "feta", 
    "e2ewtq"
]
KEY_FIELDS = ("file_name", "sheet_name")

def make_key(item: dict) -> str:
    return "|".join(str(item.get(f, "")) for f in KEY_FIELDS)

print("=" * 80)
print("1. 候選池大小分析（測試圖中有多少張表格）")
print("=" * 80)

for ds in DATASETS:
    test_graph = f"/user_data/TabGNN/data/processed/test/{ds}/graph.pt"
    train_graph = f"/user_data/TabGNN/data/processed/train/{ds}/graph.pt"

    if not Path(test_graph).exists():
        print(f"  [{ds}] 測試圖不存在，跳過")
        continue

    test_data = torch.load(test_graph, map_location='cpu', weights_only=False)
    num_test_tables = test_data['table'].x.size(0)

    # 計算 test query 數量
    test_query_file = f"/user_data/TabGNN/data/table/test/{ds}/query.jsonl"
    num_test_queries = 0
    if Path(test_query_file).exists():
        with open(test_query_file) as f:
            num_test_queries = sum(1 for _ in f)

    print(f"\n  [{ds}]")
    print(f"    測試圖表格數: {num_test_tables}")
    print(f"    測試 query 數: {num_test_queries}")
    print(f"    比率 (queries/tables): {num_test_queries/num_test_tables:.2f}" if num_test_tables > 0 else "")

print("\n")
print("=" * 80)
print("2. 訓練集 vs 測試集表格重疊分析")
print("=" * 80)

for ds in DATASETS:
    train_graph = f"/user_data/TabGNN/data/processed/train/{ds}/graph.pt"
    test_graph = f"/user_data/TabGNN/data/processed/test/{ds}/graph.pt"

    if not Path(train_graph).exists() or not Path(test_graph).exists():
        print(f"  [{ds}] 訓練或測試圖不存在，跳過")
        continue

    train_data = torch.load(train_graph, map_location='cpu', weights_only=False)
    test_data = torch.load(test_graph, map_location='cpu', weights_only=False)

    # 提取表格鍵
    train_keys = set()
    if hasattr(train_data, 'metadata_maps') and 'table_meta' in train_data.metadata_maps:
        for meta in train_data.metadata_maps['table_meta']:
            train_keys.add(make_key(meta))
    elif hasattr(train_data, 'metadata_maps') and 'table_id_to_idx' in train_data.metadata_maps:
        train_keys = set(train_data.metadata_maps['table_id_to_idx'].keys())

    test_keys = set()
    if hasattr(test_data, 'metadata_maps') and 'table_meta' in test_data.metadata_maps:
        for meta in test_data.metadata_maps['table_meta']:
            test_keys.add(make_key(meta))
    elif hasattr(test_data, 'metadata_maps') and 'table_id_to_idx' in test_data.metadata_maps:
        test_keys = set(test_data.metadata_maps['table_id_to_idx'].keys())

    overlap = train_keys & test_keys

    print(f"\n  [{ds}]")
    print(f"    訓練集表格數: {len(train_keys)}")
    print(f"    測試集表格數: {len(test_keys)}")
    print(f"    重疊表格數:   {len(overlap)}")
    print(f"    重疊率 (overlap/test): {len(overlap)/len(test_keys)*100:.1f}%" if test_keys else "")

    if overlap and len(overlap) <= 20:
        print(f"    重疊的表格鍵:")
        for k in sorted(overlap):
            print(f"      - {k}")
    elif overlap:
        print(f"    重疊的表格鍵 (僅列前 20 個):")
        for k in sorted(overlap)[:20]:
            print(f"      - {k}")

print("\n")
print("=" * 80)
print("3. 訓練集 vs 測試集 Query 重疊分析")
print("=" * 80)

for ds in DATASETS:
    train_query = f"/user_data/TabGNN/data/table/train/{ds}/query.jsonl"
    test_query = f"/user_data/TabGNN/data/table/test/{ds}/query.jsonl"

    if not Path(train_query).exists() or not Path(test_query).exists():
        print(f"  [{ds}] 訓練或測試 query 不存在，跳過")
        continue

    train_questions = set()
    with open(train_query, encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get('question', '') or (obj.get('questions', [''])[0] if obj.get('questions') else '')
            if q.strip():
                train_questions.add(q.strip())

    test_questions = set()
    with open(test_query, encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get('question', '') or (obj.get('questions', [''])[0] if obj.get('questions') else '')
            if q.strip():
                test_questions.add(q.strip())

    overlap_q = train_questions & test_questions

    print(f"\n  [{ds}]")
    print(f"    訓練集 query 數: {len(train_questions)}")
    print(f"    測試集 query 數: {len(test_questions)}")
    print(f"    重疊 query 數:   {len(overlap_q)}")
    print(f"    重疊率 (overlap/test): {len(overlap_q)/len(test_questions)*100:.1f}%" if test_questions else "")

    if overlap_q and len(overlap_q) <= 10:
        print(f"    重疊的 query:")
        for q in sorted(overlap_q):
            print(f"      - {q[:80]}...")

print("\n")
print("=" * 80)
print("4. 測試集 Ground Truth 表格 vs 測試圖中表格的匹配分析")
print("=" * 80)

for ds in DATASETS:
    test_graph = f"/user_data/TabGNN/data/processed/test/{ds}/graph.pt"
    test_query = f"/user_data/TabGNN/data/table/test/{ds}/query.jsonl"

    if not Path(test_graph).exists() or not Path(test_query).exists():
        continue

    test_data = torch.load(test_graph, map_location='cpu', weights_only=False)
    test_keys = set()
    if hasattr(test_data, 'metadata_maps') and 'table_meta' in test_data.metadata_maps:
        for meta in test_data.metadata_maps['table_meta']:
            test_keys.add(make_key(meta))

    # 檢查每個 query 的 ground truth 是否都在圖裡
    total_queries = 0
    queries_with_all_gt_in_graph = 0
    queries_with_any_gt_in_graph = 0
    queries_with_no_gt = 0

    with open(test_query, encoding='utf-8') as f:
        for line in f:
            total_queries += 1
            obj = json.loads(line)
            gt_list = obj.get('ground_truth_list', []) or []
            if not gt_list:
                queries_with_no_gt += 1
                continue

            gt_keys_in_graph = []
            for gt in gt_list:
                k = make_key(gt)
                gt_keys_in_graph.append(k in test_keys)

            if all(gt_keys_in_graph):
                queries_with_all_gt_in_graph += 1
            if any(gt_keys_in_graph):
                queries_with_any_gt_in_graph += 1

    print(f"\n  [{ds}]")
    print(f"    總 query 數: {total_queries}")
    print(f"    所有 GT 都在圖中: {queries_with_all_gt_in_graph}")
    print(f"    至少一個 GT 在圖中: {queries_with_any_gt_in_graph}")
    print(f"    沒有 GT 的 query: {queries_with_no_gt}")

print("\n✅ 檢查完畢！")
