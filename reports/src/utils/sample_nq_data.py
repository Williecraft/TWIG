import json
import random
import os
import shutil
from tqdm import tqdm
import argparse

# 設定隨機種子以確保可重複性
SEED = 42
random.seed(SEED)

# 過濾巨大表格的閾值
MAX_COLS = 30
MAX_CHARS = 15000

# 預設訓練設定 (若無參數)
DEFAULT_INPUT_DIR = "data/table/train/nq_tables"
DEFAULT_OUTPUT_DIR = "data/table/train/nq_tables_sampled"
DEFAULT_TARGET_COUNT = 7326

def is_huge_table(table):
    """檢查表格是否過大"""
    # 檢查欄位數
    if table.get('header') and len(table['header']) > MAX_COLS:
        return True
    # 檢查字串長度 (粗略估計 token 數)
    if len(json.dumps(table)) > MAX_CHARS:
        return True
    return False

def process_dataset(input_dir, output_dir, target_count):
    print(f"從 {input_dir} 取樣至 {output_dir}")
    print(f"目標表格數量: {target_count}")
    print(f"過濾條件: Columns > {MAX_COLS} or Chars > {MAX_CHARS}")

    # 確保輸出目錄存在
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # 1. 讀取所有 Query
    query_file = os.path.join(input_dir, "query.jsonl")
    print("讀取 queries...")
    with open(query_file, 'r', encoding='utf-8') as f:
        all_queries = [json.loads(line) for line in f]
    
    print(f"總共 {len(all_queries)} 筆 queries")

    # 2. 決定 Query 樣本
    # 如果是測試集 (通常 Query 數量少於 target_count)，則全部保留
    # 如果是訓練集 (通常 Query 數量多)，則隨機取樣
    if len(all_queries) <= target_count:
        sampled_queries = all_queries
        print(f"保留所有 {len(sampled_queries)} 筆 queries")
    else:
        sampled_queries = random.sample(all_queries, target_count)
        print(f"隨機取樣 {len(sampled_queries)} 筆 queries")

    # 3. 找出需要的表格 ID (Ground Truth)
    required_table_ids = set()
    for q in sampled_queries:
        for gt in q.get('ground_truth_list', []):
            if 'id' in gt:
                required_table_ids.add(str(gt['id'])) # 統一轉為字串比較
    
    print(f"需要 {len(required_table_ids)} 張 Ground Truth 表格")

    # 4. 讀取所有 Table 並篩選
    table_file = os.path.join(input_dir, "table.jsonl")
    selected_tables = []
    remaining_tables = []
    huge_gt_tables = set() # 記錄過大的 GT 表格 ID
    
    print("讀取並篩選 tables...")
    with open(table_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            table = json.loads(line)
            table_id = str(table['id']) # 統一轉為字串
            is_huge = is_huge_table(table)

            if table_id in required_table_ids:
                if is_huge:
                    huge_gt_tables.add(table_id)
                    # 雖然過大，但先加入，稍後會連同 Query 一起移除
                    selected_tables.append(table)
                else:
                    selected_tables.append(table)
            else:
                if not is_huge: # 只保留非巨大的表格作為 Distractors
                    remaining_tables.append(table)
    
    # 處理過大的 GT 表格：移除相關的 Query 和 Table
    if huge_gt_tables:
        print(f"發現 {len(huge_gt_tables)} 張過大的 Ground Truth 表格，將移除相關 Query...")
        
        # 移除依賴這些巨大表格的 Query
        valid_queries = []
        for q in sampled_queries:
            has_huge_gt = False
            for gt in q.get('ground_truth_list', []):
                if 'id' in gt and str(gt['id']) in huge_gt_tables:
                    has_huge_gt = True
                    break
            if not has_huge_gt:
                valid_queries.append(q)
        
        print(f"移除後剩餘 {len(valid_queries)} 筆有效 queries (原 {len(sampled_queries)})")
        sampled_queries = valid_queries

        # 從 selected_tables 中移除這些巨大表格
        selected_tables = [t for t in selected_tables if str(t['id']) not in huge_gt_tables]
        print(f"移除後剩餘 {len(selected_tables)} 張 Ground Truth 表格")

    # 檢查是否有遺失的 GT 表格
    required_table_ids = set() # 重新計算需要的 ID
    for q in sampled_queries:
        for gt in q.get('ground_truth_list', []):
            if 'id' in gt:
                required_table_ids.add(str(gt['id']))

    found_gt_ids = set(str(t['id']) for t in selected_tables)
    missing_gt = required_table_ids - found_gt_ids
    if missing_gt:
        print(f"警告: 有 {len(missing_gt)} 張 Ground Truth 表格在 table.jsonl 中找不到！")
        # 再次過濾
        valid_queries = []
        for q in sampled_queries:
            is_valid = True
            for gt in q.get('ground_truth_list', []):
                if 'id' in gt and str(gt['id']) in missing_gt:
                    is_valid = False
                    break
            if is_valid:
                valid_queries.append(q)
        print(f"修正後剩餘 {len(valid_queries)} 筆有效 queries")
        sampled_queries = valid_queries

    # 5. 補齊表格數量 (Distractors)
    current_table_count = len(selected_tables)
    needed_distractors = target_count - current_table_count
    
    if needed_distractors > 0:
        print(f"目前有 {current_table_count} 張表格，需補齊 {needed_distractors} 張負樣本表格...")
        if len(remaining_tables) >= needed_distractors:
            distractors = random.sample(remaining_tables, needed_distractors)
            selected_tables.extend(distractors)
            print(f"成功補齊至 {len(selected_tables)} 張表格")
        else:
            print(f"警告: 剩餘表格不足，全部加入。總數: {len(selected_tables) + len(remaining_tables)}")
            selected_tables.extend(remaining_tables)
    else:
        print(f"表格已足夠 ({current_table_count} >= {target_count})，不需補齊。")

    # 6. 寫入檔案
    out_query_path = os.path.join(output_dir, "query.jsonl")
    out_table_path = os.path.join(output_dir, "table.jsonl")
    
    print(f"寫入 {out_query_path}...")
    with open(out_query_path, 'w', encoding='utf-8') as f:
        for q in sampled_queries:
            f.write(json.dumps(q) + '\n')
            
    print(f"寫入 {out_table_path}...")
    with open(out_table_path, 'w', encoding='utf-8') as f:
        for t in selected_tables:
            f.write(json.dumps(t) + '\n')

    print("處理完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample and filter NQ tables dataset')
    parser.add_argument('--input', '-i', type=str, default=DEFAULT_INPUT_DIR, help='Input directory')
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--count', '-c', type=int, default=DEFAULT_TARGET_COUNT, help='Target table count')
    
    args = parser.parse_args()
    
    process_dataset(args.input, args.output, args.count)
