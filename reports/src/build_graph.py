"""建置異構圖 (HeteroData) 用於 GNN 訓練與評估"""
import json
import csv
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path

# ========= 可調參數 =========
SOURCES = [
    'train/mimo_en',
    'test/mimo_en',
    'dev/mimo_en',

    'train/mimo_ch',
    'test/mimo_ch',
    'dev/mimo_ch',

    'train/ottqa',
    'test/ottqa',
    'dev/ottqa',

    # 'train/feta',
    # 'test/feta',
    # 'dev/feta',

    'train/e2ewtq',
    'test/e2ewtq',
    'dev/e2ewtq',

    'train/mmqa',
    'test/mmqa',
    'dev/mmqa',
]

TABLE_JSONL_PATH = '/user_data/TabGNN/data/table/{source}/table.jsonl'
OUTPUT_GRAPH_PATH = '/user_data/TabGNN/data/processed/{source}/graph.pt'
MODEL_NAME = 'BAAI/bge-m3'
DEVICE = 'cuda'

# 相似度邊的 Top-K 設定
K_TABLE = 5   # 每張表連接 K 個最相似的表
K_COLUMN = 5  # 每個欄位連接 K 個最相似的欄位
# ===========================


def make_key(item: dict, key_fields: tuple) -> str:
    """根據 key_fields 組合產生唯一鍵"""
    return "|".join(str(item.get(f, "")) for f in key_fields)


def get_key_fields(dataset_name: str) -> tuple:
    """根據資料集名稱返回對應的 KEY_FIELDS"""
    if dataset_name in ["ottqa", "feta", "e2ewtq"]:
        return ("sheet_name", "file_name")
    elif dataset_name in ["mimo_en", "mimo_ch", "mmqa"]:
        return ("id",)
    else:
        return ("id",)


def main(source: str = None):
    if source is None:
        source = SOURCES[0]
    table_jsonl_path = TABLE_JSONL_PATH.format(source=source)
    output_graph_path = OUTPUT_GRAPH_PATH.format(source=source)
    
    # 從 source 提取資料集名稱 (e.g., "train/mmqa" -> "mmqa")
    dataset_name = source.split('/')[-1]
    KEY_FIELDS = get_key_fields(dataset_name)
    
    print(f"\n{'='*60}")
    print(f"處理 SOURCE: {source}")
    print(f"資料集: {dataset_name}, KEY_FIELDS: {KEY_FIELDS}")
    print(f"{'='*60}")

    # --- 1. 讀取資料 ---
    tables = []
    with open(table_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            tables.append(json.loads(line))
    print(f"讀取完成，共 {len(tables)} 張表格。")

    # 載入嵌入模型
    embedder = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # --- 2. 解析表格並建立節點 ---
    data = HeteroData()
    print("\n=== 解析 JSONL ===")

    table_docs = []   # 表格的文字描述
    column_docs = []  # 欄位的文字描述
    page_docs = []    # 頁面標題

    # 索引映射
    table_id_to_idx = {}
    table_meta = []  # 每個 table 節點的完整 metadata (id, file_name, sheet_name)
    column_global_id_to_idx = {}
    page_title_to_idx = {}

    # 結構邊
    edge_table_to_col_src, edge_table_to_col_dst = [], []
    edge_table_to_page_src, edge_table_to_page_dst = [], []

    # 原始資料
    meta_table_ids = []
    meta_page_titles = []
    meta_col_global_ids = []

    # 用於建立 same_page 邊
    page_to_table_idxs = {}

    for item in tqdm(tables, desc="解析表格"):
        try:
            # 解析 header
            # 解析 header
            header_list = item.get('header')
            if not header_list:
                continue

            # 解析 instances
            instance_rows = [next(csv.reader([row_str.replace('\n', ' ').replace('\r', ' ')])) for row_str in item.get('instances', [])]

            file_name = item.get('file_name', '')
            sheet_name = item.get('sheet_name', '')
            table_id = make_key(item, KEY_FIELDS)
            if table_id in table_id_to_idx:
                continue  # 跳過重複

            # 表格節點索引
            current_table_idx = len(table_docs)
            table_id_to_idx[table_id] = current_table_idx
            meta_table_ids.append(table_id)
            table_meta.append({
                'id': item.get('id', ''),
                'file_name': file_name,
                'sheet_name': sheet_name,
            })

            # 頁面節點
            metadata = item.get('metadata', {})
            page_title = metadata.get('table_page_title') or \
                         metadata.get('title') or \
                         metadata.get('table_section_title') or \
                         item.get('file_name') or \
                         f"__UNKNOWN_PAGE_{table_id}__"
            if page_title not in page_title_to_idx:
                current_page_idx = len(page_docs)
                page_title_to_idx[page_title] = current_page_idx
                page_docs.append(page_title)
                meta_page_titles.append(page_title)
            else:
                current_page_idx = page_title_to_idx[page_title]

            # Table -> Page 邊
            edge_table_to_page_src.append(current_table_idx)
            edge_table_to_page_dst.append(current_page_idx)

            # 記錄同頁表格
            if page_title not in page_to_table_idxs:
                page_to_table_idxs[page_title] = []
            page_to_table_idxs[page_title].append(current_table_idx)

            # 表格文字描述
            table_doc = " ".join(filter(None, [
                f"Page: {page_title}",
                f"Sheet: {item.get('sheet_name', '')}",
                f"Section: {item.get('metadata', {}).get('table_section_title', '')}",
                f"Columns: {', '.join(header_list)}",
                f"Data: {'; '.join([', '.join(row) for row in instance_rows[:5]])}"
            ]))
            table_docs.append(table_doc)

            # 欄位節點
            for col_idx, col_name in enumerate(header_list):
                col_global_id = f"{table_id}::{col_name}::{col_idx}"
                if col_global_id in column_global_id_to_idx:
                    continue

                current_col_idx = len(column_docs)
                column_global_id_to_idx[col_global_id] = current_col_idx
                meta_col_global_ids.append(col_global_id)

                # Table -> Column 邊
                edge_table_to_col_src.append(current_table_idx)
                edge_table_to_col_dst.append(current_col_idx)

                # 欄位文字描述
                sample_values = [row[col_idx] for row in instance_rows[:10] if len(row) > col_idx and row[col_idx].strip()]
                col_doc = " ".join(filter(None, [
                    f"Column: {col_name}",
                    f"Belongs to table: {page_title} - {item.get('sheet_name', '')}",
                    f"Values: {', '.join(sample_values)}"
                ]))
                column_docs.append(col_doc)

        except Exception as e:
            print(f"處理表格 {item.get('id')} 時發生錯誤: {e}")
            continue

    print(f"\n--- 建圖 (共 {len(table_docs)} 表, {len(column_docs)} 欄, {len(page_docs)} 頁) ---")

    # --- 3. 計算嵌入向量（分批處理避免 OOM）---
    print("開始 embeddings...")
    batch_size = 128
    
    # Table embeddings
    table_embeds = []
    for i in range(0, len(table_docs), batch_size):
        batch = table_docs[i:i+batch_size]
        embeds = embedder.encode(batch, show_progress_bar=False)
        table_embeds.append(embeds)
        if (i // batch_size) % 5 == 0:
            torch.cuda.empty_cache()
    data['table'].x = torch.tensor(np.vstack(table_embeds), dtype=torch.float)
    del table_embeds
    torch.cuda.empty_cache()
    print(f"  Table embeddings 完成 ({len(table_docs)} tables)")
    
    # Column embeddings
    column_embeds = []
    for i in range(0, len(column_docs), batch_size):
        batch = column_docs[i:i+batch_size]
        embeds = embedder.encode(batch, show_progress_bar=False)
        column_embeds.append(embeds)
        if (i // batch_size) % 5 == 0:
            torch.cuda.empty_cache()
    data['column'].x = torch.tensor(np.vstack(column_embeds), dtype=torch.float)
    del column_embeds
    torch.cuda.empty_cache()
    print(f"  Column embeddings 完成 ({len(column_docs)} columns)")
    
    # Page embeddings
    data['page'].x = torch.tensor(embedder.encode(page_docs, show_progress_bar=False), dtype=torch.float)
    torch.cuda.empty_cache()
    print(f"  Page embeddings 完成 ({len(page_docs)} pages)")
    print("所有 embeddings 完成。")

    # 儲存原始 ID
    data['table'].id = meta_table_ids
    data['column'].global_id = meta_col_global_ids
    data['page'].title = meta_page_titles

    # 儲存映射表
    data.metadata_maps = {
        'table_id_to_idx': table_id_to_idx,
        'table_meta': table_meta,
        'key_fields': KEY_FIELDS,
        'column_global_id_to_idx': column_global_id_to_idx,
        'page_title_to_idx': page_title_to_idx
    }

    # --- 4. 建立結構邊 ---
    data['table', 'has_column', 'column'].edge_index = torch.tensor([edge_table_to_col_src, edge_table_to_col_dst], dtype=torch.long)
    data['table', 'comes_from', 'page'].edge_index = torch.tensor([edge_table_to_page_src, edge_table_to_page_dst], dtype=torch.long)

    # Same Page 邊
    print("正在建立同一 page 的 table 連邊 (same_page)...")
    same_src, same_dst = [], []
    for t_idxs in page_to_table_idxs.values():
        if len(t_idxs) <= 1:
            continue
        for i in range(len(t_idxs)):
            for j in range(i + 1, len(t_idxs)):
                same_src.extend([t_idxs[i], t_idxs[j]])
                same_dst.extend([t_idxs[j], t_idxs[i]])

    if same_src:
        data['table', 'same_page', 'table'].edge_index = torch.tensor([same_src, same_dst], dtype=torch.long)
        print(f"建立 same_page edges: {len(same_src)} 條（含反向）")
    else:
        print("沒有發現同一 page 下有多於 1 張 table，未建立 same_page edges。")

    # 稍後在所有邊添加完成後再建立反向邊

    # --- 5. 建立 Embedding-based 表格相似度邊 ---
    print("\n計算 'table' <-> 'table' Embedding 相似度邊...")
    K_TABLE = 5
    
    xt = F.normalize(data['table'].x, p=2, dim=1)
    sim_matrix = torch.matmul(xt, xt.T)
    sim_matrix.fill_diagonal_(float('-inf'))
    _, top_k_indices = torch.topk(sim_matrix, k=min(K_TABLE, xt.size(0) - 1), dim=1)
    
    sim_src, sim_dst = [], []
    for i in range(xt.size(0)):
        for j in top_k_indices[i]:
            sim_src.append(i)
            sim_dst.append(j.item())
    
    if sim_src:
        data['table', 'similar_table', 'table'].edge_index = torch.tensor([sim_src, sim_dst], dtype=torch.long)
        print(f"建立 similar_table edges (embedding-based): {len(sim_src)} 條")
    
    # --- 6. 建立 Embedding-based 欄位相似度邊 ---
    print("計算 'column' <-> 'column' Embedding 相似度邊...")
    K_COLUMN = 5
    
    xc = F.normalize(data['column'].x, p=2, dim=1)
    sim_matrix_c = torch.matmul(xc, xc.T)
    
    sim_matrix_c.fill_diagonal_(float('-inf'))
    _, top_k_indices_c = torch.topk(sim_matrix_c, k=min(K_COLUMN, xc.size(0) - 1), dim=1)
    
    sim_src_ce, sim_dst_ce = [], []
    for i in range(xc.size(0)):
        for j in top_k_indices_c[i]:
            sim_src_ce.append(i)
            sim_dst_ce.append(j.item())
    
    if sim_src_ce:
        data['column', 'similar_content', 'column'].edge_index = torch.tensor([sim_src_ce, sim_dst_ce], dtype=torch.long)
        print(f"建立 similar_content edges (embedding-based): {len(sim_src_ce)} 條")
    
    # --- 7. 建立 Name-based 表格相似度邊 (基於共享欄位名稱) ---
    print("建立 'table' <-> 'table' 基於共享欄位名稱的連邊...")
    from itertools import combinations

    # 建立 column_idx -> table_idx 的映射
    col_idx_to_table_idx = {}
    for table_idx, col_idx in zip(edge_table_to_col_src, edge_table_to_col_dst):
        col_idx_to_table_idx[col_idx] = table_idx
    
    # 從 column global_id 中提取 column name
    # global_id 格式: "{table_id}::{col_name}::{col_idx}"
    col_name_to_table_indices = defaultdict(set)
    
    for idx, global_id in enumerate(meta_col_global_ids):
        parts = global_id.split('::')
        if len(parts) >= 2:
            col_name = parts[1].strip()  # 提取 column name
            # 過濾空字串，避免所有表格互連
            if col_name and col_name != '':
                # 找到這個 column 所屬的 table
                if idx in col_idx_to_table_idx:
                    table_idx = col_idx_to_table_idx[idx]
                    col_name_to_table_indices[col_name].add(table_idx)
    
    table_edge_set = set()
    
    # 對於每個 column name，將擁有這個 column name 的所有 table 互連
    for table_indices in col_name_to_table_indices.values():
        table_indices_list = list(table_indices)
        if len(table_indices_list) > 1:
            for i, j in combinations(table_indices_list, 2):
                table_edge_set.add((i, j))
                table_edge_set.add((j, i))
    
    # 轉換為列表
    if table_edge_set:
        shared_src, shared_dst = zip(*table_edge_set)
        shared_src, shared_dst = list(shared_src), list(shared_dst)
    else:
        shared_src, shared_dst = [], []
    
    if shared_src:
        data['table', 'shared_column_name', 'table'].edge_index = torch.tensor([shared_src, shared_dst], dtype=torch.long)
        print(f"建立 shared_column_name edges (table-table): {len(shared_src)} 條")
    else:
        print("未建立任何 shared_column_name edges (沒有共享的 column names)")

    # --- 8. 建立反向邊（手動為所有邊類型添加）---
    print("\n正在建立反向邊...")
    
    # 為每個異構邊類型添加反向邊
    edge_types_to_reverse = list(data.edge_types)
    for src_type, edge_type, dst_type in edge_types_to_reverse:
        if src_type != dst_type:  # 跳過自環型邊（它們已經是雙向的）
            # 創建反向邊類型名稱
            rev_edge_type = f'rev_{edge_type}'
            edge_index = data[src_type, edge_type, dst_type].edge_index
            # 添加反向邊（交換 src 和 dst）
            data[dst_type, rev_edge_type, src_type].edge_index = edge_index.flip(0)
            print(f"  添加反向邊: ({dst_type}, {rev_edge_type}, {src_type})")
    
    print(f"反向邊建立完成，共 {len(data.edge_types)} 種邊類型")
    
    # --- 9. 儲存圖 ---
    print("\n--- 圖構建完成！ ---")
    print(f"最終圖結構: {data}")
    # print(data) # 避免打印所有節點 ID 造成卡頓

    print(f"\n將圖儲存至 {output_graph_path}...")
    Path(output_graph_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_graph_path)
    print("完成！")


if __name__ == "__main__":
    for src in SOURCES:
        table_path = Path(TABLE_JSONL_PATH.format(source=src))
        if not table_path.exists():
            print(f"跳過 {src}：找不到 {table_path}")
            continue
        main(source=src)