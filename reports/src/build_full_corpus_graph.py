#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建立全語料庫圖（train + dev + test 合併），存為：
  data/processed/full/{dataset}/graph.pt

同時輸出 query 對應檔（ground truth 重映射到全域 index）：
  data/table/full/{dataset}/query.jsonl   (test queries, GT mapped to full-corpus ids)

用法：
  cd reports/src && python build_full_corpus_graph.py
  cd reports/src && python build_full_corpus_graph.py --datasets feta ottqa

處理邏輯：
- 對每個 dataset，讀取 train/dev/test 三個 graph.pt
- 以「真實內容 key」（非序號 id）對 table 節點去重合併
  - feta/ottqa/e2ewtq: key = sheet_name|file_name（已是全局唯一）
  - mimo_en/mimo_ch/mmqa: key = file_name|sheet_name（從 metadata_maps.table_meta 讀）
    若 metadata 沒有，退回用 table embedding fingerprint（cosine）
- 合併後的圖保留 train split 的邊結構（最完整），
  dev/test 的新增表不加跨 split 相似度邊（has_column/comes_from/same_page 重建）
- 輸出 full/{dataset}/query.jsonl：test queries，GT key 映射到全語料庫的新 table idx
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F  # noqa: F401 (used in make_content_key collision check)
from torch_geometric.data import HeteroData

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

DATASETS = ['feta', 'ottqa', 'mimo_en', 'mimo_ch', 'e2ewtq', 'mmqa']


def get_key_fields(dataset):
    return ("sheet_name", "file_name") if dataset in ["ottqa", "feta", "e2ewtq"] else ("id",)


def make_content_key(meta, dataset):
    """從 table_meta 條目建立跨 split 唯一的 content key。
    mimo/mmqa 的 id 是各 split 序號，改用 embedding fingerprint（由呼叫方提供）。
    feta/ottqa/e2ewtq 的 sheet_name|file_name 是全局唯一的。
    """
    fn = meta.get('file_name', '')
    sn = meta.get('sheet_name', '')
    if dataset in ('feta', 'ottqa', 'e2ewtq'):
        return f"{sn}|{fn}"
    else:
        # mimo/mmqa: file_name+sheet_name 可能重複（同檔案多張相同名表），
        # 呼叫方需用 embedding fingerprint 補充辨識，此處只回傳前綴。
        return f"{fn}|{sn}" if (fn or sn) else None


def build_full_corpus(dataset):
    """合併三個 split 的圖，回傳 (merged_graph, test_global_id_to_idx, key_to_global_idx)"""
    print(f"\n{'='*60}")
    print(f"  {dataset.upper()}")
    print(f"{'='*60}")

    splits = ['train', 'dev', 'test']
    graphs = {}
    for sp in splits:
        p = PROJECT_DIR / f'data/processed/{sp}/{dataset}/graph.pt'
        if p.exists():
            graphs[sp] = torch.load(str(p), map_location='cpu', weights_only=False)
            print(f"  loaded {sp}: {graphs[sp]['table'].x.size(0)} tables")
        else:
            print(f"  {sp}: not found, skipping")

    if not graphs:
        print(f"  No graphs found for {dataset}, skipping")
        return None, None, None

    # Step 1: collect all unique tables by content key
    # key_to_info: content_key -> (split, local_idx, embedding)
    key_to_embedding = {}   # content_key -> tensor (embed_dim,)
    key_to_meta = {}        # content_key -> meta dict
    global_key_order = []   # ordered list of content keys (insertion order = global idx)

    for sp in splits:
        if sp not in graphs:
            continue
        g = graphs[sp]
        mm = g.metadata_maps if hasattr(g, 'metadata_maps') else {}
        table_meta = mm.get('table_meta', [])
        table_embs = g['table'].x  # (N, D)

        for local_idx in range(table_embs.size(0)):
            meta = table_meta[local_idx] if local_idx < len(table_meta) else {}
            ck = make_content_key(meta, dataset)

            if ck is None:
                ck = '_emb_' + '_'.join(f'{v:.3f}' for v in table_embs[local_idx, :8].tolist())

            # For mimo/mmqa, file_name|sheet_name can collide (multiple tables same name).
            # Use embedding cosine similarity to check if it's truly the same table.
            if ck in key_to_embedding and dataset not in ('feta', 'ottqa', 'e2ewtq'):
                existing_emb = key_to_embedding[ck]
                new_emb = table_embs[local_idx]
                cos_sim = float(F.cosine_similarity(existing_emb.unsqueeze(0), new_emb.unsqueeze(0)))
                if cos_sim < 0.9999:
                    ck = ck + '|_emb_' + '_'.join(f'{v:.3f}' for v in new_emb[:4].tolist())

            if ck not in key_to_embedding:
                key_to_embedding[ck] = table_embs[local_idx]
                key_to_meta[ck] = meta
                global_key_order.append(ck)

    n_global = len(global_key_order)
    key_to_global_idx = {k: i for i, k in enumerate(global_key_order)}
    print(f"  Total unique tables: {n_global} "
          f"(train={graphs.get('train', HeteroData())['table'].num_nodes if 'train' in graphs else 0}, "
          f"after dedup)")

    # Step 2: build merged node feature matrix
    embed_dim = list(key_to_embedding.values())[0].size(0)
    table_x = torch.stack([key_to_embedding[k] for k in global_key_order])  # (N_global, D)

    # Step 3: use train graph's edge structure as base, remap local -> global indices
    # for column and page nodes, just concatenate all splits (they are split-local)
    # For cross-split edges (has_column, comes_from, same_page), rebuild from train only
    # dev/test-only tables get no graph edges (they exist as isolated nodes for embedding lookup)

    # For simplicity: use train graph as the merged graph base, add dev/test-only tables
    # as isolated nodes. This is valid because:
    #   - ottqa/mimo/mmqa: dev/test ⊆ train → no new tables, same graph
    #   - feta/e2ewtq: dev/test tables are new → added as isolated nodes without edges

    if 'train' in graphs:
        base = graphs['train']
    else:
        base = list(graphs.values())[0]

    merged = HeteroData()

    # Build local->global index map for train tables
    train_mm = base.metadata_maps if hasattr(base, 'metadata_maps') else {}
    train_meta = train_mm.get('table_meta', [])
    train_local_to_global = {}
    for li in range(base['table'].x.size(0)):
        meta = train_meta[li] if li < len(train_meta) else {}
        ck = make_content_key(meta, dataset)
        if ck is None:
            ck = '_emb_' + '_'.join(f'{v:.3f}' for v in base['table'].x[li, :8].tolist())
        if ck in key_to_global_idx:
            train_local_to_global[li] = key_to_global_idx[ck]

    # Remap all edges from train graph (table nodes only need remapping;
    # column/page nodes keep their split-local indices as-is)
    def remap_edge(edge_index, src_type, dst_type):
        """Remap table node indices from train-local to global. Non-table stays as-is."""
        src, dst = edge_index[0], edge_index[1]
        if src_type == 'table':
            src = torch.tensor([train_local_to_global.get(i.item(), i.item()) for i in src])
        if dst_type == 'table':
            dst = torch.tensor([train_local_to_global.get(i.item(), i.item()) for i in dst])
        return torch.stack([src, dst])

    # Copy node features
    merged['table'].x = table_x  # global pool
    for nt in ['column', 'page']:
        if nt in base.node_types:
            for attr in base[nt].keys():
                merged[nt][attr] = base[nt][attr]

    # Copy and remap edges
    for et in base.edge_types:
        src_t, rel, dst_t = et
        ei = base[et].edge_index
        merged[et].edge_index = remap_edge(ei, src_t, dst_t)

    # Metadata maps: update table_id_to_idx and table_meta for full corpus
    new_id_to_idx = {}
    new_table_meta = []
    for gi, ck in enumerate(global_key_order):
        meta = key_to_meta[ck]
        new_table_meta.append(meta)
        kf = get_key_fields(dataset)
        kid = '|'.join(str(meta.get(f, '')) for f in kf)
        new_id_to_idx[kid] = gi

    merged.metadata_maps = {
        'table_id_to_idx': new_id_to_idx,
        'table_meta': new_table_meta,
        'key_fields': get_key_fields(dataset),
    }

    print(f"  Merged graph: {merged['table'].x.size(0)} table nodes")
    print(f"  Edge types: {[et[1] for et in merged.edge_types]}")

    # Step 4: build test query file with GT remapped to global idx
    # For mimo/mmqa where key_fields=('id',), we need a different lookup strategy:
    # the test graph's local indices → global indices (built via embedding matching during dedup).
    test_graph_path = PROJECT_DIR / f'data/processed/test/{dataset}/graph.pt'
    test_local_to_global = {}
    if dataset not in ('feta', 'ottqa', 'e2ewtq') and test_graph_path.exists():
        test_g = graphs.get('test')
        if test_g is not None:
            test_embs = test_g['table'].x
            for li in range(test_embs.size(0)):
                # find the global table with highest cosine similarity to this test table
                sims = F.cosine_similarity(test_embs[li].unsqueeze(0), table_x)
                best = int(sims.argmax())
                if float(sims[best]) > 0.9999:
                    test_local_to_global[li] = best

    # Build test_split id→local_idx map (for mimo/mmqa, id is local sequence number)
    test_mm = (graphs.get('test') or HeteroData()).metadata_maps if hasattr(graphs.get('test', HeteroData()), 'metadata_maps') else {}
    test_id_to_local = {}
    if dataset not in ('feta', 'ottqa', 'e2ewtq'):
        for li, m in enumerate(test_mm.get('table_meta', [])):
            test_id_to_local[str(m.get('id', li))] = li

    test_query_path = PROJECT_DIR / f'data/table/test/{dataset}/query.jsonl'
    remapped_queries = []
    if test_query_path.exists():
        kf = get_key_fields(dataset)
        for line in open(test_query_path, encoding='utf-8'):
            obj = json.loads(line)
            new_gt = []
            for gt in obj.get('ground_truth_list') or []:
                if dataset in ('feta', 'ottqa', 'e2ewtq'):
                    ck = '|'.join([str(gt.get(f, '')) for f in kf])
                    if ck in key_to_global_idx:
                        new_gt.append({'_global_idx': key_to_global_idx[ck], '_ck': ck})
                else:
                    # mimo/mmqa: use local index → global via embedding matching
                    gt_id = str(gt.get('id', ''))
                    li = test_id_to_local.get(gt_id)
                    if li is not None and li in test_local_to_global:
                        gi = test_local_to_global[li]
                        new_gt.append({'_global_idx': gi, '_ck': global_key_order[gi]})
            obj['_global_ground_truth'] = new_gt
            remapped_queries.append(obj)
        print(f"  Test queries remapped: {len(remapped_queries)} "
              f"({sum(1 for q in remapped_queries if q['_global_ground_truth'])} with GT)")

    return merged, remapped_queries, key_to_global_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    args = ap.parse_args()

    for dataset in args.datasets:
        merged, queries, key_map = build_full_corpus(dataset)
        if merged is None:
            continue

        # Save graph
        out_graph_dir = PROJECT_DIR / f'data/processed/full/{dataset}'
        out_graph_dir.mkdir(parents=True, exist_ok=True)
        graph_path = out_graph_dir / 'graph.pt'
        torch.save(merged, str(graph_path))
        print(f"  Saved graph → {graph_path}")

        # Save remapped test queries
        if queries:
            out_query_dir = PROJECT_DIR / f'data/table/full/{dataset}'
            out_query_dir.mkdir(parents=True, exist_ok=True)
            query_path = out_query_dir / 'query.jsonl'
            with open(query_path, 'w', encoding='utf-8') as f:
                for q in queries:
                    f.write(json.dumps(q, ensure_ascii=False) + '\n')
            print(f"  Saved queries → {query_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
