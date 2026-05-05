#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融實驗一：固定表示 (M0) vs 查詢參與式表示 (M1)

M0：固定表示（Baseline）
    - 離線 GNN 產生固定 table embedding
    - 查詢向量與固定 table embedding 做 cosine similarity 排序

M1：查詢參與式表示（Query-node + Subgraph Inference）
    - 先用固定 embedding 粗排取 top-k 候選（k > TOP_K）
    - 建立動態子圖，加入 query 節點 + Query–Table 連邊
    - 在子圖上執行 GNN forward
    - 用更新後的 table embedding 重新排序
"""

import json
import math
import gc
import sys
from typing import List, Set

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphSAGE, to_hetero, GraphNorm
from torch import nn
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_model import DiffusionModel, get_embedder
from run_edge_ablation import filter_edges

# ========= 可調參數 =========
DATASETS = [
    ("mimo_en", "mimo_en"),
    ("mimo_ch", "mimo_ch"),
    ("ottqa", "ottqa"),
    ("feta", "feta"),
    ("e2ewtq", "e2ewtq"),
    ("mmqa", "mmqa"),
]

SUBGRAPH_K = 20  # 粗排取 top-20 候選，再從中重排序選 top-10

# 每個資料集的最佳基礎圖連邊配置
BEST_EDGE_CONFIGS = {
    "feta": ["has_column", "same_page"],
    "ottqa": ["has_column", "similar_content"],
    "mimo_en": ["has_column", "similar_content", "shared_column_name"],
    "mimo_ch": ["similar_content"],
    "e2ewtq": ["similar_content"],
    "mmqa": ["similar_table", "has_column", "comes_from", "same_page", "similar_content"],
}

QUERY_FILE = "/user_data/TabGNN/data/table/test/{source}/query.jsonl"
MODEL_PATH = "/user_data/TabGNN/checkpoints/{dataset}/model.pt"
GRAPH_PATH = "/user_data/TabGNN/data/processed/test/{source}/graph.pt"
RESULT_DIR = "/user_data/TabGNN/results/ablation1_fixed_vs_query"
TOP_K = 10
# ===========================


def get_key_fields(ds):
    if ds in ["ottqa", "feta", "e2ewtq"]:
        return ("sheet_name", "file_name")
    return ("id",)


def make_key(item, kf):
    return "|".join(str(item.get(f, "")) for f in kf)


def load_graph_and_model(graph_path, model_path, kf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(graph_path, map_location=device, weights_only=False)

    if hasattr(data, 'metadata_maps') and 'table_meta' in data.metadata_maps:
        tid = {}
        for idx, m in enumerate(data.metadata_maps['table_meta']):
            k = make_key(m, kf)
            if k not in tid:
                tid[k] = idx
    else:
        try: tid = data.metadata_maps['table_id_to_idx']
        except: tid = {i: i for i in range(data['table'].x.size(0))}

    ckpt = torch.load(model_path, map_location=device)
    hps = ckpt.get('hps', {'HIDDEN_CHANNELS': 128, 'DROPOUT': 0.2})

    model = DiffusionModel(
        embed_dim=data['table'].x.size(1),
        hidden_channels=hps.get('HIDDEN_CHANNELS', 128),
        metadata=data.metadata(),
        dropout=hps.get('DROPOUT', 0.2),
        sage_aggr=hps.get('SAGE_AGGR', 'sum'),
        hetero_aggr=hps.get('HETERO_AGGR', 'sum'),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    return device, data, {v: str(k) for k, v in tid.items()}, model, hps


def parse_queries(qf, mk, kf):
    queries = []
    with open(qf, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get("questions", [None])[0] if "questions" in obj else obj.get("question")
            if not q: continue
            q = q.strip()
            gts = set()
            for gt in (obj.get("ground_truth_list") or []):
                if all(gt.get(f) is not None for f in kf):
                    k = make_key(gt, kf)
                    if k in mk: gts.add(k)
            if q: queries.append((q, gts))
    return queries


# ========= 指標 =========
def recall_at_k(ret, rel, k):
    if not rel: return 0.0
    return len(set(ret[:k]) & rel) / len(rel)

def rr(ret, rel):
    for i, r in enumerate(ret, 1):
        if r in rel: return 1.0 / i
    return 0.0

def ndcg(ret, rel, k):
    dcg = sum(1.0/math.log2(i+1) for i,r in enumerate(ret[:k],1) if r in rel)
    idcg = sum(1.0/math.log2(i+1) for i in range(1, min(len(rel),k)+1))
    return dcg/idcg if idcg > 0 else 0.0


# ========= 子圖 =========
def build_subgraph(data, q_vec, cands, device):
    """建立動態子圖 + query 節點（Query–Table 連邊）"""
    sub = HeteroData()
    cs = set(cands)
    nc = len(cands)
    t_map = {old: new for new, old in enumerate(cands)}

    col_s, pg_s = set(), set()
    for et, dst_set in [
        (('table','has_column','column'), col_s),
        (('table','comes_from','page'), pg_s)
    ]:
        if et in data.edge_types:
            ei = data[et].edge_index
            for i in range(ei.size(1)):
                if ei[0,i].item() in cs:
                    dst_set.add(ei[1,i].item())

    cols = sorted(col_s); pgs = sorted(pg_s)
    c_map = {o:n for n,o in enumerate(cols)}
    p_map = {o:n for n,o in enumerate(pgs)}

    sub['table'].x = data['table'].x[cands].to(device)
    if cols: sub['column'].x = data['column'].x[cols].to(device)
    if pgs: sub['page'].x = data['page'].x[pgs].to(device)

    def remap(et, sm, dm):
        if et not in data.edge_types: return
        ei = data[et].edge_index
        ns, nd = [], []
        for i in range(ei.size(1)):
            s,d = ei[0,i].item(), ei[1,i].item()
            if s in sm and d in dm: ns.append(sm[s]); nd.append(dm[d])
        if ns: sub[et].edge_index = torch.tensor([ns,nd], dtype=torch.long, device=device)

    remap(('table','has_column','column'), t_map, c_map)
    remap(('column','rev_has_column','table'), c_map, t_map)
    remap(('table','comes_from','page'), t_map, p_map)
    remap(('page','rev_comes_from','table'), p_map, t_map)
    for r in ['same_page','similar_table','shared_column_name']:
        remap(('table',r,'table'), t_map, t_map)
    remap(('column','similar_content','column'), c_map, c_map)

    # Query 節點 + Query–Table 邊
    sub['query'].x = q_vec.to(device)
    sub['query','queries','table'].edge_index = torch.tensor([[0]*nc, list(range(nc))], dtype=torch.long, device=device)
    sub['table','rev_queries','query'].edge_index = torch.tensor([list(range(nc)), [0]*nc], dtype=torch.long, device=device)

    return sub, t_map


def compute_metrics(retrieved, relevant):
    """計算一組查詢的所有指標"""
    return {
        'R@1': recall_at_k(retrieved, relevant, 1),
        'R@5': recall_at_k(retrieved, relevant, 5),
        'R@10': recall_at_k(retrieved, relevant, 10),
        'MRR': rr(retrieved, relevant),
        'nDCG@10': ndcg(retrieved, relevant, 10),
    }


def main():
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    all_results = {}

    embedder = get_embedder(device=('cuda' if torch.cuda.is_available() else 'cpu'))

    for source, dataset in DATASETS:
        kf = get_key_fields(dataset)
        qf = QUERY_FILE.format(source=source)
        mp = MODEL_PATH.format(dataset=dataset)
        gp = GRAPH_PATH.format(source=source)

        if not Path(gp).exists() or not Path(mp).exists():
            print(f"跳過 {source}", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"資料集: {source.upper()}", flush=True)
        print(f"{'='*60}", flush=True)

        device, data, idx_to_id, model, hps = load_graph_and_model(gp, mp, kf)

        # 過濾圖的連邊為該資料集的最佳配置
        best_edges = BEST_EDGE_CONFIGS.get(source, [])
        if best_edges:
            print(f"過濾連邊為最佳配置: {best_edges}", flush=True)
            original_edge_types = data.metadata()[1]
            data = filter_edges(data, best_edges)
            # 補上空邊以保持 metadata 一致
            for et in original_edge_types:
                if et not in data.edge_types:
                    data[et].edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

        queries = parse_queries(qf, set(idx_to_id.values()), kf)
        questions = [q for q,_ in queries]
        relevants = [g for _,g in queries]
        ec = sum(1 for g in relevants if g)
        print(f"查詢數: {len(queries)}, 可評估: {ec}", flush=True)

        qv = embedder.encode(questions, show_progress_bar=True, convert_to_tensor=True).to(device)
        qv = F.normalize(qv, p=2, dim=1)

        # === M0 ===
        print(f"\n--- M0: 固定表示 (Baseline) ---", flush=True)
        m0_metrics = {k: 0.0 for k in ['R@1','R@5','R@10','MRR','nDCG@10']}
        with torch.no_grad():
            t_emb = model.forward(data.x_dict, data.edge_index_dict)
            scores = torch.matmul(qv, t_emb.T)
            for qi in tqdm(range(len(queries)), desc="M0"):
                if not relevants[qi]: continue
                _, ti = torch.topk(scores[qi], k=min(TOP_K, scores.size(1)))
                ret = [idx_to_id.get(i.item(),"") for i in ti]
                m = compute_metrics(ret, relevants[qi])
                for k in m0_metrics: m0_metrics[k] += m[k]
        m0 = {k: v/ec for k,v in m0_metrics.items()}
        m0['eval_count'] = ec
        print(f"  R@1={m0['R@1']:.4f}  R@5={m0['R@5']:.4f}  R@10={m0['R@10']:.4f}  MRR={m0['MRR']:.4f}  nDCG@10={m0['nDCG@10']:.4f}", flush=True)

        # === M1 ===
        print(f"\n--- M1: 查詢參與式表示 (k={SUBGRAPH_K}) ---", flush=True)
        m1_metrics = {k: 0.0 for k in ['R@1','R@5','R@10','MRR','nDCG@10']}

        # 建立一次 query-aware 模型（固定 metadata）
        # 先用一個 dummy subgraph 取得 metadata
        dummy_cands = list(range(min(SUBGRAPH_K, data['table'].num_nodes)))
        dummy_qv = qv[0:1]
        dummy_sub, _ = build_subgraph(data, dummy_qv, dummy_cands, device)
        qa_metadata = dummy_sub.metadata()

        qa_model = DiffusionModel(
            embed_dim=data['table'].x.size(1),
            hidden_channels=hps.get('HIDDEN_CHANNELS', 128),
            metadata=qa_metadata,
            dropout=hps.get('DROPOUT', 0.2),
            sage_aggr=hps.get('SAGE_AGGR', 'sum'),
            hetero_aggr=hps.get('HETERO_AGGR', 'sum'),
        ).to(device)
        qa_model.load_state_dict(model.state_dict(), strict=False)
        qa_model.eval()
        del dummy_sub

        with torch.no_grad():
            coarse = torch.matmul(qv, t_emb.T)
            for qi in tqdm(range(len(queries)), desc="M1"):
                if not relevants[qi]: continue
                q = qv[qi:qi+1]

                _, ti = torch.topk(coarse[qi], k=min(SUBGRAPH_K, coarse.size(1)))
                cands = ti.cpu().tolist()

                sg, tm = build_subgraph(data, q, cands, device)

                # 確保子圖 metadata 與 qa_model 一致
                # 如果某些邊類型缺失，補上空邊
                for nt in qa_metadata[0]:
                    if nt not in sg.node_types:
                        sg[nt].x = torch.zeros(0, data['table'].x.size(1), device=device)
                for et in qa_metadata[1]:
                    if et not in sg.edge_types:
                        sg[et].edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

                xo = qa_model.hetero_sage(sg.x_dict, sg.edge_index_dict)
                if 'table' in xo and xo['table'].size(0) > 0:
                    xt = qa_model.norm(xo['table'])
                    ue = F.normalize(qa_model.proj_head(xt), p=2, dim=1)
                    rs = torch.matmul(q, ue.T).squeeze(0)
                    _, ri = torch.topk(rs, k=min(TOP_K, rs.size(0)))
                    n2o = {v:k for k,v in tm.items()}
                    ret = [idx_to_id.get(n2o[i.item()],"") for i in ri]
                else:
                    _, ti2 = torch.topk(coarse[qi], k=min(TOP_K, coarse.size(1)))
                    ret = [idx_to_id.get(i.item(),"") for i in ti2]

                del sg, xo
                m = compute_metrics(ret, relevants[qi])
                for k in m1_metrics: m1_metrics[k] += m[k]

                if qi % 20 == 0:
                    torch.cuda.empty_cache()

        m1 = {k: v/ec for k,v in m1_metrics.items()}
        m1['eval_count'] = ec
        print(f"  R@1={m1['R@1']:.4f}  R@5={m1['R@5']:.4f}  R@10={m1['R@10']:.4f}  MRR={m1['MRR']:.4f}  nDCG@10={m1['nDCG@10']:.4f}", flush=True)

        # 比較
        dr = m1['R@10'] - m0['R@10']
        dm = m1['MRR'] - m0['MRR']
        print(f"\n  Δ R@10: {'↑' if dr>0 else '↓'}{abs(dr):.4f}  |  Δ MRR: {'↑' if dm>0 else '↓'}{abs(dm):.4f}", flush=True)

        all_results[source] = {"M0": m0, "M1": m1}
        with open(f"{RESULT_DIR}/{source}.json", 'w') as f:
            json.dump({"source": source, "M0": m0, "M1": m1, "subgraph_k": SUBGRAPH_K}, f, indent=2)

        del data, model, qa_model, qv, t_emb, coarse
        gc.collect(); torch.cuda.empty_cache()

    del embedder
    torch.cuda.empty_cache()

    # ========= 彙總 =========
    print(f"\n\n{'='*100}", flush=True)
    print(f"消融實驗一：固定表示 (M0) vs 查詢參與式表示 (M1)  [subgraph_k={SUBGRAPH_K}]", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"{'Dataset':<12} {'M0 R@1':>8} {'M0 R@5':>8} {'M0 R@10':>9} {'M0 MRR':>8}  |  {'M1 R@1':>8} {'M1 R@5':>8} {'M1 R@10':>9} {'M1 MRR':>8}  |  {'ΔR@10':>7} {'ΔMRR':>7}", flush=True)
    print("-"*110, flush=True)

    s0r, s1r, s0m, s1m, cnt = 0,0,0,0,0
    for src in [d[0] for d in DATASETS]:
        if src not in all_results: continue
        a,b = all_results[src]["M0"], all_results[src]["M1"]
        dr = b['R@10']-a['R@10']; dm = b['MRR']-a['MRR']
        print(f"{src:<12} {a['R@1']:>8.4f} {a['R@5']:>8.4f} {a['R@10']:>9.4f} {a['MRR']:>8.4f}  |  {b['R@1']:>8.4f} {b['R@5']:>8.4f} {b['R@10']:>9.4f} {b['MRR']:>8.4f}  |  {'↑' if dr>0 else '↓'}{abs(dr):>6.4f} {'↑' if dm>0 else '↓'}{abs(dm):>6.4f}", flush=True)
        s0r+=a['R@10']; s1r+=b['R@10']; s0m+=a['MRR']; s1m+=b['MRR']; cnt+=1

    if cnt:
        print("-"*110, flush=True)
        dr = s1r/cnt - s0r/cnt; dm = s1m/cnt - s0m/cnt
        print(f"{'AVERAGE':<12} {'':>8} {'':>8} {s0r/cnt:>9.4f} {s0m/cnt:>8.4f}  |  {'':>8} {'':>8} {s1r/cnt:>9.4f} {s1m/cnt:>8.4f}  |  {'↑' if dr>0 else '↓'}{abs(dr):>6.4f} {'↑' if dm>0 else '↓'}{abs(dm):>6.4f}", flush=True)
    print(f"{'='*100}", flush=True)

    with open(f"{RESULT_DIR}/summary.json", 'w') as f:
        json.dump({"subgraph_k": SUBGRAPH_K, "results": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\n結果已儲存至 {RESULT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
