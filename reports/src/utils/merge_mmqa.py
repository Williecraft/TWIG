import os
import json
import re
from difflib import SequenceMatcher
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

# ================== 全域變數：你只要改這裡 ==================
dataset1_path = "/user_data/TabGNN/data/downloads/mmqa2"   # 這個資料夾底下有 table.jsonl / query.jsonl
dataset2_path = "/user_data/TabGNN/data/downloads/mmqa3"   # 這個資料夾底下有 table.jsonl / query.jsonl
out_dataset = "mmqa_merged"  # 輸出資料夾名稱（會在 dataset1_path 同層創建）

SIM_TABLE = 0.95
SIM_QUERY = 0.95
# ===========================================================


_WS = re.compile(r"\s+")


def norm_text(s: str) -> str:
    return _WS.sub(" ", (s or "").strip().lower())


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Failed to parse JSONL: {path} at line {line_no}: {e}")
    return items


def table_rep(table_obj: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (header_str, rep_string) for table dedup.
    """
    header_list = table_obj.get("header", [])
    header_str = header_list[0] if isinstance(header_list, list) and header_list else ""
    instances = table_obj.get("instances", [])
    if not isinstance(instances, list):
        instances = []
    rep = norm_text(header_str + "\n" + "\n".join(map(str, instances)))
    return header_str, rep


def query_rep(query_obj: Dict[str, Any]) -> str:
    q = query_obj.get("question", "")
    a = query_obj.get("answer", "")
    return norm_text(str(q) + "\n" + str(a))


def main():
    # 輸出位置：與 dataset1_path 同層
    base_dir = os.path.dirname(os.path.abspath(dataset1_path.rstrip("/")))
    out_dir = os.path.join(base_dir, out_dataset)
    os.makedirs(out_dir, exist_ok=True)

    out_table_path = os.path.join(out_dir, "table.jsonl")
    out_query_path = os.path.join(out_dir, "query.jsonl")

    # 讀兩個資料集
    dsets = [
        ("d1", dataset1_path),
        ("d2", dataset2_path),
    ]

    # 先把兩邊 tables 全讀進來，建立 old_id -> table_obj
    tables_by_ds: Dict[str, Dict[int, Dict[str, Any]]] = {}
    queries_by_ds: Dict[str, List[Dict[str, Any]]] = {}

    for tag, p in dsets:
        tpath = os.path.join(p, "table.jsonl")
        qpath = os.path.join(p, "query.jsonl")
        if not os.path.exists(tpath):
            raise FileNotFoundError(f"Missing table.jsonl: {tpath}")
        if not os.path.exists(qpath):
            raise FileNotFoundError(f"Missing query.jsonl: {qpath}")

        t_items = read_jsonl(tpath)
        q_items = read_jsonl(qpath)

        tb = {}
        for t in t_items:
            old_id = t.get("id")
            if not isinstance(old_id, int):
                # 若 id 是字串也試著轉 int
                try:
                    old_id = int(old_id)
                except Exception:
                    continue
            tb[old_id] = t

        tables_by_ds[tag] = tb
        queries_by_ds[tag] = q_items

    # Dedup 索引
    # Table：先用 header 分桶，桶內才做 SequenceMatcher
    seen_tables_by_header: Dict[str, List[Tuple[str, int]]] = {}  # header_str -> [(rep, new_table_id)]
    # 可選：rep_len 剪枝用
    seen_table_rep_len: Dict[int, int] = {}  # new_table_id -> len(rep)

    # Query：全部放同一桶（也可以用前綴分桶，但先簡單）
    seen_queries: List[Tuple[str, int]] = []  # [(rep, new_query_id)]
    seen_query_rep_len: Dict[int, int] = {}   # new_query_id -> len(rep)

    next_table_id = 0
    next_query_id = 0

    # 為了讓 query 能快速把舊 table 轉成新 table id
    # key: (ds_tag, old_table_id) -> new_table_id
    old_to_new_table: Dict[Tuple[str, int], int] = {}

    def get_or_add_table(ds_tag: str, old_table_id: int) -> Optional[int]:
        nonlocal next_table_id

        key = (ds_tag, old_table_id)
        if key in old_to_new_table:
            return old_to_new_table[key]

        table_obj = tables_by_ds[ds_tag].get(old_table_id)
        if table_obj is None:
            print(f"[WARN] Missing table for {ds_tag} old_id={old_table_id}, skipped reference")
            return None

        header_str, rep = table_rep(table_obj)
        rep_len = len(rep)

        bucket = seen_tables_by_header.get(header_str, [])
        matched_new_id = None

        # 桶內比對 + 簡單剪枝
        for prev_rep, prev_new_id in bucket:
            prev_len = seen_table_rep_len.get(prev_new_id, len(prev_rep))
            if prev_len == 0 or rep_len == 0:
                continue
            if abs(prev_len - rep_len) / max(prev_len, rep_len) > 0.20:
                continue
            if sim(rep, prev_rep) >= SIM_TABLE:
                matched_new_id = prev_new_id
                break

        if matched_new_id is not None:
            old_to_new_table[key] = matched_new_id
            return matched_new_id

        # 新增 table：換新 id
        new_id = next_table_id
        next_table_id += 1

        out_t = dict(table_obj)
        out_t["id"] = new_id

        # 寫出
        ft.write(json.dumps(out_t, ensure_ascii=False) + "\n")

        seen_tables_by_header.setdefault(header_str, []).append((rep, new_id))
        seen_table_rep_len[new_id] = rep_len
        old_to_new_table[key] = new_id
        return new_id

    def is_dup_query(rep: str) -> bool:
        rep_len = len(rep)
        for prev_rep, prev_qid in seen_queries:
            prev_len = seen_query_rep_len.get(prev_qid, len(prev_rep))
            if prev_len == 0 or rep_len == 0:
                continue
            # query 可以更寬鬆剪枝（避免漏掉很像的）
            if abs(prev_len - rep_len) / max(prev_len, rep_len) > 0.35:
                continue
            if sim(rep, prev_rep) >= SIM_QUERY:
                return True
        return False

    # 合併順序：先 dataset1，再 dataset2
    all_queries = []
    for tag, _ in dsets:
        for q in queries_by_ds[tag]:
            all_queries.append((tag, q))

    with open(out_table_path, "w", encoding="utf-8") as ft, \
         open(out_query_path, "w", encoding="utf-8") as fq:

        # 讓內部 helper 用到檔案 handle
        #（Python 閉包需要先定義 ft 才能用，所以 helper 定義在 open 之後）
        # -> 上面 helper 用到了 ft，所以我們把 helper 放在 open 之後不方便；
        #    因此在這裡用一個小技巧：把 ft 變成外層變數
        pass

    # 重新開一次（為了讓 helper 能用到 ft/fq）
    with open(out_table_path, "w", encoding="utf-8") as ft, \
         open(out_query_path, "w", encoding="utf-8") as fq:

        # 重新宣告 helper（綁定到這次的 ft/fq）
        def get_or_add_table(ds_tag: str, old_table_id: int) -> Optional[int]:
            nonlocal next_table_id

            key = (ds_tag, old_table_id)
            if key in old_to_new_table:
                return old_to_new_table[key]

            table_obj = tables_by_ds[ds_tag].get(old_table_id)
            if table_obj is None:
                print(f"[WARN] Missing table for {ds_tag} old_id={old_table_id}, skipped reference")
                return None

            header_str, rep = table_rep(table_obj)
            rep_len = len(rep)

            bucket = seen_tables_by_header.get(header_str, [])
            matched_new_id = None

            for prev_rep, prev_new_id in bucket:
                prev_len = seen_table_rep_len.get(prev_new_id, len(prev_rep))
                if prev_len == 0 or rep_len == 0:
                    continue
                if abs(prev_len - rep_len) / max(prev_len, rep_len) > 0.20:
                    continue
                if sim(rep, prev_rep) >= SIM_TABLE:
                    matched_new_id = prev_new_id
                    break

            if matched_new_id is not None:
                old_to_new_table[key] = matched_new_id
                return matched_new_id

            new_id = next_table_id
            next_table_id += 1

            out_t = dict(table_obj)
            out_t["id"] = new_id
            ft.write(json.dumps(out_t, ensure_ascii=False) + "\n")

            seen_tables_by_header.setdefault(header_str, []).append((rep, new_id))
            seen_table_rep_len[new_id] = rep_len
            old_to_new_table[key] = new_id
            return new_id

        def is_dup_query(rep: str) -> bool:
            rep_len = len(rep)
            for prev_rep, prev_qid in seen_queries:
                prev_len = seen_query_rep_len.get(prev_qid, len(prev_rep))
                if prev_len == 0 or rep_len == 0:
                    continue
                if abs(prev_len - rep_len) / max(prev_len, rep_len) > 0.35:
                    continue
                if sim(rep, prev_rep) >= SIM_QUERY:
                    return True
            return False

        # 開始處理 queries（同時按需把 table 寫出）
        for ds_tag, q in tqdm(all_queries, desc="Merging queries"):
            rep = query_rep(q)
            if is_dup_query(rep):
                continue

            new_qid = next_query_id
            next_query_id += 1

            # 轉換 ground_truth_list -> 新 table ids
            gt = q.get("ground_truth_list", [])
            new_gt = []
            if isinstance(gt, list):
                for g in gt:
                    if not isinstance(g, dict):
                        continue
                    old_tid = g.get("id")
                    if old_tid is None:
                        continue
                    try:
                        old_tid = int(old_tid)
                    except Exception:
                        continue

                    new_tid = get_or_add_table(ds_tag, old_tid)
                    if new_tid is None:
                        continue

                    # file_name / sheet_name：以原本 query 的 g 為主，沒有就用 table 內的
                    file_name = g.get("file_name")
                    sheet_name = g.get("sheet_name")

                    if file_name is None or sheet_name is None:
                        t_obj = tables_by_ds[ds_tag].get(old_tid, {})
                        file_name = file_name or t_obj.get("file_name", "")
                        sheet_name = sheet_name or t_obj.get("sheet_name", "")

                    new_gt.append({
                        "id": new_tid,
                        "file_name": file_name,
                        "sheet_name": sheet_name,
                    })

            out_q = dict(q)
            out_q["id"] = new_qid
            out_q["ground_truth_list"] = new_gt

            fq.write(json.dumps(out_q, ensure_ascii=False) + "\n")

            seen_queries.append((rep, new_qid))
            seen_query_rep_len[new_qid] = len(rep)

    print("✅ Done")
    print(f"Output dir: {out_dir}")
    print(f"Unique tables: {next_table_id}")
    print(f"Unique queries: {next_query_id}")
    print(out_table_path)
    print(out_query_path)


if __name__ == "__main__":
    main()
