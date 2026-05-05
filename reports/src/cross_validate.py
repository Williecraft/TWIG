# cross_validate.py
# -*- coding: utf-8 -*-
"""
K-fold cross validation runner for TabGNN project.

Pipeline per fold:
  1) build_graph on train split
  2) train_model using train graph + train queries
  3) build_graph on test split
  4) evaluate_retrieval using test graph + test queries + trained model

Splitting rule:
  - Use DSU union over tables that co-occur in the same query's ground_truth_list,
    ensuring all gold tables for a query stay in the same fold.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple


K_FOLD = 5
SOURCE = "mimo_en"
SEED = 42
# 唯一鍵欄位（可選: "id", "file_name", "sheet_name" 的任意組合）
KEY_FIELDS = ("file_name", "sheet_name")


# -------------------------
# I/O helpers (same spirit as split_dataset.py)
# -------------------------
def read_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def make_key(item: dict, key_fields: tuple) -> str:
    """根據 key_fields 組合產生唯一鍵"""
    return "|".join(str(item.get(f, "")) for f in key_fields)


def table_key(t: dict) -> str:
    """Return composite key determined by KEY_FIELDS."""
    return make_key(t, KEY_FIELDS)


def tables_keys_in_original_order(tables: List[dict], key_set: Set[str]) -> List[str]:
    out: List[str] = []
    for t in tables:
        tk = table_key(t)
        if tk in key_set:
            out.append(tk)
    return out


# -------------------------
# DSU (Union-Find) (adapted from split_dataset.py)
# -------------------------
class DSU:
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.size: Dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def find(self, x: str) -> str:
        p = self.parent[x]
        while p != self.parent[p]:
            p = self.parent[p]
        while x != p:
            nxt = self.parent[x]
            self.parent[x] = p
            x = nxt
        return p

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def groups(self) -> Dict[str, List[str]]:
        g: Dict[str, List[str]] = {}
        for x in self.parent.keys():
            r = self.find(x)
            g.setdefault(r, []).append(x)
        return g


# -------------------------
# K-fold split with DSU constraint
# -------------------------
def build_components(name: str, in_dir: Path) -> Tuple[List[dict], List[dict], Dict[str, dict], List[Tuple[int, List[str]]]]:
    table_path = in_dir / "table.jsonl"
    query_path = in_dir / "query.jsonl"

    tables = read_jsonl(table_path)
    queries = read_jsonl(query_path)

    table_by_key: Dict[str, dict] = {}
    for t in tables:
        tk = table_key(t)
        if tk in table_by_key:
            raise ValueError(f"[{name}] table.jsonl key 重複: {tk}")
        table_by_key[tk] = t

    dsu = DSU()
    for tk in table_by_key.keys():
        dsu.add(tk)

    missing_gold_tables = 0
    for q in queries:
        gtl = q.get("ground_truth_list", []) or []
        gold_keys: List[str] = []
        for gt in gtl:
            # 檢查所有 key fields 是否存在（允許空字串，但不允許 None）
            if not all(gt.get(f) is not None for f in KEY_FIELDS):
                continue
            gk = make_key(gt, KEY_FIELDS)
            if gk not in table_by_key:
                missing_gold_tables += 1
                continue
            gold_keys.append(gk)

        if len(gold_keys) >= 2:
            base = gold_keys[0]
            for other in gold_keys[1:]:
                dsu.union(base, other)

    groups = dsu.groups()
    comps: List[Tuple[int, List[str]]] = [(len(members), members) for members in groups.values()]

    print(f"[{name}] tables={len(tables)} queries={len(queries)} components={len(comps)} missing_gold_refs={missing_gold_tables}")
    return tables, queries, table_by_key, comps


def assign_components_to_folds(
    comps: List[Tuple[int, List[str]]],
    k: int,
    seed: int = SEED
) -> List[Set[str]]:
    """
    Assign DSU components to k folds while balancing by total table count.
    Greedy: shuffle comps, then put next comp into fold with smallest current size.
    """
    rng = random.Random(seed)
    comps_shuffled = comps[:]
    rng.shuffle(comps_shuffled)

    fold_tables: List[Set[str]] = [set() for _ in range(k)]
    fold_sizes = [0] * k

    for sz, members in comps_shuffled:
        j = min(range(k), key=lambda idx: fold_sizes[idx])
        fold_tables[j].update(members)
        fold_sizes[j] += sz

    for i in range(k):
        print(f"  fold {i}: {fold_sizes[i]} tables")
    return fold_tables


def write_fold_split(
    name: str,
    tables: List[dict],
    queries: List[dict],
    table_by_key: Dict[str, dict],
    fold_tables: List[Set[str]],
    out_base_dir: Path
) -> None:
    """
    For each fold i:
      test_tables = fold_tables[i]
      train_tables = all others
    Then split queries by their first valid gold table key (file_name|sheet_name).
    """
    all_table_keys = set(table_by_key.keys())

    for i in range(len(fold_tables)):
        test_keys = set(fold_tables[i])
        train_keys = all_table_keys - test_keys

        fold_dir = out_base_dir / f"fold_{i}"
        out_train_dir = fold_dir / "train"
        out_test_dir = fold_dir / "test"

        out_train_tables = [table_by_key[tk] for tk in tables_keys_in_original_order(tables, train_keys)]
        out_test_tables = [table_by_key[tk] for tk in tables_keys_in_original_order(tables, test_keys)]

        out_train_queries: List[dict] = []
        out_test_queries: List[dict] = []
        bad_queries = 0

        for q in queries:
            gtl = q.get("ground_truth_list", []) or []
            gold_keys = []
            for gt in gtl:
                # 檢查所有 key fields 是否存在（允許空字串，但不允許 None）
                if not all(gt.get(f) is not None for f in KEY_FIELDS):
                    continue
                gk = make_key(gt, KEY_FIELDS)
                if gk in table_by_key:
                    gold_keys.append(gk)

            if not gold_keys:
                bad_queries += 1
                out_test_queries.append(q)
                continue

            first = gold_keys[0]
            if first in train_keys:
                out_train_queries.append(q)
            elif first in test_keys:
                out_test_queries.append(q)
            else:
                bad_queries += 1
                out_test_queries.append(q)

        write_jsonl(out_train_dir / "table.jsonl", out_train_tables)
        write_jsonl(out_test_dir / "table.jsonl", out_test_tables)
        write_jsonl(out_train_dir / "query.jsonl", out_train_queries)
        write_jsonl(out_test_dir / "query.jsonl", out_test_queries)

        print(f"[{name}] fold_{i} -> train_tables={len(out_train_tables)} test_tables={len(out_test_tables)} "
              f"train_queries={len(out_train_queries)} test_queries={len(out_test_queries)} bad_queries={bad_queries}")
        print(f"  train_out: {out_train_dir}")
        print(f"  test_out : {out_test_dir}")


# -------------------------
# Run pipeline by monkey-patching existing scripts
# -------------------------
def run_build_graph(source_rel: str, out_graph_path: Path) -> None:
    import build_graph  # user provided
    # Patch module-level variables (because build_graph.py uses constants evaluated at import time)
    build_graph.SOURCE = source_rel
    build_graph.TABLE_JSONL_PATH = f"/user_data/TabGNN/data/table/{source_rel}/table.jsonl"
    build_graph.OUTPUT_GRAPH_PATH = str(out_graph_path)
    # Keep other params (MODEL_NAME, DEVICE, BATCH_SIZE, K_TABLE, K_COLUMN) as-is unless you want to patch too.
    print(f"\n[build_graph] SOURCE={source_rel}")
    print(f"[build_graph] TABLE_JSONL_PATH={build_graph.TABLE_JSONL_PATH}")
    print(f"[build_graph] OUTPUT_GRAPH_PATH={build_graph.OUTPUT_GRAPH_PATH}")
    build_graph.main()


def run_train_model(graph_path: Path, query_path: Path, save_model_path: Path) -> None:
    import train_model  # user provided
    train_model.GRAPH_FILE = str(graph_path)
    train_model.QUERY_FILE = str(query_path)
    train_model.SAVE_PATH = str(save_model_path)

    print(f"\n[train_model] GRAPH_FILE={train_model.GRAPH_FILE}")
    print(f"[train_model] QUERY_FILE={train_model.QUERY_FILE}")
    print(f"[train_model] SAVE_PATH={train_model.SAVE_PATH}")
    train_model.main()


def run_evaluate(query_path: Path, graph_path: Path, model_path: Path) -> dict:
    import evaluate_retrieval  # user provided
    evaluate_retrieval.QUERY_FILE = str(query_path)
    evaluate_retrieval.GRAPH_PATH = str(graph_path)
    evaluate_retrieval.MODEL_PATH = str(model_path)

    print(f"\n[evaluate] QUERY_FILE={evaluate_retrieval.QUERY_FILE}")
    print(f"[evaluate] GRAPH_PATH={evaluate_retrieval.GRAPH_PATH}")
    print(f"[evaluate] MODEL_PATH={evaluate_retrieval.MODEL_PATH}")

    # evaluate_retrieval.evaluate() prints metrics; we also want the returned dict.
    # Your evaluate_retrieval.py currently prints but doesn't return results.
    # We'll re-run the computation by capturing prints is annoying; instead:
    # - If evaluate() returns None, we still log "None".
    try:
        ret = evaluate_retrieval.evaluate(
            query_file=evaluate_retrieval.QUERY_FILE,
            model_path=evaluate_retrieval.MODEL_PATH,
            graph_path=evaluate_retrieval.GRAPH_PATH,
            top_k=getattr(evaluate_retrieval, "TOP_K", 10),
        )
        return {"returned": ret}
    except TypeError:
        # fallback: if signature differs
        ret = evaluate_retrieval.evaluate()
        return {"returned": ret}


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=SOURCE, help="dataset folder name under /user_data/TabGNN/data/table/train/")
    parser.add_argument("--k", type=int, default=K_FOLD, help="number of folds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in_dir", default=None, help="override input dir (must contain table.jsonl & query.jsonl)")
    args = parser.parse_args()

    dataset = args.dataset
    k = args.k
    seed = args.seed

    # Input dataset directory
    if args.in_dir:
        in_dir = Path(args.in_dir)
    else:
        in_dir = Path(f"/user_data/TabGNN/data/table/train/{dataset}")

    if not (in_dir / "table.jsonl").exists() or not (in_dir / "query.jsonl").exists():
        raise FileNotFoundError(f"Input dir missing table.jsonl/query.jsonl: {in_dir}")

    # Output base directory (what you requested)
    out_base_dir = Path(f"/user_data/TabGNN/data/table/train/{dataset}_CV/{k}")
    out_base_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build DSU components and create folds
    tables, queries, table_by_id, comps = build_components(dataset, in_dir)
    fold_tables = assign_components_to_folds(comps, k=k, seed=seed)
    write_fold_split(dataset, tables, queries, table_by_id, fold_tables, out_base_dir)

    # 2) Run CV pipeline per fold
    processed_base = Path(f"/user_data/TabGNN/data/processed/{dataset}_CV/{k}")
    ckpt_base = Path(f"/user_data/TabGNN/checkpoints/{dataset}_CV/{k}")
    results_dir = Path(f"/user_data/TabGNN/results/{dataset}_CV/{k}")

    processed_base.mkdir(parents=True, exist_ok=True)
    ckpt_base.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_jsonl = results_dir / "cv_results.jsonl"
    summary_json = results_dir / "cv_summary.json"

    all_fold_rows: List[dict] = []

    # Clear previous results
    if results_jsonl.exists():
        results_jsonl.unlink()

    for i in range(k):
        t0 = time.time()
        fold_dir = out_base_dir / f"fold_{i}"
        train_dir = fold_dir / "train"
        test_dir = fold_dir / "test"

        # SOURCE string used by build_graph.py: relative under data/table/
        train_source_rel = f"train/{dataset}_CV/{k}/fold_{i}/train"
        test_source_rel = f"train/{dataset}_CV/{k}/fold_{i}/test"

        # Graph/model paths per fold
        graph_train_path = processed_base / f"fold_{i}_graph_train.pt"
        graph_test_path = processed_base / f"fold_{i}_graph_test.pt"
        model_path = ckpt_base / f"fold_{i}_model.pt"

        print("\n" + "=" * 80)
        print(f"[CV] Fold {i+1}/{k}")
        print("=" * 80)

        # (A) build_graph(train)
        run_build_graph(train_source_rel, graph_train_path)

        # (B) train_model(train)
        run_train_model(
            graph_path=graph_train_path,
            query_path=train_dir / "query.jsonl",
            save_model_path=model_path
        )

        # (C) build_graph(test)
        run_build_graph(test_source_rel, graph_test_path)

        # (D) evaluate(test)
        eval_ret = run_evaluate(
            query_path=test_dir / "query.jsonl",
            graph_path=graph_test_path,
            model_path=model_path
        )

        row = {
            "fold": i,
            "train_source": train_source_rel,
            "test_source": test_source_rel,
            "graph_train": str(graph_train_path),
            "graph_test": str(graph_test_path),
            "model": str(model_path),
            "eval": eval_ret,
            "elapsed_sec": round(time.time() - t0, 3),
        }
        all_fold_rows.append(row)

        # append to jsonl
        with results_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 3) Write summary
    summary = {
        "dataset": dataset,
        "k": k,
        "seed": seed,
        "out_base_dir": str(out_base_dir),
        "processed_base": str(processed_base),
        "ckpt_base": str(ckpt_base),
        "results_dir": str(results_dir),
        "folds": all_fold_rows,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("[CV] Done.")
    print(f"Results jsonl : {results_jsonl}")
    print(f"Summary json  : {summary_json}")
    print("=" * 80)

    # -------------------------
    # 4) Print CV mean results
    # -------------------------
    print("\n" + "=" * 80)
    print("[CV] Average Results over all folds")
    print("=" * 80)

    # 收集所有 metric
    metric_values: Dict[str, List[float]] = {}

    for row in all_fold_rows:
        eval_ret = row.get("eval", {}).get("returned")
        if not eval_ret:
            continue

        for k, v in eval_ret.items():
            if k in ("eval_count", "total"):
                continue
            if v is None:
                continue
            metric_values.setdefault(k, []).append(float(v))

    # 印出平均（可順便印 std）
    for metric, values in metric_values.items():
        if not values:
            continue
        mean_v = sum(values) / len(values)
        # 若你想要 std，可打開下面三行
        # var = sum((x - mean_v) ** 2 for x in values) / len(values)
        # std_v = var ** 0.5
        # print(f"{metric}: {mean_v:.4f} ± {std_v:.4f}")

        print(f"{metric}: {mean_v:.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
