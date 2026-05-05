import json
import pandas as pd
from pathlib import Path

RESULT_DIR = Path("/user_data/TabGNN/results/evaluate_query_aware_final")
OUTPUT_CSV = "/user_data/TabGNN/results/query_aware_summary.csv"

datasets = ["mimo_en", "mimo_ch", "ottqa", "feta", "e2ewtq", "mmqa"]
modes = ["E0", "E1", "E2", "E3", "E4"]
ks = [10, 20]

records = []

for ds in datasets:
    for k in ks:
        for m in modes:
            # E0 only runs for k=10
            if m == "E0" and k != 10:
                continue
                
            fpath = RESULT_DIR / f"{ds}_{m}_k{k}.json"
            if not fpath.exists():
                print(f"Missing results: {fpath}")
                continue
                
            with open(fpath, "r") as f:
                data = json.load(f)
            
            res = data.get("results", {})
            if "total" not in res:
                # Fallback if eval failed
                continue
                
            records.append({
                "Dataset": ds,
                "k": k if m != "E0" else "-",
                "Mode": m,
                "Recall@10": res.get("Recall@10", 0.0),
                "MRR": res.get("MRR@k", 0.0),
                "eval_count": res.get("eval_count", 0),
            })

df = pd.DataFrame(records)

# Pivot to compare R@10 easily
# R@10 Table
try:
    pivot_r10 = df.pivot(index="Dataset", columns=["k", "Mode"], values="Recall@10")
    print("\n--- Recall@10 Summary ---")
    print(pivot_r10.round(4).to_string())
except Exception as e:
    print(f"Pivot R10 failed: {e}")

# MRR Table
try:
    pivot_mrr = df.pivot(index="Dataset", columns=["k", "Mode"], values="MRR")
    print("\n--- MRR Summary ---")
    print(pivot_mrr.round(4).to_string())
except Exception as e:
    pass

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved raw summary to {OUTPUT_CSV}")
