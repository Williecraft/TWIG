# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TabGNN is a **table retrieval system** using heterogeneous Graph Neural Networks (GNNs). Given a natural language query, it retrieves the most relevant tables from a multi-table corpus. The pipeline is: raw table data → build heterogeneous graph → train GNN → evaluate retrieval.

## Common Commands

All scripts live under `reports/src/`. The project venv is at the repo root.

```bash
# Activate venv
source .venv/bin/activate

# 1. Build graph (must run before training)
cd reports/src && python build_graph.py

# 2. Train base TWIG model
cd reports/src && python train_model.py

# 3. Evaluate retrieval
cd reports/src && python evaluate_retrieval.py

# 4. Base TWIG edge ablation (A0–A63)
cd reports/src && python run_edge_ablation.py --datasets feta --gpu 0 --ablation $(seq 0 63)

# 5. Query-Aware v2 pipeline (train + evaluate)
cd reports/src/query_aware && python run_pipeline_v2.py

# 6. QA v2 edge ablation
cd reports/src/query_aware && python run_qa_edge_ablation.py --datasets feta --gpu 0 --ablation $(seq 0 63)

# 7. Cross-dataset generalization test
cd reports/src/query_aware && python run_cross_dataset_eval.py --gpu 0

# 8. Cross-validation
cd reports/src && python cross_validate.py

# 9. Run tests (must be run from reports/src/ so imports resolve)
cd reports/src && python -m pytest ../../tests/test_training.py ../../tests/test_retrieval.py
```

## Data Layout

```
data/
  table/{split}/{dataset}/
    table.jsonl   # raw tables (header, instances, metadata)
    query.jsonl   # queries with ground_truth_list
  processed/{split}/{dataset}/
    graph.pt      # HeteroData saved by build_graph.py
checkpoints/{dataset}/
  model.pt                  # base TWIG model
  model_query_aware_v2.pt   # QA v2 model
results/
  edge_ablation_extended/   # base TWIG ablation results
  qa_edge_ablation/         # QA v2 ablation results
  cross_dataset_eval/       # cross-dataset generalization results
abandon/                    # deprecated scripts (do not use)
reports/src/utils/          # data processing utilities, LaTeX table generators
```

Supported datasets: `feta`, `ottqa`, `e2ewtq`, `mimo_en`, `mimo_ch`, `mmqa`.

**Splits**: `train`, `dev`, `test`. The active DATASETS/SOURCES lists inside each script control which datasets are processed — comment/uncomment as needed.

## Architecture

### Graph Construction (`reports/src/build_graph.py`)

Builds a `HeteroData` (PyG) with three node types:
- **table** — embedding of page title + sheet name + column names + first 5 rows
- **column** — embedding of column name + sample values
- **page** — embedding of page title

Six edge types (used in ablation with 6-bit binary encoding):
| Bit | Name | Description |
|-----|------|-------------|
| 5 | `similar_table` (tt) | Top-K embedding similarity between tables |
| 4 | `has_column` (tc) | Table → its columns (structural) |
| 3 | `comes_from` (tp) | Table → its page (structural) |
| 2 | `same_page` (sp) | Tables on the same Wikipedia page |
| 1 | `similar_content` (cc) | Top-K embedding similarity between columns |
| 0 | `shared_column_name` (sc) | Tables sharing identical column names |

All edges also have `rev_*` reverse edges added automatically. The graph is saved with `torch.save` including `data.metadata_maps` (`table_id_to_idx`, `table_meta`, `key_fields`).

### Table Key Fields

Key uniqueness depends on dataset type:
- `ottqa`, `feta`, `e2ewtq` → key = `sheet_name|file_name`
- `mimo_en`, `mimo_ch`, `mmqa` → key = `id`

### Model (`reports/src/train_model.py` — class `DiffusionModel`)

A 2-layer `GraphSAGE` converted to heterogeneous via `to_hetero`, followed by `GraphNorm` and a 2-layer MLP projection head. Output is L2-normalized table embeddings.

Training uses:
- InfoNCE (cross-entropy with temperature annealing) loss over batch
- Hard negative margin loss with periodically re-mined top-K negatives
- AdamW + cosine LR scheduler with linear warmup
- Mixed precision (`torch.cuda.amp`)
- Early stopping on validation Recall@10

Key hyperparameters in `BEST_PARAMS` (at top of `train_model.py`):
- `HIDDEN_CHANNELS`: 768
- `SAGE_AGGR`: `'min'`, `HETERO_AGGR`: `'max'`
- `LEARNING_RATE`: ~5.5e-4, `DROPOUT`: ~0.10, `WEIGHT_DECAY`: ~0.032

Set `USE_HYPEROPT = True` to run Bayesian hyperparameter search via `hyperopt` (up to `MAX_EVALS = 300` trials).

Embedding model: `BAAI/bge-m3` (via `sentence-transformers`).

### Evaluation (`reports/src/evaluate_retrieval.py`)

Loads a trained model + test graph, embeds queries with the same `bge-m3` encoder, runs GNN forward pass, computes cosine similarity, and reports Recall@1/5/10, MRR, nDCG@5/10, Precision@5, Full Recall@5.

`evaluate_retrieval.py` imports `DiffusionModel` and `get_embedder` from `train_model.py` — must be run from `reports/src/` for the relative import to resolve. `retrieval.py` is a thin wrapper that exposes the same classes for use as a library.

### Query-Aware v2 (`reports/src/query_aware/`)

Conditions the GNN on the query embedding via additional query→table/column/page edges. Two-phase approach:
1. Load pretrained TWIG checkpoint
2. Fine-tune `QueryAwareModel` with query edges zero-initialized and differential learning rates (base LR 1e-4, query-edge LR higher)

Per-dataset best edge configs are hardcoded in `BEST_EDGE_CONFIGS` at the top of `train_query_aware_v2.py`.

Key scripts:
- `train_query_aware_v2.py` — Fine-tuning with subgraph-based QA training
- `evaluate_query_aware_v2.py` — Evaluation with coarse→rerank pipeline
- `run_pipeline_v2.py` — End-to-end train + evaluate wrapper
- `run_qa_edge_ablation.py` — QA v2 edge ablation (A0–A63)
- `run_cross_dataset_eval.py` — Cross-dataset generalization (N×N matrix)

### Ablation Encoding

Edge configs use 6-bit binary: bits 5–0 map to tt/tc/tp/sp/cc/sc. A0 = no edges, A63 = all edges. Pass `--ablation $(seq 0 63)` to sweep all 64 configs. Set GPU with `--gpu <id>` — this sets `CUDA_VISIBLE_DEVICES` before torch imports.

### Cross-Validation (`reports/src/cross_validate.py`)

K-fold (default K=5) runner that uses DSU union to keep all gold tables for a query in the same fold, then runs build → train → evaluate per fold.

## Dependencies

```
torch, torch_geometric
sentence-transformers   # bge-m3 embedder
pandas, tabulate, matplotlib
hyperopt                # Bayesian hyperparam search
google-genai            # optional, used in abandoned scripts only
```

Install: `pip install -r requirements.txt`

## Paper Revision (SIGIR Reject → Resubmit)

The paper was rejected from SIGIR. Full reviewer comments are in [Reviewer_中文整理.md](Reviewer_中文整理.md). When working on experiments or writing, address these open issues:

### P0 — Must fix before resubmission

- **Complete Method section**: every node type, edge type, formula, symbol, and similarity computation must be defined precisely enough for reproduction. Reviewer s7TT specifically flagged Section 3.1.1 for incomplete set/range definitions.
- **Fix edge configuration selection protocol**: the current approach uses ablation results to pick the best edge config per dataset, then tests on the same data — this is data leakage. Selection must be based solely on the **validation set**. Report both a fixed configuration and the validation-tuned configuration side by side.
- **Add strong baselines**: the comparison currently lacks graph-aware, structure-aware, and connectivity-aware retrieval methods. The nearest prior work must be compared fairly.
- **Moderate Results claims**: TWIG does not outperform baselines on most datasets. Results section must clearly distinguish datasets where TWIG wins, ties, and loses, and discuss why.

### P1 — Directly affects persuasiveness

- **Efficiency numbers are missing**: the paper claims efficiency advantage over LLM-based pipelines, but provides no data. Must report: graph construction time, similarity-edge computation time, indexing time, memory usage, graph storage size, training time, query latency — and how these scale with corpus size.
- **Dataset characteristic analysis**: explain which properties (table count, query type, page overlap, etc.) correlate with TWIG's gains or failures.
- **Clarify novelty**: explicitly state what is new compared to existing heterogeneous graph retrieval and relation-aware GraphSAGE work.

### P2 — Writing and presentation

- Move method description currently placed in Related Work into the Method section.
- Expand Related Work beyond brief enumeration of prior work.
- Add dataset source, scale, split sizes, and characteristics to the Evaluation section.
- **Fix Table 1 bolding**: MMQA R@2 `52.24` should be bold, not `51.40`.
- Ensure all tables, symbols, and figure captions are self-contained.
