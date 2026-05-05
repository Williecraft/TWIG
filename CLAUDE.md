# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TabGNN is a **table retrieval system** using heterogeneous Graph Neural Networks (GNNs). Given a natural language query, it retrieves the most relevant tables from a multi-table corpus. The pipeline is: raw table data → build heterogeneous graph → train GNN → evaluate retrieval.

## Common Commands

All scripts are run from `src/` directory or with the full path. The project uses a `.venv` at the repo root.

```bash
# Activate venv
source .venv/bin/activate

# 1. Build graph (must run before training)
cd src && python build_graph.py

# 2. Train base TWIG model
cd src && python train_model.py

# 3. Evaluate retrieval
cd src && python evaluate_retrieval.py

# 4. Base TWIG edge ablation (A0–A63)
cd src && python run_edge_ablation.py --datasets feta --gpu 0 --ablation $(seq 0 63)

# 5. Query-Aware v2 pipeline (train + evaluate)
cd src/query_aware && python run_pipeline_v2.py

# 6. QA v2 edge ablation (find best edge config for QA)
cd src/query_aware && python run_qa_edge_ablation.py --datasets feta --gpu 0 --ablation $(seq 0 63)

# 7. Cross-dataset generalization test
cd src/query_aware && python run_cross_dataset_eval.py --gpu 0

# 8. Cross-validation
cd src && python cross_validate.py

# 9. Run tests (minimal stubs)
cd tests && python -m pytest test_training.py test_retrieval.py
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
abandon/                    # deprecated/abandoned scripts
src/utils/                  # data processing utilities, LaTeX table generators
```

Supported datasets: `feta`, `ottqa`, `e2ewtq`, `mimo_en`, `mimo_ch`, `mmqa`.

**Splits**: `train`, `dev`, `test`. The active DATASETS/SOURCES lists inside each script control which datasets are processed — comment/uncomment as needed.

## Architecture

### Graph Construction (`src/build_graph.py`)

Builds a `HeteroData` (PyG) with three node types:
- **table** — embedding of concatenated page title + sheet name + column names + first 5 rows
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

All heterogeneous edges also have `rev_*` reverse edges added automatically.

The graph is serialized with `torch.save(data, graph.pt)` including `data.metadata_maps` with `table_id_to_idx`, `table_meta`, `key_fields`.

### Table Key Fields

Key uniqueness depends on dataset type:
- `ottqa`, `feta`, `e2ewtq` → key = `sheet_name|file_name`
- `mimo_en`, `mimo_ch`, `mmqa` → key = `id`

### Model (`src/train_model.py` — class `DiffusionModel`)

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

Set `USE_HYPEROPT = True` to run Bayesian hyperparameter search via `hyperopt`.

Embedding model: `BAAI/bge-m3` (via `sentence-transformers`).

### Evaluation (`src/evaluate_retrieval.py`)

Loads a trained model + test graph, embeds queries with the same `bge-m3` encoder, runs GNN forward pass for table embeddings, computes cosine similarity, and reports: Recall@1/5/10, MRR, nDCG@5/10, Precision@5, Full Recall@5.

`evaluate_retrieval.py` imports `DiffusionModel` and `get_embedder` from `train_model.py` — run from `src/` so the relative import resolves.

### Query-Aware v2 (`src/query_aware/`)

Conditions the GNN on the query embedding via additional query→table/column/page edges. Uses a two-phase approach:
1. Train base TWIG model (or load pretrained checkpoint)
2. Fine-tune `QueryAwareModel` with query edges zero-initialized, differential learning rates

Key scripts:
- `train_query_aware_v2.py` — Training with subgraph-based QA fine-tuning
- `evaluate_query_aware_v2.py` — Evaluation with coarse→rerank pipeline
- `run_pipeline_v2.py` — End-to-end train + evaluate wrapper
- `run_qa_edge_ablation.py` — QA v2 edge ablation (A0–A63, all 64 configs)
- `run_cross_dataset_eval.py` — Cross-dataset generalization test (N×N matrix)

### Ablation & Analysis

- `src/run_edge_ablation.py` — Base TWIG edge ablation (A0–A63)
- `src/query_aware/run_qa_edge_ablation.py` — QA v2 edge ablation (A0–A63). Runs TWIG train → QA fine-tune → evaluate for each config.
- `src/query_aware/run_cross_dataset_eval.py` — Cross-dataset generalization. Tests each dataset's trained model on all other datasets.

Set `CUDA_VISIBLE_DEVICES` via `--gpu` before torch is imported (handled internally).

## Dependencies

```
torch, torch_geometric
sentence-transformers   # bge-m3 embedder
google-genai            # (optional, used in abandoned scripts)
pandas, tabulate, matplotlib
hyperopt                # Bayesian hyperparam search
```

Install: `pip install -r requirements.txt`
