# BGE-M3 Dense Retrieval Baseline（無圖 / 無 GNN）

> 產生時間：2026-06-04 16:34:03

## 設定

- Embedding model：`BAAI/bge-m3`
- 方法：純 dense retrieval（cosine top-K），**不建圖、不跑 GNN**
- table 文字組成：與 `build_graph.py` table node 完全一致（Page / Sheet / Section / Columns / 前 5 列）
- 指標：Recall@k = 找回的 gold 表格比例（同 `evaluate_retrieval.py`），在有 gold 的 query 上平均
- 硬體：NVIDIA TITAN RTX（device=cuda）
- batch_size（線下編碼）：64；線上延遲量測：batch=1
- split：test

## Table 3 用：檢索品質（BGE-M3 no-graph 列）

| 資料集 | #tables | #queries(eval) | R@1 | R@5 | R@10 | R@50 | MRR |
|---|--:|--:|--:|--:|--:|--:|--:|
| feta | 501 | 501(501) | 0.9341 | 0.9741 | 0.9800 | 0.9940 | 0.9514 |
| ottqa | 405 | 554(554) | 0.9097 | 0.9819 | 0.9874 | 1.0000 | 0.9437 |
| mimo_en | 81 | 65(63) | 0.2114 | 0.3746 | 0.4228 | 0.8889 | 0.3594 |
| mimo_ch | 150 | 100(100) | 0.1968 | 0.3366 | 0.4230 | 0.6971 | 0.3580 |
| e2ewtq | 61 | 61(61) | 0.8197 | 0.9672 | 0.9836 | 1.0000 | 0.8771 |
| mmqa | 323 | 248(248) | 0.3765 | 0.7825 | 0.8733 | 0.9730 | 0.8939 |

## Table 4 用：效率（BGE-M3 列）

| 資料集 | #tables | 線下編碼語料庫(s) | 編碼吞吐(tables/s) | 線上 ms/query(mean) | 線上 ms/query(median) |
|---|--:|--:|--:|--:|--:|
| feta | 501 | 6.4 | 78.1 | 14.2 | 14.2 |
| ottqa | 405 | 11.8 | 34.4 | 13.9 | 13.9 |
| mimo_en | 81 | 5.0 | 16.3 | 13.9 | 13.9 |
| mimo_ch | 150 | 13.7 | 11.0 | 14.2 | 14.2 |
| e2ewtq | 61 | 3.7 | 16.6 | 14.0 | 14.0 |
| mmqa | 323 | 3.9 | 83.4 | 13.8 | 13.7 |

> 線上延遲 = 單一 query 的「bge-m3 編碼 + 與全語料庫 cosine + top-50」端到端時間（batch=1，含 GPU 同步）。線下 = 一次編碼整個 table 語料庫的時間。
