# 實驗紀錄

> 本文件整理本次對話中所有實驗修改、執行過程與跑出來的分數。

---

## 一、問題修正

### 1. 移除加權平均（alpha 插值）

**問題**：`evaluate_query_aware_v2.py` 與 `run_qa_edge_ablation.py` 的最終分數都是用
`final_scores = 0.3 × coarse + 0.7 × rerank`（alpha=0.3）計算，學姐要求改成整個流程的結果，不做插值。

**修改**：
- `evaluate_query_aware_v2.py`：`ALPHA = 0.0`，`final_scores = rerank_scores`
- `run_qa_edge_ablation.py`：`EVAL_ALPHA = 0.0`，`final_scores = rerank_scores`

---

### 2. BEST_EDGE_CONFIGS 更新（QA 最佳邊配置）

**問題**：`train_query_aware_v2.py` 裡的 `BEST_EDGE_CONFIGS` 是根據 TWIG ablation 結果設的，
不是 QA pipeline 的最佳邊。

**做法**：讀取已完成的 QA edge ablation 結果（`results/qa_edge_ablation/`），
依 `QA_R@10` 選出每個 dataset 的最佳 config。

**舊設定 → 新設定**：

| Dataset | 舊 (TWIG-based) | 新 (QA ablation best) | Config |
|---------|----------------|----------------------|--------|
| feta | has_column, same_page | similar_table, has_column, comes_from, same_page, shared_column_name | A61 |
| ottqa | has_column, similar_content | similar_table, has_column, comes_from, same_page, similar_content, shared_column_name | A63 |
| mimo_en | has_column, similar_content, shared_column_name | similar_table, comes_from, same_page, shared_column_name | A45 |
| mimo_ch | similar_content | has_column, comes_from, same_page, similar_content | A30 |
| e2ewtq | similar_content | has_column, shared_column_name | A17 |
| mmqa | similar_table, has_column, comes_from, same_page, similar_content | has_column, comes_from, similar_content, shared_column_name | A27 |

> 注意：QA ablation 結果原本用 alpha=0.3 選出，未來若要完整重跑須考慮此偏差。
> e2ewtq 原本只跑了 32/64 個 config，A17 是在前 32 個中的最佳。

---

### 3. MMQA checkpoint 維度不符

**問題**：`checkpoints/mmqa/model.pt` 是用 `hidden_channels=512` 訓的，
其餘 dataset 都是 768，導致 QA fine-tune 時 size mismatch，等於從頭訓練。

**修正**：關閉 `USE_HYPEROPT`，強制用 `BEST_PARAMS`（hidden_channels=768）重新訓練 mmqa 的 TWIG base model，
再接著跑 QA fine-tune。

---

### 4. TWIG 評估用錯 checkpoint

**問題**：`evaluate_retrieval.py` 使用 `model.pt`（原始全邊 TWIG），
應該使用 `model_best_edges.pt`（最佳邊配置重訓的版本）。

**修正**：
- `MODEL_PATH` 改為 `model_best_edges.pt`
- 新增 `filter_graph_edges()`：讀 checkpoint 裡的 `best_edges` 欄位，評估時對 test graph 做相同的邊過濾，確保模型架構與評估圖一致。

---

### 5. TWIG-QA 從錯誤 base 出發

**問題**：`train_query_aware_v2.py` 的 `PRETRAINED_PATH` 是 `model.pt`（較弱的全邊模型），
而非 `model_best_edges.pt`（最佳邊配置、分數更高的模型），導致 TWIG-QA 部分 dataset 分數低於 TWIG。

**修正**：
- `PRETRAINED_PATH` 改為 `model_best_edges.pt`
- `evaluate_query_aware_v2.py` 的粗排也改用 `model_best_edges.pt`，並讀取其 `best_edges` 做圖過濾
- 重新訓練所有 dataset 的 QA 模型（進行中）

---

## 二、新增指標

### R@2（MMQA 專用）
Reviewer 指出 Table 1 MMQA 使用 R@2 為起始指標（不是 R@1）。
在 `evaluate_query_aware_v2.py` 新增 `recall2` 累積與 `Recall@2` 輸出。

### R@50（全 dataset）
`evaluate_retrieval.py` 及 `evaluate_query_aware_v2.py` 均新增 `Recall@50`：
- `TOP_K` 從 10 改為 50
- 在指標 dict 與 print 加入 `Recall@50`

---

## 三、計算成本量測

### 方法
- **Online**：每個 method 在獨立 subprocess 跑（避免跨 method GPU cache 污染），各跑 3 次取 median。
- **Offline**：TWIG 實際執行 graph build + 計時一個 epoch 外推；QGPT 隨機抽樣 50 張表計時 LLM call，外推至全體表數。

### 硬體

| 項目 | 規格 |
|------|------|
| GPU | NVIDIA TITAN RTX (23.6 GB VRAM) |
| CPU | Intel Core i9-10900K @ 3.70 GHz |
| RAM | 125.7 GB |

### Online Cost（OTT-QA test，8,889 tables，41,468 queries）

| Method | Total | Per-query | vs bge-m3 |
|--------|-------|-----------|-----------|
| bge-m3 (baseline) | 1.70 min | 2.46 ms | — |
| TWIG | 1.72 min | 2.49 ms | +1.2% |
| TWIG-QA | 24.18 min | 34.99 ms | +14.2× |

> TWIG 的 GNN forward 整個 corpus 只做一次，攤分到每個 query 幾乎可忽略。
> TWIG-QA 每個 query 需建 subgraph + rerank，per-query latency 約 33 ms。

---

### Offline Cost（FeTaQA，7,326 張表，TITAN RTX）

與 QGPT（arXiv 2508.06168）比較。QGPT 使用 LLaMA-3.1-8B-Instruct 對每張表生成 pseudo-query，
本實驗透過 Ollama 呼叫相同模型（`llama3.1:8b-instruct-fp16`），抽 50 張表計時外推。

完整流程：TWIG 直接呼叫 `build_graph.main()` + 完整訓練迴圈（含 hard-neg mining + validation per epoch）；QGPT 抽 50 張表計時外推。

| 步驟 | TWIG | QGPT |
|------|------|------|
| Graph build（bge-m3 embed 全表 + 建邊） | **3.2 min** | — |
| Query embedding（訓練用） | 12.2 s | — |
| 模型訓練（30 epochs × 6.8s/epoch） | **3.4 min** | — |
| LLM pseudo-query 生成 | — | **12.5 h**（avg 6.2s/table × 7,326） |
| Embedding（table snippet + questions） | — | 10.6 min |
| **Total** | **≈ 6.8 min** | **≈ 12.7 h** |

> TWIG offline 約為 QGPT 的 **1/112**。QGPT 瓶頸為逐表 LLM inference（median 5.2s/table）。
> 舊版量測（14.3 min）有誤：build_graph 跑了其他 dataset（feta 被 comment 掉），訓練迴圈也不完整，已修正。

## 四、實驗結果

### TWIG（model_best_edges.pt，邊過濾匹配）

| Dataset | R@1 | R@5 | R@10 | R@50 | MRR |
|---------|-----|-----|------|------|-----|
| FeTaQA | 0.8822 | 0.9621 | 0.9820 | 0.9980 | 0.9176 |
| OTT-QA | 0.8303 | 0.9675 | 0.9838 | 0.9982 | 0.8913 |
| MIMO-EN | 0.3286 | 0.6220 | 0.6749 | 0.9114 | 0.5005 |
| MIMO-CH | 0.3578 | 0.6135 | 0.6792 | 0.8377 | 0.5560 |
| E2EWTQ | 0.4590 | 0.8197 | 0.8852 | 1.0000 | 0.6046 |
| MMQA | 0.3890 | 0.6545 | 0.7583 | 0.9206 | 0.9153 |

> MMQA 使用重訓的 hidden_channels=768 模型。

### TWIG-QA v1（從 model.pt fine-tune，QA 最佳邊配置，alpha=0）

| Dataset | R@1 | R@5 | R@10 | R@50 | MRR |
|---------|-----|-----|------|------|-----|
| FeTaQA | 0.8743 | 0.9321 | 0.9461 | 0.9561 | 0.8993 |
| OTT-QA | 0.8755 | 0.9711 | 0.9765 | 0.9783 | 0.9172 |
| MIMO-EN | 0.3696 | 0.6201 | 0.7598 | 0.9471 | 0.5843 |
| MIMO-CH | 0.3728 | 0.6345 | 0.7348 | 0.8380 | 0.5883 |
| E2EWTQ | 0.4754 | 0.7377 | 0.8525 | 1.0000 | 0.5932 |
| MMQA | 0.4018 | 0.7015 | 0.7936 | 0.9438 | 0.9264 |

> MMQA R@2 = 0.5224（依 reviewer 要求，MMQA 從 R@2 開始報告）。

### TWIG-QA v2（從 model_best_edges.pt fine-tune，alpha=0）✅

| Dataset | R@1 | R@5 | R@10 | R@50 | MRR |
|---------|-----|-----|------|------|-----|
| FeTaQA | 0.8942 | 0.9601 | 0.9760 | 0.9980 | 0.9252 |
| OTT-QA | 0.8736 | 0.9765 | 0.9856 | 0.9964 | 0.9193 |
| MIMO-EN | 0.4085 | 0.5907 | 0.7201 | 0.9365 | 0.5968 |
| MIMO-CH | 0.3673 | 0.5858 | 0.6925 | 0.8663 | 0.5716 |
| E2EWTQ | 0.4590 | 0.7869 | 0.8689 | 1.0000 | 0.5981 |
| MMQA | 0.3971 | 0.6864 | 0.7819 | 0.9258 | 0.9268 |

> MMQA R@2 = 0.5419（依 reviewer 要求，MMQA 從 R@2 開始報告）。

---

## 五、待辦 / 已知問題

- [x] TWIG-QA v2（從 best_edges 出發）已完成
- [x] Offline cost 量測（TWIG vs QGPT）已完成
- [ ] QA edge ablation 原本用 alpha=0.3 選 best config，未來可考慮重跑 alpha=0 版本（但成本極高）
- [ ] e2ewtq QA ablation 只跑了 A0–A31，A17 是在前 32 個中選出的
- [ ] TWIG-QA v1 vs v2 差異分析（v2 在 MIMO 上較差，原因待查）

---

## 六、腳本路徑對照

| 腳本 | 說明 |
|------|------|
| `reports/src/evaluate_retrieval.py` | TWIG 評估，現用 model_best_edges.pt + 邊過濾 |
| `reports/src/query_aware/evaluate_query_aware_v2.py` | TWIG-QA 評估，alpha=0，粗排用 best_edges |
| `reports/src/query_aware/train_query_aware_v2.py` | QA fine-tune，現從 model_best_edges.pt 出發 |
| `reports/src/query_aware/run_pipeline_v2.py` | 一鍵 train+eval pipeline |
| `reports/src/measure_cost.py` | 計算成本量測（online），每 method 獨立 subprocess |
| `results/qa_edge_ablation/` | QA edge ablation 結果（64 configs × 6 datasets） |
| `results/query_aware_v2/` | TWIG-QA 評估結果 JSON |
| `results/evaluate/` | TWIG 評估結果 JSON |
| `results/cost_measurement/` | 計算成本量測結果 JSON |
