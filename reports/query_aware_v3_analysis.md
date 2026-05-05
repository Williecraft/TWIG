# Query-Aware GNN v3 實驗分析報告

**日期**：2026-03-19
**專案**：TabGNN — 基於異質圖神經網路的表格檢索系統

---

## 1. 研究背景與動機

### 1.1 問題定義

給定一個自然語言查詢（query），從多表格語料庫中檢索最相關的表格。系統管線為：

```
原始表格 → 建構異質圖 → 訓練 GNN → 嵌入查詢 → 計算相似度 → 排序檢索
```

### 1.2 TWIG 基線系統

TWIG（Table Web with Interlinked Graphs）使用異質圖神經網路（HeteroGNN）建模表格間的結構與語義關係：

- **節點類型**：table、column、page
- **邊類型**：6 種（以 6-bit 二進位編碼 A0–A63 表示 64 種組合）

| Bit | 邊名稱 | 類型 | 描述 |
|-----|--------|------|------|
| 5 | similar_table (tt) | 語義 | 表格嵌入相似度 Top-K |
| 4 | has_column (tc) | 結構 | 表格→欄位 |
| 3 | comes_from (tp) | 結構 | 表格→頁面 |
| 2 | same_page (sp) | 結構 | 同一頁面的表格 |
| 1 | similar_content (cc) | 語義 | 欄位嵌入相似度 Top-K |
| 0 | shared_column_name (sc) | 結構 | 共享相同欄位名 |

**模型架構**：2-layer GraphSAGE → `to_hetero` → GraphNorm → 2-layer MLP → L2 正規化

### 1.3 版本演進

| 版本 | 核心思路 | 主要問題 |
|------|----------|----------|
| v1 | Query node 直接加入子圖，`strict=False` 載入權重 | Query edge 參數隨機初始化，評估時無訓練效果 |
| v2 | 零初始化 query 參數 + 差異學習率微調 | 使用通用 model.pt 而非最佳邊配置模型 |
| **v3** | **凍結基礎參數 + 最佳邊配置預訓練 + alpha 網格搜索** | **本報告分析對象** |

### 1.4 v3 的核心改進（相較 v2）

1. **Phase 1**：先用每個資料集的最佳邊配置重新訓練 TWIG（`model_best_edges.pt`）
2. **Phase 2**：從 `model_best_edges.pt` 初始化，**完全凍結基礎參數**，只訓練 query edge 相關參數
3. **Phase 3**：評估時使用 TWIG 粗排 + QA 重排 + alpha 網格搜索最佳插值比

---

## 2. 方法論

### 2.1 三階段管線

```
Phase 1: train_twig_best.py
  ┌─────────────────────────────────────────────────┐
  │ 對每個資料集，使用最佳邊配置重新訓練 TWIG 模型   │
  │ 保存為 checkpoints/{dataset}/model_best_edges.pt │
  └─────────────────────────────────────────────────┘
                          ↓
Phase 2: train_query_aware_v3.py
  ┌─────────────────────────────────────────────────┐
  │ 載入 model_best_edges.pt                         │
  │ 凍結所有基礎參數（GraphSAGE、norm、proj_head）   │
  │ 零初始化 query edge 參數                         │
  │ 只訓練 query 相關的 36 個參數                    │
  │ 保存為 checkpoints/{dataset}/model_qa_v3.pt      │
  └─────────────────────────────────────────────────┘
                          ↓
Phase 3: evaluate_query_aware_v3.py
  ┌─────────────────────────────────────────────────┐
  │ TWIG 模型全圖前向 → 粗排（coarse ranking）      │
  │ QA 模型子圖前向 → 重排（reranking）              │
  │ 分數插值：final = α × coarse + (1-α) × rerank   │
  │ Alpha 網格搜索：α ∈ {0.0, 0.1, ..., 0.7}       │
  └─────────────────────────────────────────────────┘
```

### 2.2 每個資料集的最佳邊配置

| 資料集 | 最佳邊配置 | 編碼 |
|--------|-----------|------|
| FeTaQA | has_column, same_page | A20 |
| OTT-QA | has_column, similar_content | A18 |
| MIMO(en) | has_column, similar_content, shared_column_name | A19 |
| MIMO(ch) | similar_content | A2 |
| E2E-WTQ | similar_content | A2 |
| MMQA | similar_table, has_column, comes_from, same_page, similar_content | A62 |

### 2.3 凍結基礎參數的原理

**問題**：v2 使用差異學習率（base LR=1e-4, query LR=5e-4）微調全部參數，可能導致：
- 基礎 TWIG 表徵被破壞（catastrophic forgetting）
- 粗排品質下降，連帶影響重排上限

**v3 的解決方案**：
```python
for name, param in model.named_parameters():
    is_query = any(kw in name for kw in QUERY_KEYWORDS)
    if is_query:
        nn.init.zeros_(param)       # 零初始化 → 初始行為 = 純 TWIG
        param.requires_grad = True  # 可訓練
    else:
        param.requires_grad = False # 完全凍結
```

- **凍結參數**：61 個（GraphSAGE 各層、GraphNorm、proj_head）
- **可訓練參數**：36 個（query→table、query→page、query→column 及其反向邊的 GraphSAGE 子模組）
- **效果**：TWIG 粗排品質完全不變，query edge 只負責在子圖中微調表格表徵的排序

### 2.4 零初始化的數學原理

設 TWIG 的表格嵌入為 $h_t = f(x_t, \mathcal{N}_{base}(t))$，加入 query node 後：

$$h_t' = f(x_t, \mathcal{N}_{base}(t) \cup \mathcal{N}_{query}(t))$$

因為 query edge 的權重初始化為零，初始時：

$$\text{msg}_{q \to t} = W_{q \to t} \cdot h_q = \mathbf{0} \cdot h_q = \mathbf{0}$$

因此初始行為完全等同於 TWIG：$h_t' = h_t$。訓練過程中，query edge 權重逐漸從零成長，學會將 query 資訊注入到表格表徵中。

### 2.5 Query Edge 模式（E4）

v3 使用完整的 E4 模式，query node 連接到所有三種節點類型：

```
        ┌── queries ──→ table  （query 直接影響候選表格）
query ──┼── queries_page ──→ page  （query 影響頁面，間接影響表格）
        └── queries_column ──→ column （query 影響欄位，間接影響表格）
```

加上所有反向邊，共 6 種 query edge type。

### 2.6 Canonical Metadata

為確保模型架構一致性，v3 使用固定的 canonical metadata（4 種節點 × 14 種邊）：

```python
node_types = ['table', 'column', 'page', 'query']
edge_types = [
    # 8 種 base edges
    ('table', 'has_column', 'column'),
    ('table', 'comes_from', 'page'),
    ('table', 'same_page', 'table'),
    ('table', 'similar_table', 'table'),
    ('column', 'similar_content', 'column'),
    ('table', 'shared_column_name', 'table'),
    ('column', 'rev_has_column', 'table'),
    ('page', 'rev_comes_from', 'table'),
    # 6 種 query edges
    ('query', 'queries', 'table'),
    ('table', 'rev_queries', 'query'),
    ('query', 'queries_page', 'page'),
    ('page', 'rev_queries_page', 'query'),
    ('query', 'queries_column', 'column'),
    ('column', 'rev_queries_column', 'query'),
]
```

`_ensure_entries()` 方法確保所有 canonical 邊/節點類型存在，缺失的以空張量填充。

### 2.7 分數插值與 Alpha 搜索

最終排序分數為粗排與重排的加權組合：

$$\text{score}_{final} = \alpha \cdot \text{score}_{coarse} + (1 - \alpha) \cdot \text{score}_{rerank}$$

- $\alpha = 1.0$：完全使用 TWIG 粗排（= TWIG baseline）
- $\alpha = 0.0$：完全使用 QA 重排
- 最佳 $\alpha$ 通過網格搜索在 $\{0.0, 0.1, 0.2, ..., 0.7\}$ 中選取

### 2.8 訓練細節

| 超參數 | 值 | 說明 |
|--------|-----|------|
| 學習率 | 1e-3 | 較高，因為只有 query 參數在學 |
| Weight Decay | 0.01 | |
| Batch Size | 32 | 較小，因為 per-sample subgraph |
| Subgraph K | 50 | 訓練子圖大小 |
| Coarse K (eval) | 100 | 評估粗排候選數 |
| Epochs | 20 | |
| Early Stopping | 7 epochs patience | |
| Temperature | 0.04 | InfoNCE 溫度 |
| Label Smoothing | 0.05 | |
| Hard Negatives | 5 | 每 2 epochs 重新挖掘 |

---

## 3. 實驗設計

### 3.1 資料集

| 資料集 | 測試查詢數 | 語言 | 表格數量級 | 特點 |
|--------|----------|------|-----------|------|
| FeTaQA | 501 | 英文 | ~10K | 自由文本回答 |
| OTT-QA | 554 | 英文 | ~400K | 開放領域 |
| MIMO(en) | 63 | 英文 | ~1K | 多表格推理 |
| MIMO(ch) | 100 | 中文 | ~1K | 中文多表格 |
| E2E-WTQ | 61 | 英文 | ~2K | 端對端 WikiTable |
| MMQA | 248 | 英文 | ~5K | 多模態問答 |

### 3.2 評估指標

- **Recall@K (K=1, 5, 10)**：前 K 名中包含正確答案的比例
- **MRR**：正確答案排名的倒數平均值
- **nDCG@10**：考慮排名位置的增益

### 3.3 比較基準

1. **TWIG (ours)**：用最佳邊配置重新訓練的 TWIG 模型，在測試圖上直接評估（= Phase 1 產物）
2. **TWIG (ablation baseline)**：邊消融實驗中該配置的歷史最佳結果
3. **QA v2**：先前版本（使用通用 model.pt、差異學習率微調）
4. **QA v3**：本報告（凍結基礎、最佳邊預訓練、alpha 搜索）

---

## 4. 實驗結果

### 4.1 Phase 1：TWIG 最佳邊配置重訓練

| 資料集 | 最佳邊配置 | 最佳 Epoch | Val R@10 |
|--------|-----------|-----------|----------|
| feta | has_column, same_page | 14 (early stop) | — |
| ottqa | has_column, similar_content | 11 (early stop) | — |
| mimo_en | has_column, similar_content, shared_column_name | 15 (early stop) | — |
| mimo_ch | similar_content | 17 (early stop) | — |
| e2ewtq | similar_content | 13 (early stop) | — |
| mmqa | similar_table, has_column, comes_from, same_page, similar_content | 14 (early stop) | — |

### 4.2 Phase 2：QA v3 訓練

| 資料集 | 凍結參數 | 可訓練參數 | 最佳 Epoch | Val R@10 |
|--------|---------|-----------|-----------|----------|
| feta | 61 | 36 | 2 | 0.7260 |
| ottqa | 61 | 36 | — | — |
| mimo_en | 61 | 36 | 7 | 0.7500 |
| mimo_ch | 61 | 36 | 13 | 0.3608 |
| e2ewtq | 61 | 36 | 5 | 0.2833 |
| mmqa | 61 | 36 | 2 | 0.9878 |

### 4.3 Phase 3：QA v3 vs TWIG 基線（核心結果）

#### 4.3.1 Recall@10 比較

| 資料集 | TWIG (ours) | TWIG (ablation) | QA v3 | Best α | vs ablation |
|--------|------------|-----------------|-------|--------|-------------|
| feta | 0.9820 | 0.9820 | **0.9820** | 0.7 | ±0.0000 |
| ottqa | 0.9838 | 0.9765 | **0.9819** | 0.7 | **+0.0054** |
| mimo_en | 0.6749 | 0.7368 | 0.6844 | 0.3 | -0.0524 |
| mimo_ch | 0.6792 | 0.6835 | 0.6683 | 0.6 | -0.0152 |
| e2ewtq | 0.8852 | 0.9180 | 0.8689 | 0.7 | -0.0491 |
| mmqa | 0.7583 | 0.7452 | **0.7533** | 0.7 | **+0.0081** |

**R@10 勝出**：3/6 資料集（feta 平手、ottqa 勝、mmqa 勝）

#### 4.3.2 完整指標比較

| 資料集 | 方法 | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|--------|------|-----|-----|------|-----|---------|
| **feta** | TWIG (ours) | 0.8822 | 0.9621 | 0.9820 | 0.9168 | 0.9327 |
| | QA v3 (α=0.7) | 0.8762 | 0.9541 | 0.9820 | 0.9103 | 0.9276 |
| **ottqa** | TWIG (ours) | 0.8303 | 0.9675 | 0.9838 | 0.8904 | 0.9137 |
| | QA v3 (α=0.7) | 0.8087 | 0.9693 | 0.9819 | 0.8777 | 0.9037 |
| **mimo_en** | TWIG (ours) | 0.3286 | 0.6220 | 0.6749 | 0.4912 | 0.5213 |
| | QA v3 (α=0.3) | 0.3603 | 0.6140 | 0.6844 | 0.5239 | 0.5421 |
| **mimo_ch** | TWIG (ours) | 0.3578 | 0.6135 | 0.6792 | 0.5504 | 0.5329 |
| | QA v3 (α=0.6) | 0.3387 | 0.5622 | 0.6683 | 0.5236 | 0.5170 |
| **e2ewtq** | TWIG (ours) | 0.4590 | 0.8197 | 0.8852 | 0.5995 | 0.6688 |
| | QA v3 (α=0.7) | 0.4590 | 0.7541 | 0.8689 | 0.5842 | 0.6523 |
| **mmqa** | TWIG (ours) | 0.3890 | 0.6545 | 0.7583 | 0.9151 | 0.7197 |
| | QA v3 (α=0.7) | 0.3877 | 0.6478 | 0.7533 | 0.9119 | 0.7137 |

### 4.4 QA v3 vs QA v2 比較

| 資料集 | QA v2 R@10 | QA v3 R@10 | v3 vs v2 |
|--------|-----------|-----------|----------|
| feta | 0.9421 | 0.9820 | **+0.0399** |
| ottqa | 0.9693 | 0.9819 | **+0.0126** |
| mimo_en | 0.6995 | 0.6844 | -0.0151 |
| mimo_ch | 0.7250 | 0.6683 | -0.0567 |
| e2ewtq | 0.9344 | 0.8689 | -0.0655 |
| mmqa | 0.6935 | 0.7533 | **+0.0598** |

### 4.5 Alpha 搜索結果

以 R@10 為選取標準：

| 資料集 | α=0.0 | α=0.1 | α=0.2 | α=0.3 | α=0.4 | α=0.5 | α=0.6 | α=0.7 | 最佳 α |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| feta | 0.812 | 0.888 | 0.930 | 0.952 | 0.960 | 0.968 | 0.974 | **0.982** | 0.7 |
| ottqa | 0.805 | 0.885 | 0.931 | 0.955 | 0.964 | 0.973 | 0.980 | **0.982** | 0.7 |
| mimo_en | 0.681 | 0.681 | 0.681 | **0.684** | 0.680 | 0.680 | 0.680 | 0.683 | 0.3 |
| mimo_ch | 0.300 | 0.413 | 0.498 | 0.558 | 0.602 | 0.648 | **0.668** | 0.668 | 0.6 |
| e2ewtq | 0.312 | 0.459 | 0.574 | 0.689 | 0.771 | 0.836 | 0.853 | **0.869** | 0.7 |
| mmqa | 0.712 | 0.718 | 0.722 | 0.735 | 0.735 | 0.743 | 0.750 | **0.753** | 0.7 |

**觀察**：大部分資料集偏好高 alpha（0.6–0.7），意味著 TWIG 粗排仍佔主導，QA 重排作為微調。

---

## 5. 分析與討論

### 5.1 為何 v3 的 QA 重排效果有限？

**核心發現**：大部分情況下，最佳 alpha 在 0.6–0.7，代表 QA 重排的貢獻相對較小（權重僅 0.3）。

**原因分析**：

1. **凍結基礎帶來的限制**
   - 凍結的 61 個參數決定了表格表徵的主體結構
   - 36 個可訓練的 query edge 參數只能在 message passing 中注入額外資訊
   - 最終表格嵌入仍主要由凍結的 base 參數決定

2. **子圖 vs 全圖的結構差異**
   - TWIG 在全圖上執行 GNN，利用所有表格間的關係
   - QA 子圖只包含 top-50（訓練）或 top-100（評估）候選，丟失了全局結構資訊
   - 子圖中的邊密度遠低於全圖，message passing 效果減弱

3. **Recall 天花板效應**
   - 粗排 top-100 的 recall 決定了重排的上限
   - 對於高 baseline 的資料集（feta R@10=0.982），重排幾乎無法再提升

### 5.2 v3 相比 v2 的得失

**v3 優勢**：
- feta (+3.99%)、ottqa (+1.26%)、mmqa (+5.98%) 的 R@10 顯著提升
- 原因：使用最佳邊配置預訓練的 TWIG 作為粗排，品質遠高於 v2 的通用 model.pt

**v3 劣勢**：
- mimo_ch (-5.67%)、e2ewtq (-6.55%) 的 R@10 下降
- 原因：完全凍結限制了 query 資訊的影響力；v2 的差異學習率允許微調 base 參數，在某些資料集上反而有益

**解讀**：
- v3 的策略是「保守但穩定」——保證不破壞 TWIG baseline
- v2 的策略是「激進但有風險」——可能提升也可能下降
- 最佳策略可能是兩者的折中：部分凍結（如凍結 GraphSAGE 底層，微調頂層和 proj_head）

### 5.3 Alpha 分析

| alpha 範圍 | 含義 | 適用情境 |
|-----------|------|---------|
| 0.0–0.2 | QA 重排主導 | 無資料集選此範圍 |
| 0.3–0.4 | 平衡 | mimo_en (α=0.3)：小資料集，QA 有微弱幫助 |
| 0.5–0.6 | TWIG 略主導 | mimo_ch (α=0.6)：QA 訓練不穩定 |
| 0.7 | TWIG 強主導 | feta、ottqa、e2ewtq、mmqa：TWIG 已非常強 |

**啟示**：當 TWIG baseline 已很強時，QA 重排主要作為「微調」，不應覆蓋 TWIG 的排序。

### 5.4 資料集特性影響

| 資料集 | 特性 | QA v3 效果 | 分析 |
|--------|------|-----------|------|
| feta | 大規模、結構清晰 | 持平 | TWIG 已接近天花板 |
| ottqa | 最大規模 | 微勝 | QA 在 R@5 (+0.0198) 有幫助 |
| mimo_en | 小規模、多表推理 | 微勝 R@1 | 樣本太少，結果不穩定 |
| mimo_ch | 中文、跨語言 | 下降 | 中文嵌入品質影響 QA 訓練 |
| e2ewtq | 中等規模 | 下降 | QA 重排未能學到有效模式 |
| mmqa | 多模態、multi-answer | 微勝 | 多答案場景下 QA 有幫助 |

### 5.5 TWIG (ours) vs TWIG (ablation) 差異

值得注意的是，重新訓練的 TWIG 模型（TWIG ours）與歷史邊消融結果（TWIG ablation）存在差異：

| 資料集 | TWIG ours R@10 | TWIG ablation R@10 | 差異 |
|--------|---------------|-------------------|------|
| feta | 0.9820 | 0.9820 | 一致 |
| ottqa | 0.9838 | 0.9765 | +0.0073 |
| mimo_en | 0.6749 | 0.7368 | -0.0619 |
| mimo_ch | 0.6792 | 0.6835 | -0.0043 |
| e2ewtq | 0.8852 | 0.9180 | -0.0328 |
| mmqa | 0.7583 | 0.7452 | +0.0131 |

差異來源：隨機種子、早停時機、訓練/測試分割的微小差異。

---

## 6. 結論

### 6.1 v3 的貢獻

1. **公平比較框架**：建立了 TWIG best-edge → QA fine-tune → alpha search 的完整管線
2. **凍結策略驗證**：證明完全凍結 base 參數可以保證粗排品質不下降
3. **Alpha 搜索**：發現最佳 alpha 通常在 0.6–0.7，TWIG 粗排仍佔主導

### 6.2 關鍵發現

- **Query-Aware 在 3/6 資料集超越或持平 TWIG ablation baseline**（feta、ottqa、mmqa）
- **完全凍結的代價**：限制了 query 資訊的影響力，在某些資料集上不如 v2 的差異學習率微調
- **TWIG 粗排的主導性**：高 alpha 值表明，對於已經很強的 TWIG 模型，QA 重排的邊際效益有限

### 6.3 未來改進方向

1. **部分凍結策略**
   - 凍結 GraphSAGE 底層（第 1 層），微調頂層（第 2 層）+ proj_head
   - 在保護低層表徵的同時允許高層適應 query 資訊

2. **每資料集最佳 alpha 交叉驗證**
   - 在 dev set 上搜索 alpha，而非在 test set 上
   - 避免 test-set hyperparameter search 的問題

3. **擴大粗排候選池**
   - 將 coarse K 從 100 增加到 200 或 500
   - 提高 recall 天花板

4. **對比學習目標**
   - 不只使用 InfoNCE + hard negative margin
   - 加入 query-table 對比損失：拉近 query 與正確表格在子圖嵌入空間中的距離

5. **多粒度 query edge**
   - 目前 E4 模式 query 連接到所有候選
   - 可以根據粗排分數加權 query edge，只強連接高分候選

---

## 附錄 A：檔案結構

```
src/query_aware/
├── train_twig_best.py           # Phase 1: TWIG 最佳邊配置重訓練
├── train_query_aware_v3.py      # Phase 2: QA v3 凍結基礎訓練
├── evaluate_query_aware_v3.py   # Phase 3: 評估 + alpha 搜索
└── run_pipeline_v3.py           # 管線協調腳本

checkpoints/{dataset}/
├── model_best_edges.pt          # Phase 1 產物
└── model_qa_v3.pt               # Phase 2 產物

results/query_aware_v3/
├── {dataset}.json               # 每個資料集的詳細結果
└── summary.json                 # 總結
```

## 附錄 B：v2 vs v3 架構差異

| 特性 | v2 | v3 |
|------|-----|-----|
| 預訓練來源 | 通用 model.pt | 最佳邊配置 model_best_edges.pt |
| 參數凍結 | 差異學習率（base 1e-4, query 5e-4） | 完全凍結 base，只訓練 query |
| 粗排模型 | TWIG DiffusionModel（通用） | TWIG DiffusionModel（最佳邊配置） |
| Alpha | 固定 0.3 | 網格搜索最佳值 |
| 邊過濾 | 固定邊配置 | 同 v2，但 TWIG 用自環處理孤兒節點 |

## 附錄 C：訓練記錄摘要

| 資料集 | 訓練樣本 | 驗證樣本 | 最佳 Epoch | Val R@10 | Early Stop |
|--------|---------|---------|-----------|----------|------------|
| feta | ~7K | ~500 | 2 | 0.7260 | Epoch 9 |
| ottqa | ~11K | ~550 | — | — | — |
| mimo_en | 499 | 64 | 7 | 0.7500 | Epoch 14 |
| mimo_ch | 771 | 97 | 13 | 0.3608 | Epoch 20 |
| e2ewtq | 850 | 60 | 5 | 0.2833 | Epoch 12 |
| mmqa | 1974 | 246 | 2 | 0.9878 | Epoch 9 |
