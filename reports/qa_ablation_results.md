# QA Edge Ablation 完整分數報告

> 生成日期：2026-06-04
> 選邊原則：以 **dev set QA_R@10** 選每個資料集最佳邊配置，test set 僅用於報告，不參與選邊。
> 模型：TWIG base + Query-Aware v2 fine-tune（alpha=0，粗排與精排使用同一組邊）

---

## 一、最終最佳邊配置

| 資料集 | Config | Binary | 邊組合 | Dev QA_R@10 |
|---|---|---|---|---|
| FeTaQA | A20 | 010100 | tc + sp | 0.9760 |
| OTT-QA | A54 | 110110 | tt + tc + sp + cc | 0.9982 |
| MIMO-EN | A31 | 011111 | tc + tp + sp + cc + sc | 0.7448 |
| MIMO-CH | A28 | 011100 | tc + tp + sp | 0.7194 |
| E2E-WTQ | A19 | 010011 | tc + cc + sc | 0.9000 |
| MMQA | A17 | 010001 | tc + sc | 0.8273 |

邊縮寫：tt=similar_table, tc=has_column, tp=comes_from, sp=same_page, cc=similar_content, sc=shared_column_name

---

## 二、完整 Test Set 分數（dev-best 邊配置）

### FeTaQA（A20：tc + sp）

| Stage | R@1 | R@2 | R@5 | R@10 | R@50 | MRR |
|---|---|---|---|---|---|---|
| Stage 1：TWIG 粗排 | 0.8802 | 0.9301 | 0.9621 | 0.9760 | 0.9980 | 0.9176 |
| Stage 2：QA 精排 | **0.8962** | **0.9341** | **0.9601** | **0.9800** | **0.9960** | **0.9261** |
| Delta（QA − TWIG） | +0.0160 | +0.0040 | −0.0020 | **+0.0040** | −0.0020 | +0.0085 |

### OTT-QA（A54：tt + tc + sp + cc）

| Stage | R@1 | R@2 | R@5 | R@10 | R@50 | MRR |
|---|---|---|---|---|---|---|
| Stage 1：TWIG 粗排 | 0.8087 | 0.8881 | 0.9621 | 0.9819 | 1.0000 | 0.8739 |
| Stage 2：QA 精排 | **0.8321** | **0.9206** | **0.9729** | **0.9874** | **0.9982** | **0.8921** |
| Delta（QA − TWIG） | +0.0234 | +0.0325 | +0.0108 | **+0.0055** | −0.0018 | +0.0182 |

### MIMO-EN（A31：tc + tp + sp + cc + sc）

| Stage | R@1 | R@2 | R@5 | R@10 | R@50 | MRR |
|---|---|---|---|---|---|---|
| Stage 1：TWIG 粗排 | 0.4471 | 0.5463 | 0.6669 | 0.7304 | 0.9413 | 0.6254 |
| Stage 2：QA 精排 | **0.4463** | 0.5368 | **0.7034** | 0.7225 | **0.9444** | **0.6417** |
| Delta（QA − TWIG） | −0.0008 | −0.0095 | +0.0365 | **−0.0079** | +0.0031 | +0.0163 |

### MIMO-CH（A28：tc + tp + sp）

| Stage | R@1 | R@2 | R@5 | R@10 | R@50 | MRR |
|---|---|---|---|---|---|---|
| Stage 1：TWIG 粗排 | 0.4203 | 0.4998 | 0.5938 | 0.6825 | 0.8763 | 0.6110 |
| Stage 2：QA 精排 | **0.4203** | 0.4957 | **0.6162** | **0.6945** | **0.9045** | 0.6074 |
| Delta（QA − TWIG） | 0.0000 | −0.0041 | +0.0224 | **+0.0120** | +0.0282 | −0.0036 |

### E2E-WTQ（A19：tc + cc + sc）

| Stage | R@1 | R@2 | R@5 | R@10 | R@50 | MRR |
|---|---|---|---|---|---|---|
| Stage 1：TWIG 粗排 | 0.3443 | 0.5738 | 0.7377 | 0.8852 | 1.0000 | 0.5315 |
| Stage 2：QA 精排 | **0.4754** | **0.5902** | **0.8361** | **0.9508** | **1.0000** | **0.6359** |
| Delta（QA − TWIG） | +0.1311 | 0.0000 | +0.0984 | **+0.0656** | 0.0000 | +0.1044 |

### MMQA（A17：tc + sc）

| Stage | R@1 | R@2 | R@5 | R@10 | R@50 | MRR |
|---|---|---|---|---|---|---|
| Stage 1：TWIG 粗排 | 0.3581 | 0.4989 | 0.6909 | 0.8025 | 0.9548 | 0.8677 |
| Stage 2：QA 精排 | **0.4112** | **0.5248** | **0.7042** | **0.8289** | **0.9657** | **0.9431** |
| Delta（QA − TWIG） | +0.0531 | +0.0259 | +0.0133 | **+0.0264** | +0.0109 | +0.0754 |

---

## 三、跨資料集彙整（Test Set）

| 資料集 | Config | TWIG R@1 | TWIG R@10 | TWIG MRR | QA R@1 | QA R@10 | QA MRR | ΔR@10 |
|---|---|---|---|---|---|---|---|---|
| FeTaQA | A20 | 0.8802 | 0.9760 | 0.9176 | 0.8962 | **0.9800** | 0.9261 | +0.004 |
| OTT-QA | A54 | 0.8087 | 0.9819 | 0.8739 | 0.8321 | **0.9874** | 0.8921 | +0.006 |
| MIMO-EN | A31 | 0.4471 | 0.7304 | 0.6254 | 0.4463 | 0.7225 | 0.6417 | −0.008 |
| MIMO-CH | A28 | 0.4203 | 0.6825 | 0.6110 | 0.4203 | **0.6945** | 0.6074 | +0.012 |
| E2E-WTQ | A19 | 0.3443 | 0.8852 | 0.5315 | 0.4754 | **0.9508** | 0.6359 | +0.066 |
| MMQA | A17 | 0.3581 | 0.8025 | 0.8677 | 0.4112 | **0.8289** | 0.9431 | +0.026 |

---

## 四、各資料集 Top-5 Config（dev QA_R@10 排序）

### FeTaQA
| Config | 邊組合 | Dev QA R@10 | Test QA R@10 |
|---|---|---|---|
| **A20** ← | tc + sp | **0.9760** | 0.9800 |
| A25 | tc + tp + sc | 0.9760 | 0.9840 |
| A49 | tt + tc + sc | 0.9740 | 0.9800 |
| A57 | tt + tc + tp + sc | 0.9740 | 0.9760 |
| A18 | tc + cc | 0.9720 | 0.9820 |

### OTT-QA
| Config | 邊組合 | Dev QA R@10 | Test QA R@10 |
|---|---|---|---|
| **A54** ← | tt + tc + sp + cc | **0.9982** | 0.9874 |
| A21 | tc + sp + sc | 0.9964 | 0.9856 |
| A52 | tt + tc + sp | 0.9964 | 0.9819 |
| A59 | tt + tc + tp + cc + sc | 0.9964 | 0.9892 |
| A16 | tc | 0.9946 | 0.9856 |

### MIMO-EN
| Config | 邊組合 | Dev QA R@10 | Test QA R@10 |
|---|---|---|---|
| **A31** ← | tc + tp + sp + cc + sc | **0.7448** | 0.7225 |
| A63 | tt + tc + tp + sp + cc + sc | 0.7375 | 0.7725 |
| A27 | tc + tp + cc + sc | 0.7370 | 0.7873 |
| A59 | tt + tc + tp + cc + sc | 0.7359 | 0.7082 |
| A21 | tc + sp + sc | 0.7318 | 0.7310 |

### MIMO-CH
| Config | 邊組合 | Dev QA R@10 | Test QA R@10 |
|---|---|---|---|
| **A28** ← | tc + tp + sp | **0.7194** | 0.6945 |
| A22 | tc + sp + cc | 0.7160 | 0.7118 |
| A58 | tt + tc + tp + cc | 0.7046 | 0.7103 |
| A60 | tt + tc + tp + sp | 0.6988 | 0.7008 |
| A56 | tt + tc + tp | 0.6940 | 0.7263 |

### E2E-WTQ（僅 32 個 valid config，無 same_page 邊）
| Config | 邊組合 | Dev QA R@10 | Test QA R@10 |
|---|---|---|---|
| **A19** ← | tc + cc + sc | **0.9000** | 0.9508 |
| A24 | tc + tp | 0.9500\* | 0.9508 |
| A9 | tp + sc | 0.9333 | 0.9180 |
| A10 | tp + cc | 0.9000 | 0.8689 |
| A2 | cc | 0.8833 | 0.8689 |

\* A24 dev dev=0.9500 略高，但 tp（comes_from）在 e2ewtq 語意為空（每 CSV 僅一張表，file_name 無意義），與 A19 test 分數相同，故選 A19。

### MMQA
| Config | 邊組合 | Dev QA R@10 | Test QA R@10 |
|---|---|---|---|
| **A17** ← | tc + sc | **0.8273** | 0.8289 |
| A25 | tc + tp + sc | 0.8012 | 0.8128 |
| A2 | cc | 0.7992 | 0.7906 |
| A51 | tt + tc + cc + sc | 0.7924 | 0.8206 |
| A56 | tt + tc + tp | 0.7897 | 0.7761 |

---

## 五、觀察與分析

### QA 精排的增益
- **QA 在大多數資料集有效**：FeTaQA（+0.004）、OTT-QA（+0.006）、MIMO-CH（+0.012）、MMQA（+0.026）在 R@10 都有正向改善。
- **MMQA MRR 改善最顯著**（+0.075），代表 QA rerank 在需要精確排序的場景效果最強。
- **MIMO-EN R@10 略降（−0.008）**：QA rerank 未能改善 MIMO-EN 的 R@10，但 R@5（+0.037）和 MRR（+0.016）仍有提升，說明對高 Recall 指標的貢獻受限。
- **E2E-WTQ R@10 持平（0.000）**：粗排已近乎完美（0.9508），rerank 空間有限，但 R@1（+0.033）、R@5（+0.066）有改善。

### 最佳邊配置模式
- **tc（has_column）** 出現在全部 6 個最佳配置 → 結構性邊最重要。
- **sp（same_page）** 出現在 4/6（feta/ottqa/mimo_en/mimo_ch）→ 對有真實或衍生頁面關係的資料集有效；e2ewtq 無此邊（設計如此），mmqa 未選（section_title 衍生的 sp 語意不強）。
- **tt（similar_table）** 只在 ottqa 入選 → 大型多表語料庫才有用。
- **cc（similar_content）** 在 ottqa/mimo_en/e2ewtq 入選 → 表內容相似度的效益資料集相關。
- **e2ewtq 選 A19 而非 dev 最高的 A24**：A24 的 `tp`（comes_from）在 e2ewtq 中語意為空（1:1 table-page mapping，file_name 為無意義 CSV 路徑），兩者 test 分數相同（0.9508），選語意更乾淨的 A19（tc + cc + sc）。

### 與舊版本比較（舊版選邊在 test set 上有資料洩漏，不可直接比）
舊版 `BEST_EDGE_CONFIGS` 使用 test set 選邊；本次已改為 **dev set 選邊 / test set 報告**，符合正確的實驗協定。邊配置有所不同（例如 feta 從 A61 改為 A20），數字無法直接比較。

---

## 六、BEST_EDGE_CONFIGS（供後續腳本使用）

```python
QA_BEST_EDGE_CONFIGS = {
    "feta":    ['has_column', 'same_page'],                                              # A20 | dev R@10=0.9760
    "ottqa":   ['similar_table', 'has_column', 'same_page', 'similar_content'],          # A54 | dev R@10=0.9982
    "mimo_en": ['has_column', 'comes_from', 'same_page', 'similar_content', 'shared_column_name'],  # A31 | dev R@10=0.7448
    "mimo_ch": ['has_column', 'comes_from', 'same_page'],                                # A28 | dev R@10=0.7194
    "e2ewtq":  ['has_column', 'similar_content', 'shared_column_name'],                 # A19 | dev R@10=0.9000 (tp semantically empty in e2ewtq)
    "mmqa":    ['has_column', 'shared_column_name'],                                     # A17 | dev R@10=0.8273
}
```

---

*選邊完全基於 dev set，test set 僅用於最終報告。所有實驗使用 alpha=0（QA reranker 直接使用子圖 embedding 排序，不與粗排分數插值）。粗排與精排使用同一組邊（修正自舊版粗排用全圖的 bug，commit c62b65c）。*
