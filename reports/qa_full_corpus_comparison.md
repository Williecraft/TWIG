# Per-Split vs Full Corpus 評估對比報告

> 生成日期：2026-06-05
> 選邊：以 dev QA_R@10 選出（不用 test set）
> 兩種評估設定的差異見下方說明

---

## 兩種評估設定說明

| 設定 | 候選池 | 說明 |
|---|---|---|
| **Per-split** | 各 split 自己的表（test split 只含 61–501 張）| 封閉小語料庫；候選少，數字偏高 |
| **Full corpus** | train + dev + test 全部合併（291–8889 張）| 更接近真實檢索場景；候選多，更難 |

**Full corpus 候選池大小（vs per-split）**：

| 資料集 | per-split | full corpus | 倍數 |
|---|---|---|---|
| FeTaQA | 501 | 8,328 | 16.6× |
| OTT-QA | 405 | 8,889 | 22× |
| MIMO-EN | 81 | 291 | 3.6× |
| MIMO-CH | 150 | 458 | 3× |
| E2E-WTQ | 61 | 742 | 12× |
| MMQA | 323 | 646 | 2× |

---

## TWIG（粗排）R@10 對比

| 資料集 | Per-split R@10 | Full corpus R@10 | 差值 |
|---|---|---|---|
| FeTaQA | 0.9760 | 0.3234 | −0.653 |
| OTT-QA | 0.9819 | 0.7238 | −0.258 |
| MIMO-EN | 0.7304 | 0.5159 | −0.215 |
| MIMO-CH | 0.6825 | 0.5708 | −0.111 |
| E2E-WTQ | 0.9508 | 0.0164 | −0.934 |
| MMQA | 0.8025 | 0.6713 | −0.131 |

## QA（精排）R@10 對比

| 資料集 | Per-split R@10 | Full corpus R@10 | 差值 |
|---|---|---|---|
| FeTaQA | 0.9800 | 0.0479 | −0.932 |
| OTT-QA | 0.9874 | 0.7744 | −0.213 |
| MIMO-EN | 0.7225 | 0.5971 | −0.125 |
| MIMO-CH | 0.6945 | 0.5752 | −0.119 |
| E2E-WTQ | 0.9508 | 0.0820 | −0.869 |
| MMQA | 0.8289 | 0.6458 | −0.183 |

---

## Full Corpus 完整指標（Test Set，dev-best 邊）

### TWIG 粗排

| 資料集 | Config | R@1 | R@5 | R@10 | R@50 | MRR |
|---|---|---|---|---|---|---|
| FeTaQA | A20 (tc+sp) | 0.0978 | 0.2355 | 0.3234 | 0.5629 | 0.1652 |
| OTT-QA | A54 (tt+tc+sp+cc) | 0.3502 | 0.6390 | 0.7238 | 0.8971 | 0.4794 |
| MIMO-EN | A31 (tc+tp+sp+cc+sc) | 0.2460 | 0.4206 | 0.5159 | 0.7241 | 0.3599 |
| MIMO-CH | A28 (tc+tp+sp) | 0.3503 | 0.4822 | 0.5708 | 0.7495 | 0.5201 |
| E2E-WTQ | A24 (tc+tp) | 0.0000 | 0.0000 | 0.0164 | 0.3934 | 0.0175 |
| MMQA | A17 (tc+sc) | 0.3131 | 0.5732 | 0.6713 | 0.8499 | 0.7679 |

### QA 精排

| 資料集 | Config | R@1 | R@5 | R@10 | R@50 | MRR |
|---|---|---|---|---|---|---|
| FeTaQA | A20 | 0.0060 | 0.0240 | 0.0479 | — | 0.0210 |
| OTT-QA | A54 | 0.4477 | 0.6841 | 0.7744 | — | 0.5566 |
| MIMO-EN | A31 | 0.2751 | 0.5058 | 0.5971 | — | 0.4476 |
| MIMO-CH | A28 | 0.3603 | 0.5073 | 0.5752 | — | 0.5367 |
| E2E-WTQ | A24 | 0.0000 | 0.0492 | 0.0820 | — | 0.0353 |
| MMQA | A17 | 0.3165 | 0.5382 | 0.6458 | — | 0.7709 |

---

## 觀察與解讀

### OTT-QA / MIMO / MMQA（test⊂train，候選池 = train 圖）
這些資料集的 dev/test 表全部包含在 train 中，full corpus 就是 train 圖（22×更大）。
- **OTT-QA 表現最穩**：R@10 從 0.987 降到 0.774（−0.213），仍有合理的檢索能力。模型在 8889 張表中能找到 77% 的答案表，說明 GNN 的表示學習有效。
- **MMQA 下降中等**（−0.183），MIMO 下降幅度更小（−0.11 ~ −0.22）。

### FeTaQA / E2E-WTQ（disjoint，dev/test 表在訓練時從未見過）
- **FeTaQA 嚴重下滑**（TWIG 0.976→0.323，QA 0.980→0.048）：候選池從 501 張擴為 8328 張，其中 7326 張是 TWIG 訓練用的表——模型把 test 查詢推向熟悉的訓練表，導致 test 答案表幾乎排不上去。這是典型的 **training distribution bias**。
- **E2E-WTQ 近乎崩潰**（TWIG R@10=0.016）：train 有 621 張表、test 只有 61 張，候選池 12× 倍增後，test 的 61 張小表完全淹沒在 train 的 621 張「熟悉」表裡。

### Full corpus 設定的意義
Full corpus 數字揭示了 **in-domain bias** 問題：
- FeTaQA/E2E-WTQ 的分割方式（disjoint tables）讓 full corpus 評估變成 **open-domain 難題**，而不是原始論文設計的封閉語料庫設定。
- OTT-QA/MIMO/MMQA 的分割方式（test⊂train）下，full corpus 才是合理的評估設定，因為模型看過所有表，問題只是「在所有表裡找到正確的那張」。

### 建議（供論文選擇）
- 若與原始 FeTaQA/E2E-WTQ 論文比較 → 用 **per-split 設定**（符合原始評估協定）。
- 若強調「在真實大型語料庫上的檢索能力」→ 用 **full corpus 設定 for OTT-QA/MIMO/MMQA**（這幾個才公平）；FeTaQA/E2E-WTQ 則需說明 train/test 表不重疊、full corpus 評估有 domain shift 問題。
- 最誠實的方式：**兩種都報告**，並在論文 Setup 章節明確說明候選池的定義。

---

*Full corpus 圖建構：`build_full_corpus_graph.py`（embedding-based dedup for mimo/mmqa）*
*模型：dev-best TWIG + QA checkpoints，`train_twig_qa_best.py`*
