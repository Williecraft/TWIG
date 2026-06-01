# Reviewer 意見中文整理

> 原始評語來源：[Reviewer.md](Reviewer.md)  
> 論文：*Weaving the Table Web: Inductive Graph Neural Networks for Table Retrieval*  
> 決定：**Reject**

## 一、Meta-Review 總結

論文提出一個 retrieval framework，將 table corpus 建模為 heterogeneous graph，以捕捉 table 之間的依賴關係，應用於 open-domain QA。

### 肯定之處

- 問題具有實務重要性。
- 方法簡單、容易理解。
- 具有一定程度的可重現性。
- 在多個資料集上進行實驗。

### 主要問題

- 新穎性有限。
- 技術深度不足。
- 對 effectiveness 與 efficiency 的宣稱缺乏足夠實證。
- Baseline 不夠強，無法充分證明方法優勢。

## 二、三位 Reviewer 的共同意見

### 1. 研究問題有價值

三位 reviewer 都認為 table retrieval 或 table QA 是具有研究價值的問題。論文注意到：若每張 table 都獨立編碼，就無法充分使用跨 table 的結構與語意關係。嘗試建模 inter-table dependencies 是合理方向。

### 2. 方法容易理解，但貢獻不夠新

目前方法的核心是：

- 將 table、column、page 建模為 heterogeneous graph 中的 nodes。
- 使用 structural edges 與 semantic edges 連接 nodes。
- 使用 relation-aware GraphSAGE 更新 table embeddings。

Reviewer 認為這些設計直覺且容易實作，但也相當標準。Graph-based retrieval、relation-aware schema modeling、structure-aware retrieval 等方向已有相關研究，因此目前論文較像是將既有技術應用到 open-domain table QA，而不是提出新的 modeling principle。

### 3. Method 細節不足

Reviewer 認為 Method 只描述高層次概念，缺少足夠技術細節。例如：

- Graph 中各類 node 和 edge 的完整定義。
- Section 3.1.1 中部分集合或符號的差異與計算方式。
- 相似度如何計算。
- 不同 edge type 如何產生。
- 如何決定各資料集應使用哪些 edge types。

原始 reviewer 匯出內容有部分公式遺失，因此無法確認 Section 3.1.1 中 reviewer 指的是哪些符號。修稿時應直接回頭檢查該節所有集合、範圍、公式與符號定義。

### 4. 效率宣稱缺乏數據

論文強調不需要 offline LLM-based pseudo-query generation，因此較有效率。但 reviewer 認為這不足以支持 efficiency claim，因為 graph construction 與 similarity-edge computation 也可能產生明顯成本。

需要補充：

- Graph construction time。
- Table-table 與 column-column similarity 計算時間。
- Indexing time。
- Memory usage。
- Graph storage size。
- Training time。
- Query latency。
- 不同 corpus scale 下的成本變化。

### 5. Baseline 不夠強

目前比較偏向 query augmentation 與部分 LLM-supervised retrievers。Reviewer 認為缺少更直接相關的比較對象，例如：

- Graph-aware retrieval。
- Structure-aware retrieval。
- Connectivity-aware retrieval。
- 與 TWIG 使用相近結構資訊的方法。

由於方法的新穎性受到質疑，更需要和最接近的 prior work 與 strong baselines 公平比較。

### 6. 實驗結果不足以支持優勢

Reviewer 指出：

- FeTaQA 的 recall improvement 值得肯定。
- Edge-type ablation 有分析價值。
- 但在多數資料集上，既有方法仍優於 TWIG。
- 論文宣稱 TWIG 能兼顧 retrieval effectiveness 與 computational efficiency，目前證據不足。

結果段落應誠實區分：

- TWIG 表現明顯較好的資料集。
- 提升有限的資料集。
- 表現落後的資料集。
- 哪些 dataset characteristics 可能影響效果。

### 7. Edge configuration 的選擇方式不合理

Reviewer 特別指出：論文使用 ablation study 找出每個 dataset 的最佳 edge configuration，再將該設定用於 retrieval 比較，這種做法不合理。

問題包括：

- Ablation 不應直接變成 test-time model selection。
- 方法沒有提供可執行的規則，決定新資料集該使用哪些 edge types。
- 若每個 dataset 都需要看結果後再挑最佳設定，方法較難泛化。

需要改成：

- 只使用 validation set 選擇 edge configuration。
- 清楚說明 selection protocol。
- 或提出固定設定與可泛化的自動選擇方法。
- 額外報告 fixed configuration 和 dataset-specific configuration 的差異。

### 8. 論文寫作品質需要改善

Reviewer 指出：

- Related Work 開頭包含問題描述，較適合移到 Method。
- Related Work 對 prior work 的討論太少。
- Method 描述過於簡略。
- Dataset 只被列舉，沒有交代來源與細節。

## 三、各 Reviewer 意見翻譯

## Reviewer HrDN

### 評分

| 項目 | 評分 |
| --- | --- |
| 與會議主題相關性 | Relevant |
| Relevance Score | 0 fair |
| Novelty | -1 poor |
| Technical Soundness | 0 fair |
| Quality of Presentation | 0 fair |
| Overall Recommendation | -2 reject |

### 優點

- 論文處理的是合理且具有動機的問題。
- 獨立編碼 table 可能無法捕捉 multi-table retrieval signals 與 corpus-level structural dependencies，這確實是 table retrieval 的限制。
- 將 table、column 和 page 建模為 heterogeneous graph nodes，再使用 structural 與 semantic edges 連接，概念清楚且容易實作。
- 不依賴 offline LLM-based pseudo-query generation，具有實務吸引力。
- FeTaQA 的 recall improvement 值得肯定。
- Edge-type ablation 能幫助分析不同 relations 對各資料集的作用。

### 缺點

- 核心想法缺乏新穎性。Graph-based、relation-aware 與 structure-aware retrieval 已有相關研究。
- Table-column-page graph 與 relation-aware GraphSAGE 都相當標準，沒有新的 modeling principle。
- 論文沒有充分解釋為何 TWIG 應該優於過去的 graph-aware 或 structure-aware retrieval approaches。
- 沒有量化 graph construction 與 similarity-edge computation 的成本。
- Corpus scale 的 top-k similarity edges 可能帶來 indexing、memory 與 runtime overhead。
- Baseline comparison 偏弱，缺少更直接相關的方法。

### Reviewer 的整體結論

問題具有意義，方法容易理解，FeTaQA 結果與 edge ablation 也有價值。但目前在 novelty、technical depth、efficiency validation 與 baseline selection 上仍有重大缺點，因此尚不足以接受。

## Reviewer hZJw

### 評分

| 項目 | 評分 |
| --- | --- |
| 與會議主題相關性 | Table-based QA 是 SIGIR 關注的領域 |
| Relevance Score | 0 fair |
| Novelty | -1 poor |
| Technical Soundness | -1 poor |
| Quality of Presentation | 0 fair |
| Overall Recommendation | -2 reject |

### 優點

- 論文處理活躍研究領域中的重要問題：table-based QA 與 reasoning。
- 方法大致上說明清楚，但仍有改善空間。
- 使用容易取得的 components，因此多數內容可重現。
- 在公開 benchmark 上進行評估。

### 缺點

- Method components 的技術描述有限。
- 貢獻主要停留在 table representation。
- 和現有 baselines 相比，改善幅度有限。
- Related Work 以 problem description 開頭，內容放置位置不恰當。
- Related Work 對 prior work 的討論不足。
- Method 只有高層次 graph representation 描述。
- Evaluation 只列出資料集，沒有說明資料集來源和細節。
- 多數資料集上，既有方法優於本文方法。

### Reviewer 的整體結論

論文需要大幅修改寫作品質，並進一步發展 methodology，才有機會被接受。

## Reviewer s7TT

### 評分

| 項目 | 評分 |
| --- | --- |
| 與會議主題相關性 | Table Retrieval 與 SIGIR 高度相關 |
| Relevance Score | 2 excellent |
| Novelty | 0 fair |
| Technical Soundness | -1 poor |
| Quality of Presentation | 0 fair |
| Overall Recommendation | -2 reject |

### 優點

- Table retrieval 具有重要實務價值。
- 實驗涵蓋多個資料集，包括 single-table 與 multi-table scenarios。
- 論文提供 ablation studies，分析不同 edge types 對 retrieval performance 的影響。

### 缺點

- Method 描述不夠清楚。
- Section 3.1.1 中部分集合、範圍與計算方式未交代完整。
- 使用 ablation 找出各 dataset 最佳 edge configuration，再套用於比較，做法不合理。
- 多個資料集上，TWIG 和 baselines 仍有明顯差距。
- 「兼顧 retrieval effectiveness 與 computational efficiency」的宣稱缺乏支持。
- 雖然提出多種 edge types，但沒有提供可執行的方法來決定該使用哪些 edges。

### Minor Issue

- Table 1 的 MMQA R@2：應將 `52.24` 加粗，而不是 `51.40`。

### Reviewer 的整體結論

方法描述不夠清楚，實驗也無法有說服力地證明方法優勢。論文需要大幅修改。

## 四、建議修改優先順序

## P0：必須先修正

- [ ] 補完整 Method：node types、edge types、公式、符號、相似度與 graph construction 流程。
- [ ] 檢查 Section 3.1.1 所有集合與範圍，修正 reviewer 指出的不清楚之處。
- [ ] 改正 edge configuration selection protocol，避免使用 test set 或 ablation 結果直接挑最佳設定。
- [ ] 補強 strong baselines，尤其是 graph-aware、structure-aware 與 connectivity-aware retrieval。
- [ ] 修改 Results 的主張，避免將部分資料集上的提升描述成全面優勢。

## P1：直接影響說服力

- [ ] 補充 graph construction、indexing、training、memory 與 query latency 數據。
- [ ] 增加 fixed configuration 與 dataset-specific configuration 的比較。
- [ ] 解釋哪些 dataset characteristics 使 TWIG 有效或失效。
- [ ] 明確區分本文的新貢獻和既有 GraphSAGE、heterogeneous graph 技術。
- [ ] 在 Related Work 中加入最接近方法的公平比較。

## P2：寫作與呈現

- [ ] 將 Related Work 中的方法描述移至 Method。
- [ ] 擴充 Related Work，不要只簡短列舉 prior work。
- [ ] 補上每個 dataset 的來源、規模、split 與特性。
- [ ] 修正 Table 1 中 MMQA R@2 的粗體：`52.24` 應加粗，`51.40` 不應加粗。
- [ ] 檢查表格、符號和圖說能否獨立理解。

## 五、修稿時可使用的回應框架

重新撰寫論文時，可以用以下四個問題檢查全文：

1. **TWIG 到底新增了什麼？**  
   不只說使用 heterogeneous graph，也要指出和既有 graph-aware retrieval 方法相比的新設計、新分析或新實證發現。

2. **方法能否被完整重現？**  
   讀者應能從論文中重建 nodes、edges、features、training、inference 與 edge selection 流程。

3. **比較是否公平？**  
   所有設定選擇都應基於 validation set 或固定規則，並加入直接相關的 strong baselines。

4. **效率宣稱是否有量化證據？**  
   不依賴 LLM 是優點，但仍須報告 graph construction、indexing、storage 與 online retrieval 成本。

