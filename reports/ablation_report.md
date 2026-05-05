# 消融實驗結果報告

## 完成狀態

資料集 | 完成配置 | 總配置 | 完成度
---|---|---|---
ottqa | 64/64 | 64 | ✅ 100%
feta | 64/64 | 64 | ✅ 100%
mimo_ch | 64/64 | 64 | ✅ 100%
mimo_en | 64/64 | 64 | ✅ 100%
mmqa | 64/64 | 64 | ✅ 100%
**e2ewtq** | **32/64** | 64 | ⚠️ **50%**

## E2EWTQ 缺失配置分析

### 缺失的配置
E2EWTQ 缺少所有包含 `sp` (same_page) 邊的 32 個配置:
- 配置編號: 4-7, 12-15, 20-23, 28-31, 36-39, 44-47, 52-55, 60-63

### 原因分析

**這不是 bug,是正確的行為!**

1. **E2EWTQ 資料集特性**:
   - 每個表格都是獨立的 CSV 文件
   - `page_title` = `file_name` (每個文件一張表)
   - 統計數據: 621 tables, 621 pages (1:1 mapping)
   
2. **Same_page 邊建立條件** (from build_graph.py):
   ```python
   for t_idxs in page_to_table_idxs.values():
       if len(t_idxs) <= 1:  # 需要同一 page 下至少 2 張 table
           continue
   ```
   
3. **結果**:
   - E2EWTQ: 0 條 same_page 邊 (因為沒有任何 page 包含 >1 張 table)
   - MIMO_EN: 274 條 same_page 邊 (275 tables, 196 pages)
   - MMQA: 820 條 same_page 邊 (629 tables, 490 pages)

4. **程式碼邏輯** (run_edge_ablation.py 第 593-619 行):
   - 在訓練前檢查資料集圖是否包含配置要求的所有邊類型
   - 如果缺少任何邊類型,自動跳過該配置
   - E2EWTQ 沒有 same_page 邊,因此跳過了所有需要 sp 邊的配置

### MIMO 和 MMQA 為什麼有 same_page 邊?

**關鍵誤解澄清**:
- MIMO 和 MMQA **有** page 節點!
- 它們的數據來源不是單獨的 CSV,而是包含多張表的 Excel 文件或數據庫
- 同一個來源(file/database)下可以有多張相關的表
- 例如 MMQA 的 page titles: "department", "management", "city" (這些是數據庫表名,同一問題可能涉及多張表)

### 結論

**E2EWTQ 的結果是正確且完整的**:
- 該資料集物理上不可能有 same_page 邊
- 程式正確地跳過了需要 sp 邊的配置
- 完成的 32 個配置覆蓋了所有該資料集可行的邊組合

### 建議

1. **接受當前結果**: E2EWTQ 的 32 個配置已經是該資料集的全部有效配置
2. **更新文檔**: 在 LaTeX 表格中添加註釋說明 E2EWTQ 不支持 same_page 邊
3. **分析時排除**: 比較不同資料集時,考慮它們的資料源特性差異
