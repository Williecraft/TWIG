# 糾正後的 Same_Page 邊分析

## 你的觀察是對的！

### 原始資料集狀態

1. **MIMO_EN**:
   - 原始資料只有 `file_name` (Excel 文件路徑)
   - metadata 中只有 `original_path`
   - **沒有明確的 page/section 信息**

2. **MMQA**:
   - 原始資料有 `file_name` (JSON 文件路徑)和 `sheet_name` (表名)
   - metadata 中有 `table_section_title` (如 "department", "management", "city")
   - **table_section_title 是表的名稱,不是頁面標題**

3. **E2EWTQ**:
   - 原始資料有 `file_name` (CSV 文件路徑)
   - sheet_name 為空
   - metadata 中沒有任何 page 相關信息

### Build_Graph.py 的 Page_Title 推導邏輯

```python
page_title = metadata.get('table_page_title') or \
             metadata.get('title') or \
             metadata.get('table_section_title') or \  # MMQA 用這個
             item.get('file_name') or \                # MIMO_EN 用這個
             f"__UNKNOWN_PAGE_{table_id}__"
```

### Same_Page 邊的產生原因

#### MIMO_EN (275 tables → 196 pages → 274 same_page edges)
- **page_title = file_name** (因為沒有其他 metadata)
- 某些 Excel 文件包含多個 sheet
- 例如: `2015-2019JiangsuCollegeEntranceExaminationScienceandLiberalArtsScoreDistributionStatistics.xlsx` 包含 5 張表
- 39 個文件包含多張表 → 產生 same_page 邊

#### MMQA (629 tables → 490 pages → 820 same_page edges)
- **page_title = table_section_title** (如 "department", "management")
- 同一個 JSON 文件中的多個表可能有相同的 section_title
- 例如: "department" 這個 section_title 被 4 張表使用
- 61 個 section_title 被多張表共享 → 產生 same_page 邊

#### E2EWTQ (621 tables → 621 pages → 0 same_page edges)
- **page_title = file_name** (因為沒有其他 metadata)
- 每個 CSV 文件只包含一張表
- 621 個文件 = 621 張表 (1:1 mapping)
- **沒有任何文件包含多張表** → 0 條 same_page 邊

## 結論

你的記憶是正確的:**MIMO 和 MMQA 原始資料集確實沒有真正的 "page" 概念**。

Same_page 邊的產生是因為:
1. **MIMO**: 同一個 Excel 文件的多個 sheet 被視為同一 "page"
2. **MMQA**: 同名的 table_section_title 被視為同一 "page"
3. **E2EWTQ**: 每個 CSV 獨立成 "page",所以沒有 same_page 邊

這種設計**合理但有語義上的混淆**:
- "same_page" 更像是 "same_source_file" 或 "same_section"
- 對 MIMO 來說是 "同一 Excel 文件的不同 sheet"
- 對 MMQA 來說是 "同一 section/category 的不同表"
- 對 E2EWTQ 來說這個概念不存在(每個文件一張表)

## 消融實驗結果的正確性

**E2EWTQ 跳過包含 sp 邊的 32 個配置是完全正確的**,因為該資料集物理上不存在同一來源下的多張表。
