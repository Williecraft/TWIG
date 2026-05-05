from OllamaAgent import Agent as oAgent
# from GeminiAgent import Agent as gAgent
import datetime
import json
import os
import tqdm

SOURCE = "dev/datagov"
INPUT_FILE = "table.jsonl"
OUTPUT_FILE = "query_ollama.jsonl"
MAX_RETRIES = 5  # LLM 輸出不合規時的重試次數
FIX = True # 從頭開始檢查有沒有生成失敗的
MODEL = "Ollama"

VERBOSE = False
TABLE_PATH = f"/user_data/TabGNN/data/table/{SOURCE}/{INPUT_FILE}"
QUERY_PATH = f"/user_data/TabGNN/data/generated/{SOURCE}/{OUTPUT_FILE}"
LANGUAGE = "zh"  # 'en' or 'zh'

os.makedirs(os.path.dirname(QUERY_PATH), exist_ok=True)

def json2csv(jtable: dict, top=10) -> str:
    import csv
    import io
    import pandas as pd

    def clean_cell(x):
        if x is None:
            return ""
        return str(x).replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()

    def parse_csv_row(s: str) -> list[str]:
        s = clean_cell(s)
        if not s:
            return []
        try:
            return next(csv.reader(io.StringIO(s)))
        except Exception:
            try:
                return next(csv.reader([s]))
            except Exception:
                return [s]

    # ---- header ----
    header = jtable.get("header", [])
    if isinstance(header, str):
        header = parse_csv_row(header)
    elif isinstance(header, list):
        if len(header) == 1 and isinstance(header[0], str) and ("," in header[0]):
            header = parse_csv_row(header[0])
        else:
            header = [clean_cell(h) for h in header]
    else:
        header = [clean_cell(header)]

    # ---- instances ----
    instances = jtable.get("instances", [])
    rows = instances if top is None else instances[:int(top)]

    parsed_rows = []
    for r in rows:
        if isinstance(r, str):
            parsed_rows.append([clean_cell(x) for x in parse_csv_row(r)])
        elif isinstance(r, list):
            parsed_rows.append([clean_cell(x) for x in r])
        else:
            parsed_rows.append([clean_cell(r)])

    # ---- 對齊欄數 ----
    ncol = len(header)
    if ncol == 0:
        ncol = max((len(row) for row in parsed_rows), default=1)
        header = [f"col_{i}" for i in range(ncol)]

    fixed_rows = []
    for row in parsed_rows:
        if len(row) < ncol:
            row = row + [""] * (ncol - len(row))
        elif len(row) > ncol:
            row = row[:ncol]
        fixed_rows.append(row)

    df = pd.DataFrame(fixed_rows, columns=header)
    return df.to_csv(index=False)


def is_valid_output(obj):
    """
    規則：
      - 必須是 dict，含 'headers'（list[str]）與 'questions'（list[str]）
      - headers 去除空字串/ 'nan' / 'Unnamed:' 後仍需非空
      - 問題數量 > 0.5 * headers 數
    """
    if not isinstance(obj, dict):
        return False, "Not a dict"
    if "headers" not in obj or "questions" not in obj:
        return False, "Missing keys"
    headers = obj["headers"]
    questions = obj["questions"]
    if not isinstance(headers, list) or not all(isinstance(h, str) for h in headers):
        return False, "Headers not list[str]"
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        return False, "Questions not list[str]"

    # 清掉無意義表頭
    def bad(h):
        hs = h.strip()
        if hs == "": return True
        lower = hs.lower()
        return lower == "nan" or lower.startswith("unnamed")
    cleaned = [h for h in headers if not bad(h)]
    if len(cleaned) == 0:
        return False, "No valid headers after cleaning"

    # min_q = math.floor(len(cleaned) / 2) + 1  # > 0.5 * headers
    # if len(questions) < min_q:
    #     return False, f"Too few questions (< {min_q})"

    return True, ""

def parse_json_strict(s):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    s = s.replace("```json", "").replace("```", "").strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("No JSON object found")

    return json.loads(s[start:end+1])

with open(TABLE_PATH, "r", encoding="utf-8") as f:
    table_lines = [json.loads(line) for line in f.readlines()]

if not os.path.exists(QUERY_PATH):
    with open(QUERY_PATH, "w", encoding="utf-8"): pass

with open(QUERY_PATH, "r", encoding="utf-8") as f:
    query_lines = [json.loads(line) for line in f.readlines()]

with open("/user_data/TabGNN/config/api_keys.json", "r", encoding="utf-8") as f:
    api_key = json.load(f)[MODEL]

with open("/user_data/TabGNN/results/log.txt", "w", encoding="utf-8"): pass
def write_log(text):
    with open("/user_data/TabGNN/results/log.txt", "a", encoding="utf-8") as f:
        f.write(text+"\n"+"="*50+"\n")

if not os.path.exists("/user_data/TabGNN/results/progress.json"):
    with open("/user_data/TabGNN/results/progress.json", "w", encoding="utf-8") as f: 
        json.dump({}, f)

with open("/user_data/TabGNN/results/progress.json", "r", encoding="utf-8") as f:
    progress = json.load(f)

if progress.get(SOURCE) is None:
    progress[SOURCE] = {OUTPUT_FILE:0}
    with open("/user_data/TabGNN/results/progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)
elif progress[SOURCE].get(OUTPUT_FILE) is None:
    progress[SOURCE][OUTPUT_FILE] = 0
    with open("/user_data/TabGNN/results/progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)

def save_queries():
    with open(QUERY_PATH, "w", encoding="utf-8") as f:
        for line in query_lines:
            f.write(json.dumps(line, ensure_ascii=False)+"\n")

if MODEL == "Ollama":
    agent = oAgent(api_key=api_key)
# elif MODEL == "Gemini":
#     agent = gAgent(api_key=api_key)

PROMPT = """\
You are an expert in table data analysis.
Given a table with its file name, sheet name, and a portion of its content (first ten rows), your task is to **extract key headers and generate questions** based on the table & headers.

Important Considerations:
• The table may contain nan or Unnamed: values, which represent empty merged cells in the original table. These **should not** be considered as meaningful data points or headers.
• The **true column headers may not always be in the first row or first column**. Carefully analyze the table to identify the correct headers.
• If the table has **multi-level headers**, preserve the hierarchical structure without merging or altering the text.
• If the table has an **irregular header structure** (such as key-value formatted headers where column names are listed separately), extract the correct header names accordingly.
• **Ignore rows that contain mostly empty values (nan, Unnamed:) or placeholders without meaningful data.**
• **Do not generate python code, extract headers and questions on your own.**
• The type of Questions could be one of (lookup, calculate, visualize, reasoning).
• **Generate questions using ONLY the language of the table.**
• **The output language MUST strictly follow the table language (based on headers and content).**
• **NEVER use a different language, even if the user prompt is in another language.**

Tasks:
1. Extract Header Names:
• Identify the **true headers** by analyzing the structure of the table.
• **Exclude** placeholder values like "nan" and "Unnamed:".
• If the table contains **multi-level headers**, keep them as separate levels without merging.
• If the table has **key-value headers**, extract the correct column names.

2. Generate Questions (Context-Specific to the Table):
• Formulate **questions that can only be answered using this specific table**.
• Ensure **each question involves 1 to 3 different headers** to capture interactions between data & columns.
• Ensure the header diversity in all the questions.
• Use ” to mark the headers in the question.
• **Total number of questions should larger than the half number of extracted headers**

Avoid vague or biographical questions. Use only values and headers from the preview.

**Output Format (Strictly JSON format)**
Only return a single valid JSON object without any other text:
{{ "headers": ["header1", "header2", "..."], "questions": ["question1", "question2", "..."] }}

**Table Meta**
- File: {file_name}
- Sheet: {sheet_name}

**Table Preview**
{table_csv_preview}
"""

PROMPT_CH = """\
你是一個「純 JSON 產生器」，不是助理、不是解說員。

你的唯一任務是：
輸出一個可以被 Python json.loads() 直接解析的 JSON 物件。

========================
【輸出硬性規則（違反即視為錯誤）】
========================
- 只能輸出 JSON
- 不可輸出任何說明文字
- 不可輸出任何前言、結語、理由、解釋
- 不可輸出 markdown（例如 ```json）
- 不可輸出註解
- 不可輸出多個 JSON
- 不可輸出 JSON 以外的任何字元

【格式強制】
- 第一個字元必須是：{{
- 最後一個字元必須是：}}
- 中間只能是合法 JSON

========================
【JSON Schema（必須完全符合）】
========================

{{
  "headers": ["欄位1", "欄位2", "..."],
  "questions": ["問題1", "問題2", "..."]
}}

========================
【錯誤處理（非常重要）】
========================
若你無法判斷、或資料不足，也必須輸出合法 JSON：

{{
  "headers": [],
  "questions": []
}}

嚴禁輸出任何解釋文字。

========================
【任務說明】
========================
你是一位表格資料分析專家。

請根據提供的表格內容：
1. 抽取正確的欄位名稱（headers）
2. 產生高品質問題（questions）

========================
【語言規則（極重要）】
========================
- 必須根據「表格內容」判斷語言
- 所有問題必須 100% 使用表格語言
- 嚴禁混用語言
- 嚴禁翻譯欄位名稱

========================
【欄位抽取規則】
========================
- 忽略 "nan"、"Unnamed:"
- 真正欄位可能不在第一列
- 若為 multi-level header，保留層級
- 若為 key-value 結構，需正確抽取欄位名稱
- 忽略無意義或空值列

========================
【問題生成規則】
========================
- 問題必須只能透過此表格回答
- 每個問題需涉及 1～3 個欄位
- 問題中引用欄位必須用「」標示
- 問題需自然、具體
- 不可產生不存在的概念

問題類型可包含：
- 查找（lookup）
- 計算（calculate）
- 趨勢（visualize）
- 推論（reasoning）

========================
【數量規則】
========================
- 問題數量必須 > 欄位數量的一半

========================
【最終檢查（輸出前）】
========================
請在輸出前自行確認：
- 是否為合法 JSON
- 是否沒有任何多餘文字
- 是否符合 schema
- 是否語言正確
- 是否沒有虛構欄位

========================
【表格資訊】
========================

Table Meta
- File: {file_name}
- Sheet: {sheet_name}

Table Preview
{table_csv_preview}
"""

def main():
    # -------- 決定這次要處理哪些 table --------
    if FIX:
        target_indices = []
        for i in range(len(table_lines)):
            if i < len(query_lines) and isinstance(query_lines[i], dict) and query_lines[i].get("error") is not None:
                target_indices.append(i)
    else:
        start = progress[SOURCE][OUTPUT_FILE]
        target_indices = list(range(start, len(table_lines)))

    success_count = 0
    fail_count = 0

    for idx, i in enumerate(
        tqdm.tqdm(
            target_indices,
            total=len(target_indices),
            desc=f"{SOURCE}/{OUTPUT_FILE}" + (" [FIX]" if FIX else "")
        ),
        1
    ):
        table: dict = table_lines[i]

        if FIX and VERBOSE:
            print(f"Fixing table index {i+1} (ID:{table['id']})")

        file_name = table["file_name"]
        sheet_name = table["sheet_name"]
        table_id = table["id"]

        now = datetime.datetime.now() + datetime.timedelta(hours=8)
        if VERBOSE:
            print(now)
            print(f"Progress: {idx}/{len(target_indices)} (table_index={i+1}/{len(table_lines)})")

        write_log(f"Progress: {idx}/{len(target_indices)} (table_index={i+1}/{len(table_lines)})")
        write_log(f"Generating query for table: (ID:{table_id}) {sheet_name}")

        prompt_template = PROMPT_CH if LANGUAGE == "zh" else PROMPT

        base_prompt = prompt_template.format(
            file_name=file_name,
            sheet_name=sheet_name,
            table_csv_preview=json2csv(table, top=10),
        )

        attempt = 0
        final_obj = None
        while attempt <= MAX_RETRIES:
            prompt = base_prompt if attempt == 0 else (
                base_prompt + "\n\nRespond with **only** a valid JSON object matching the required schema."
                if LANGUAGE == "en"
                else base_prompt +
                "\n\n你的上一個回覆格式錯誤。這次只能輸出可被 json.loads() 直接解析的單一 JSON 物件。" +
                "\n禁止輸出任何解釋、禁止 markdown、禁止 code fence、禁止額外文字。" +
                "\n第一個字元必須是 {{，最後一個字元必須是 }}。"
            )

            write_log(prompt)
            response = agent.query(prompt)
            write_log(response)

            response = response.replace("```json", "").replace("```", "").strip()
            if VERBOSE:
                print(response)

            try:
                obj = parse_json_strict(response)
                ok, msg = is_valid_output(obj)
                if ok:
                    final_obj = obj
                    break
                else:
                    write_log(f"[VALIDATION FAIL] table_id={table_id}: {msg}")
            except Exception as e:
                write_log(f"[JSON PARSE ERROR] table_id={table_id}: {e}")

            attempt += 1

        if final_obj is None:
            result = {
                "table_id": table_id,
                "sheet_name": sheet_name,
                "headers": [],
                "questions": [],
                "error": "LLM output invalid after retries"
            }
            fail_count += 1
            status = "FAIL"
        else:
            result = {
                "table_id": table_id,
                "sheet_name": sheet_name,
                "headers": final_obj["headers"],
                "questions": final_obj["questions"]
            }
            success_count += 1
            status = "SUCCESS"

        while i >= len(query_lines):
            query_lines.append({})

        query_lines[i] = result
        save_queries()

        if not FIX:
            progress[SOURCE][OUTPUT_FILE] = i + 1
            with open("/user_data/TabGNN/results/progress.json", "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=4)

        stat_msg = (
            f"[{status}] table_index={i+1}/{len(table_lines)} "
            f"(processed={idx}/{len(target_indices)}) "
            f"success={success_count} fail={fail_count}"
        )
        
        if VERBOSE: print(stat_msg)
        write_log(stat_msg)

        if VERBOSE:
            print("\n" + "=" * 50 + "\n")

    summary = (
        f"[SUMMARY] processed={len(target_indices)} "
        f"success={success_count} fail={fail_count}"
    )
    print(summary)
    write_log(summary)

if __name__ == "__main__":
    main()