import json
from pathlib import Path

INPUT_FILE = Path("/user_data/TabGNN/data/downloads/Synthesized_two_table.json")
OUTPUT_FILE = Path("/user_data/TabGNN/data/downloads/Synthesized_two_table_fixed.json")


def show_json_error(s: str, err: Exception, context_chars: int = 250):
    # 專門處理 json.JSONDecodeError
    if isinstance(err, json.JSONDecodeError):
        line = err.lineno
        col = err.colno
        pos = err.pos

        start = max(0, pos - context_chars)
        end = min(len(s), pos + context_chars)
        snippet = s[start:end]

        print("\n==== JSONDecodeError 定位 ====")
        print(f"Message : {err.msg}")
        print(f"Line    : {line}")
        print(f"Column  : {col}")
        print(f"Pos     : {pos}")
        print("---- 附近內容（含錯誤點）----")
        print(snippet)
        print("---- 錯誤點指示 ----")
        caret_pos = pos - start
        print(" " * caret_pos + "^")
        print("==== End ====\n")
    else:
        print("非 JSONDecodeError：", repr(err))


def try_load_json_verbose(s: str):
    try:
        return json.loads(s), None
    except Exception as e:
        return None, e


def main():
    raw = INPUT_FILE.read_text(encoding="utf-8", errors="replace")

    # 這裡假設 repaired 是你原本 repair_json_text(raw) 的結果
    # 你把 repaired = repair_json_text(raw) 放回來即可
    repaired = raw  # <- 先占位，請換成 repaired = repair_json_text(raw)

    obj, err = try_load_json_verbose(repaired)
    if err is not None:
        show_json_error(repaired, err)
        OUTPUT_FILE.write_text(repaired, encoding="utf-8")
        raise ValueError("修復後仍非合法 JSON（已輸出修復後原文，並印出錯誤位置）")

    OUTPUT_FILE.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("OK")


if __name__ == "__main__":
    main()
