import json
import re
from pathlib import Path
from tqdm import tqdm


# ====== 固定用這兩個路徑 ======
INPUT_FILE = Path("/user_data/TabGNN/data/downloads/Synthesized_two_table.json")
OUTPUT_FILE = Path("/user_data/TabGNN/data/downloads/Synthesized_two_table_fixed.json")
# =================================


TOKEN_REGEX = re.compile(
    r"""
    (?P<ws>\s+)|
    (?P<brace>[{}\[\],:])|
    (?P<string>"(?:\\.|[^"\\])*")|
    (?P<number>-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?)|
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def strip_bom(s: str) -> str:
    return s.lstrip("\ufeff")


def read_text_with_progress(path: Path, encoding="utf-8") -> str:
    total = path.stat().st_size
    chunks = []
    with path.open("rb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"Reading {path.name}") as pbar:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            chunks.append(b)
            pbar.update(len(b))
    return b"".join(chunks).decode(encoding, errors="replace")


def write_text_with_progress(path: Path, text: str, encoding="utf-8") -> None:
    data = text.encode(encoding, errors="replace")
    total = len(data)
    with path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"Writing {path.name}") as pbar:
        offset = 0
        step = 1024 * 1024
        while offset < total:
            chunk = data[offset : offset + step]
            f.write(chunk)
            offset += len(chunk)
            pbar.update(len(chunk))


def remove_json_comments(s: str) -> str:
    """移除 //... 與 /*...*/ 註解（避開字串）"""
    s = strip_bom(s)
    out = []
    i, n = 0, len(s)
    in_str = False
    esc = False

    while i < n:
        ch = s[i]

        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        if ch == "/" and i + 1 < n and s[i + 1] == "/":
            i += 2
            while i < n and s[i] not in ("\n", "\r"):
                i += 1
            continue

        if ch == "/" and i + 1 < n and s[i + 1] == "*":
            i += 2
            while i + 1 < n and not (s[i] == "*" and s[i + 1] == "/"):
                i += 1
            i = min(i + 2, n)
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def single_quotes_to_double(s: str) -> str:
    """把 '...' 字串轉成 JSON 的 "..."（避開雙引號字串）"""
    out = []
    i, n = 0, len(s)
    in_dq = False
    dq_esc = False

    while i < n:
        ch = s[i]

        if in_dq:
            out.append(ch)
            if dq_esc:
                dq_esc = False
            elif ch == "\\":
                dq_esc = True
            elif ch == '"':
                in_dq = False
            i += 1
            continue

        if ch == '"':
            in_dq = True
            out.append(ch)
            i += 1
            continue

        if ch == "'":
            i += 1
            buf = []
            esc = False
            while i < n:
                c = s[i]
                if esc:
                    buf.append(c)
                    esc = False
                    i += 1
                    continue
                if c == "\\":
                    buf.append(c)
                    esc = True
                    i += 1
                    continue
                if c == "'":
                    i += 1
                    break
                buf.append(c)
                i += 1

            content = "".join(buf)
            content = content.replace("\\", "\\\\").replace('"', '\\"')
            out.append('"')
            out.append(content)
            out.append('"')
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def escape_control_chars_in_strings(s: str) -> str:
    """
    把字串內不允許的控制字元（<0x20）轉義：
    \n \r \t -> \\n \\r \\t，其餘 -> \\u00XX
    """
    out = []
    in_str = False
    esc = False

    for ch in s:
        if in_str:
            if esc:
                out.append(ch)
                esc = False
                continue

            if ch == "\\":
                out.append(ch)
                esc = True
                continue

            if ch == '"':
                out.append(ch)
                in_str = False
                continue

            o = ord(ch)
            if o < 0x20:
                if ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    out.append("\\r")
                elif ch == "\t":
                    out.append("\\t")
                else:
                    out.append(f"\\u{o:04x}")
                continue

            out.append(ch)
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
        else:
            out.append(ch)

    return "".join(out)


def fix_unterminated_strings_by_structure(s: str, lookahead: int = 64) -> str:
    """
    修復常見情況：字串少了結尾引號，後面緊接大量結構結尾，例如:
      "Grant Ave...]]}]}]
    規則（heuristic）：
    - 在字串狀態中遇到 ']' 或 '}' 時
    - 往後看一小段，如果只包含空白與 ,]} 這些結構符號（且不含新的雙引號）
      => 推斷缺少結尾引號，於該位置補上一個 '"'
    """
    out = []
    i, n = 0, len(s)
    in_str = False
    esc = False

    allowed_after = set(" \t\r\n,]}")

    while i < n:
        ch = s[i]

        if in_str:
            if esc:
                out.append(ch)
                esc = False
                i += 1
                continue

            if ch == "\\":
                out.append(ch)
                esc = True
                i += 1
                continue

            if ch == '"':
                out.append(ch)
                in_str = False
                i += 1
                continue

            # ⭐ 在字串中遇到結構 closing，嘗試判斷是否缺少結尾引號
            if ch in ("]", "}"):
                j = i
                end = min(n, i + lookahead)
                suspicious = True
                while j < end:
                    c = s[j]
                    if c == '"':
                        suspicious = False
                        break
                    if c not in allowed_after:
                        suspicious = False
                        break
                    j += 1

                if suspicious:
                    # 補上缺少的結尾引號，並結束字串狀態，但不消耗目前的 ch
                    out.append('"')
                    in_str = False
                    continue

            out.append(ch)
            i += 1
            continue

        # not in string
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        out.append(ch)
        i += 1

    # 如果真的跑到 EOF 還在字串裡，最後補一個引號（最後保底）
    if in_str:
        out.append('"')

    return "".join(out)


def quote_unquoted_keys(s: str) -> str:
    """把 {a:1} 變 {"a":1}（避開字串）"""
    out = []
    i, n = 0, len(s)
    in_str = False
    esc = False

    def is_key_start(c: str) -> bool:
        return c.isalpha() or c in ("_", "$")

    def is_key_char(c: str) -> bool:
        return c.isalnum() or c in ("_", "$", "-")

    while i < n:
        ch = s[i]

        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        if ch in "{,":
            out.append(ch)
            i += 1

            j = i
            while j < n and s[j].isspace():
                out.append(s[j])
                j += 1

            if j < n and is_key_start(s[j]):
                k = j
                while k < n and is_key_char(s[k]):
                    k += 1

                t = k
                while t < n and s[t].isspace():
                    t += 1

                if t < n and s[t] == ":":
                    key = s[j:k]
                    out.append('"'); out.append(key); out.append('"')
                    out.append(s[k:t])
                    out.append(":")
                    i = t + 1
                    continue

            i = j
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def remove_trailing_commas(s: str) -> str:
    """移除尾逗號： ,] 以及 ,}（避開字串）"""
    out = []
    i, n = 0, len(s)
    in_str = False
    esc = False

    while i < n:
        ch = s[i]

        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < n and s[j].isspace():
                j += 1
            if j < n and s[j] in ("]", "}"):
                i += 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def fix_unmatched_brackets(s: str) -> str:
    """修復 {} / [] 括號不匹配（避開字串）"""
    stack = []
    out = []
    in_str = False
    esc = False

    for ch in s:
        out.append(ch)

        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                out.pop()

    while stack:
        out.append(stack.pop())

    return "".join(out)


def tokenize(s: str):
    pos, n = 0, len(s)
    while pos < n:
        m = TOKEN_REGEX.match(s, pos)
        if not m:
            yield ("raw", s[pos], pos)
            pos += 1
            continue
        kind = m.lastgroup
        val = m.group(kind)
        yield (kind, val, pos)
        pos = m.end()


def normalize_literals(kind: str, val: str):
    if kind != "name":
        return kind, val
    low = val.lower()
    if low in ("true", "false", "null"):
        return "name", low
    if val in ("True", "False"):
        return "name", val.lower()
    if val == "None":
        return "name", "null"
    if val in ("NaN", "nan", "Infinity", "inf", "-Infinity", "-inf"):
        return "name", "null"
    return kind, val


def should_insert_comma(prev, curr) -> bool:
    pk, pv = prev
    ck, cv = curr

    if ck == "brace" and cv in ("]", "}", ",", ":"):
        return False
    if pk == "brace" and pv in ("{", "[", ":", ","):
        return False

    def is_value_end(k, v):
        if k in ("string", "number"):
            return True
        if k == "name" and v in ("true", "false", "null"):
            return True
        if k == "brace" and v in ("]", "}"):
            return True
        return False

    def is_value_start(k, v):
        if k in ("string", "number"):
            return True
        if k == "name" and v in ("true", "false", "null"):
            return True
        if k == "brace" and v in ("{", "["):
            return True
        return False

    return is_value_end(pk, pv) and is_value_start(ck, cv)


def fix_missing_commas(s: str) -> str:
    toks = []
    for kind, val, _ in tokenize(s):
        if kind == "ws":
            toks.append(("ws", val))
        elif kind == "raw":
            toks.append(("raw", val))
        else:
            kind, val = normalize_literals(kind, val)
            toks.append((kind, val))

    out = []
    for i in range(len(toks)):
        kind, val = toks[i]

        j = i + 1
        next_sig = None
        while j < len(toks):
            if toks[j][0] != "ws":
                next_sig = toks[j]
                break
            j += 1

        out.append(val)

        cur_sig = None if kind == "ws" else (kind, val)
        if cur_sig is not None and next_sig is not None:
            nk, nv = next_sig
            if should_insert_comma(cur_sig, (nk, nv)):
                out.append(",")

    return "".join(out)


def try_load_json(s: str):
    try:
        return json.loads(s), None
    except Exception as e:
        return None, e


def show_json_error(s: str, err: Exception, context_chars: int = 250):
    if isinstance(err, json.JSONDecodeError):
        line, col, pos = err.lineno, err.colno, err.pos
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
        print(" " * (pos - start) + "^")
        print("==== End ====\n")
    else:
        print("非 JSONDecodeError：", repr(err))


def repair_json_text(text: str) -> str:
    steps = [
        ("Remove comments", remove_json_comments),
        ("Single quotes -> double", single_quotes_to_double),
        ("Escape control chars in strings", escape_control_chars_in_strings),
        ("Fix unterminated strings (heuristic)", fix_unterminated_strings_by_structure),
        ("Quote unquoted keys", quote_unquoted_keys),
        ("Remove trailing commas", remove_trailing_commas),
        ("Fix missing commas (heuristic)", fix_missing_commas),
        ("Fix unmatched brackets", fix_unmatched_brackets),
        ("Remove trailing commas (again)", remove_trailing_commas),
    ]

    s = strip_bom(text)
    with tqdm(total=len(steps), desc="Repair steps") as pbar:
        for name, fn in steps:
            tqdm.write(f"[Step] {name} ...")
            s = fn(s)
            pbar.update(1)
    return s


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到輸入檔：{INPUT_FILE}")
    if INPUT_FILE.is_dir():
        raise IsADirectoryError(f"輸入路徑是資料夾不是檔案：{INPUT_FILE}")

    ensure_parent_dir(OUTPUT_FILE)
    raw = read_text_with_progress(INPUT_FILE)

    # 如果原檔其實就合法，直接格式化輸出（可改 indent=4）
    obj, err = try_load_json(raw)
    if err is None:
        text_out = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
        write_text_with_progress(OUTPUT_FILE, text_out)
        print(f"原檔合法 JSON，已輸出：{OUTPUT_FILE}")
        return

    repaired = repair_json_text(raw)

    obj2, err2 = try_load_json(repaired)
    if err2 is not None:
        show_json_error(repaired, err2)
        write_text_with_progress(OUTPUT_FILE, repaired)
        raise ValueError(
            "已嘗試修復（註解/單引號/key 無引號/尾逗號/漏逗號/括號/字串控制字元/缺結尾引號），但仍不是合法 JSON。\n"
            f"已把『修復後原文』寫到：{OUTPUT_FILE}"
        )

    # 成功：輸出格式化 JSON（要 indent=4 就改 4）
    text_out = json.dumps(obj2, ensure_ascii=False, indent=2) + "\n"
    write_text_with_progress(OUTPUT_FILE, text_out)
    print(f"已修復並輸出：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
