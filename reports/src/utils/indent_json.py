import json
from pathlib import Path
from typing import Any


# ====== 只要改這裡 ======
INPUT_FILE = Path("/user_data/TabGNN/data/downloads/Synthesized_two_table_fixed.json")
# 覆蓋原檔：輸出路徑 = 輸入路徑
OUTPUT_FILE = INPUT_FILE
# 純元素 list 允許單行的最大字元數（避免超長）
MAX_INLINE_CHARS = 10**9
# =======================


PRIMITIVES = (str, int, float, bool, type(None))


def is_simple_list(lst: list) -> bool:
    """list 內全部是純元素（不含 dict/list）"""
    return all(isinstance(x, PRIMITIVES) for x in lst)


def json_primitive(x: Any) -> str:
    """把 primitive 轉成 JSON 字面值（含字串跳脫）"""
    return json.dumps(x, ensure_ascii=False)


def pretty_dump(obj: Any, indent: int = 4, level: int = 0) -> str:
    sp = " " * (indent * level)
    sp_in = " " * (indent * (level + 1))

    # dict
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = []
        for k, v in obj.items():
            key = json.dumps(k, ensure_ascii=False)  # key 一定是字串
            val = pretty_dump(v, indent=indent, level=level + 1)
            items.append(f"{sp_in}{key}: {val}")
        return "{\n" + ",\n".join(items) + f"\n{sp}" + "}"

    # list
    if isinstance(obj, list):
        if not obj:
            return "[]"

        # 純元素 list：嘗試單行
        if is_simple_list(obj):
            inline = "[" + ", ".join(json_primitive(x) for x in obj) + "]"
            if len(inline) <= MAX_INLINE_CHARS:
                return inline

        # 其他情況：多行
        items = [f"{sp_in}{pretty_dump(x, indent=indent, level=level + 1)}" for x in obj]
        return "[\n" + ",\n".join(items) + f"\n{sp}]"

    # primitives
    if isinstance(obj, PRIMITIVES):
        return json_primitive(obj)

    # 其他型別（理論上合法 JSON 不會有，但保險）
    return json.dumps(obj, ensure_ascii=False)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到檔案：{INPUT_FILE}")
    if INPUT_FILE.is_dir():
        raise IsADirectoryError(f"輸入路徑是資料夾不是檔案：{INPUT_FILE}")

    raw = INPUT_FILE.read_text(encoding="utf-8", errors="strict")
    data = json.loads(raw)  # 你說已確認合法，這裡若炸就表示其實不合法

    formatted = pretty_dump(data, indent=4, level=0) + "\n"
    OUTPUT_FILE.write_text(formatted, encoding="utf-8")

    print(f"已覆蓋輸出：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
