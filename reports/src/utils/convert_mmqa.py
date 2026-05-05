import os
import json
import csv
import io
import re
from difflib import SequenceMatcher
from typing import Any, List, Dict, Tuple
from tqdm import tqdm

# ========= 全域變數（只需要改這裡） =========
dataset = "mmqa3"
input_json_path = "/user_data/TabGNN/data/downloads/Synthesized_three_table_fixed.json"

SIM_THRESHOLD = 0.95
# ========================================


def warn(msg: str):
    print(f"[WARN] {msg}")


def row_to_csv_line(row: List[Any]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="")
    writer.writerow(["" if v is None else v for v in row])
    return buf.getvalue()


_ws_re = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    return _ws_re.sub(" ", s.strip().lower())


def table_repr(header_str: str, instances: List[str]) -> str:
    return normalize_text(header_str + "\n" + "\n".join(instances))


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def main():
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(input_json_path))
    output_dir = os.path.join(base_dir, dataset)
    os.makedirs(output_dir, exist_ok=True)

    table_jsonl_path = os.path.join(output_dir, "table.jsonl")
    query_jsonl_path = os.path.join(output_dir, "query.jsonl")

    src_file_name = f"{dataset}/{os.path.basename(input_json_path)}"

    # header_str -> [(repr, table_id, sheet_name)]
    seen_by_header: Dict[str, List[Tuple[str, int, str]]] = {}
    next_table_id = 0

    with open(table_jsonl_path, "w", encoding="utf-8") as ft, \
         open(query_jsonl_path, "w", encoding="utf-8") as fq:

        for ex in tqdm(data, desc="Converting examples"):
            ex_id = ex.get("id_")
            question = ex.get("Question", "")
            answer = ex.get("answer", "")

            table_names = ex.get("table_names", [])
            tables = ex.get("tables", [])

            if not isinstance(table_names, list):
                warn(f"id_={ex_id}: table_names is not list")
                table_names = []

            if not isinstance(tables, list):
                warn(f"id_={ex_id}: tables is not list")
                tables = []

            if len(table_names) != len(tables):
                warn(
                    f"id_={ex_id}: table_names={len(table_names)}, tables={len(tables)} mismatch"
                )

            ground_truth_list: List[Dict[str, Any]] = []
            n = max(len(table_names), len(tables))

            for i in range(n):
                sheet_name = table_names[i] if i < len(table_names) else f"table_{i}"
                table = tables[i] if i < len(tables) else None

                if table is None:
                    warn(f"id_={ex_id}: missing table '{sheet_name}', skipped")
                    continue

                columns = table.get("table_columns", [])
                rows = table.get("table_content", [])

                if not isinstance(columns, list) or not isinstance(rows, list):
                    warn(f"id_={ex_id}, sheet='{sheet_name}': invalid table format")
                    continue

                header_str = ",".join(map(str, columns))
                instances = [row_to_csv_line(r) for r in rows]
                rep = table_repr(header_str, instances)

                bucket = seen_by_header.get(header_str, [])
                matched = None

                for prev_rep, table_id, prev_sheet in bucket:
                    if abs(len(prev_rep) - len(rep)) / max(len(prev_rep), len(rep)) > 0.2:
                        continue
                    if similarity(rep, prev_rep) >= SIM_THRESHOLD:
                        matched = table_id
                        break

                if matched is not None:
                    ground_truth_list.append({
                        "id": matched,
                        "file_name": src_file_name,
                        "sheet_name": sheet_name,
                    })
                    continue

                # ===== 新 table，給新 id =====
                table_id = next_table_id
                next_table_id += 1

                table_entry = {
                    "id": table_id,
                    "file_name": src_file_name,
                    "sheet_name": sheet_name,
                    "header": [header_str],
                    "instances": instances,
                    "metadata": {
                        "table_source_json": src_file_name,
                        "table_section_title": sheet_name,
                    },
                }
                ft.write(json.dumps(table_entry, ensure_ascii=False) + "\n")

                seen_by_header.setdefault(header_str, []).append(
                    (rep, table_id, sheet_name)
                )

                ground_truth_list.append({
                    "id": table_id,
                    "file_name": src_file_name,
                    "sheet_name": sheet_name,
                })

            # ===== query.jsonl（query id 不動） =====
            query_entry = {
                "id": ex_id,
                "question": question,
                "answer": answer,
                "ground_truth_list": ground_truth_list,
                "metadata": {},
            }
            fq.write(json.dumps(query_entry, ensure_ascii=False) + "\n")

    print("✅ Done")
    print(f"Total unique tables: {next_table_id}")
    print(table_jsonl_path)
    print(query_jsonl_path)


if __name__ == "__main__":
    main()
