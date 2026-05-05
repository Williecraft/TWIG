import json

def json2md(jtable: dict, top=10) -> str:
    import csv
    import pandas as pd

    def parse_csv_row(s: str) -> list[str]:
        return next(csv.reader([s]))

    # ---- header ----
    header = jtable.get("header", [])
    if isinstance(header, str):
        header = parse_csv_row(header)
    elif isinstance(header, list):
        if len(header) == 1 and isinstance(header[0], str) and ("," in header[0]):
            header = parse_csv_row(header[0])
        else:
            header = [("" if h is None else str(h)) for h in header]
    else:
        header = [str(header)]

    # ---- instances ----
    instances = jtable.get("instances", [])
    rows = instances if top is None else instances[:int(top)]

    parsed_rows = []
    for r in rows:
        if isinstance(r, str):
            parsed_rows.append(parse_csv_row(r))
        elif isinstance(r, list):
            parsed_rows.append([("" if x is None else str(x)) for x in r])
        else:
            parsed_rows.append([str(r)])

    # ---- 對齊欄數（避免某些列欄位數不一致）----
    ncol = len(header)
    fixed_rows = []
    for row in parsed_rows:
        if len(row) < ncol:
            row = row + [""] * (ncol - len(row))
        elif len(row) > ncol:
            row = row[:ncol]
        fixed_rows.append(row)

    df = pd.DataFrame(fixed_rows, columns=header)
    return df.to_markdown(index=False)

SOURCE = "test/feta"
INPUT_FILE = "table.jsonl"
TABLE_PATH = f"/user_data/TabGNN/data/table/{SOURCE}/{INPUT_FILE}"
for i in range(10):
    with open(TABLE_PATH, "r", encoding="utf-8") as f:
        table_lines = [json.loads(line) for line in f.readlines()]
    table:dict = table_lines[i]
    print(json2md(table, top=10))
    print("\n===========================================\n")