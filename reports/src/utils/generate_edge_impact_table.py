#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate edge type impact analysis table.
Rules:
- Missing cell shown as '-' (NOT 0)
- If either N or Y missing for a given (dataset, metric), treat BOTH as missing:
  => output '-' for both rows, and exclude from Avg.
- Rows order: N first, then Y.
- For each column, bold the larger value between N and Y (only when both exist).
"""

import json
import os
from typing import Dict, List, Optional

DATASETS = {
    "feta": "FeTaQA",
    "ottqa": "OTTQA",
    "mimo_en": "MIMO(en)",
    "mimo_ch": "MIMO(ch)",
    "e2ewtq": "E2E-WTQ",
    "mmqa": "MMQA",
}

EDGE_TYPES = ["tt", "tc", "tp", "sp", "cc", "sc"]

# bitmask decode (default): LSB=tt, bit1=tc, ..., bit5=sc
# If your encoding is reversed (MSB=tt), switch to:
# EDGE_BIT_POS = {e: 5 - i for i, e in enumerate(EDGE_TYPES)}
EDGE_BIT_POS = {e: i for i, e in enumerate(EDGE_TYPES)}  # tt:0 ... sc:5


def load_results(results_dir: str) -> Dict[str, List[dict]]:
    all_results: Dict[str, List[dict]] = {}
    for dataset in DATASETS.keys():
        json_path = f"{results_dir}/{dataset}/results.json"
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_results[dataset] = data.get("results", [])
        else:
            print(f"Warning: {json_path} not found")
            all_results[dataset] = []
    return all_results


def _to_int_code(code) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    if isinstance(code, str):
        s = code.strip()
        if s.startswith("0b"):
            return int(s, 2)
        if all(c in "01" for c in s) and len(s) <= 64:
            return int(s, 2)
        return int(s)
    return int(code)


def has_edge(code, edge_type: str) -> bool:
    c = _to_int_code(code)
    pos = EDGE_BIT_POS[edge_type]
    return ((c >> pos) & 1) == 1


def calculate_edge_impact(all_results: Dict[str, List[dict]]) -> Dict:
    impact_data = {}
    for edge_type in EDGE_TYPES:
        impact_data[edge_type] = {"with": {}, "without": {}}
        for dataset in DATASETS.keys():
            impact_data[edge_type]["with"][dataset] = {"r@1": [], "r@5": [], "r@10": []}
            impact_data[edge_type]["without"][dataset] = {"r@1": [], "r@5": [], "r@10": []}

        for dataset in DATASETS.keys():
            for result in all_results.get(dataset, []):
                code = result.get("code", 0)
                category = "with" if has_edge(code, edge_type) else "without"

                if dataset == "mmqa":
                    r1 = float(result.get("recall@2", 0.0))
                else:
                    r1 = float(result.get("recall@1", 0.0))
                r5 = float(result.get("recall@5", 0.0))
                r10 = float(result.get("recall@10", 0.0))

                impact_data[edge_type][category][dataset]["r@1"].append(r1)
                impact_data[edge_type][category][dataset]["r@5"].append(r5)
                impact_data[edge_type][category][dataset]["r@10"].append(r10)
    return impact_data


def _mean(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def _avg_ignore_missing(values_percent: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in values_percent if v is not None]
    return (sum(xs) / len(xs)) if xs else None


def generate_latex_table(results_dir: str, output_path: str):
    all_results = load_results(results_dir)
    impact_data = calculate_edge_impact(all_results)

    latex_lines: List[str] = []
    latex_lines.append("\\begin{table*}[t]")
    latex_lines.append("  \\scriptsize")
    latex_lines.append("  \\centering")
    latex_lines.append("  \\caption{Edge type impact analysis. Y = with edge, N = without edge.}")
    latex_lines.append("  \\label{tab:edge_impact}")
    latex_lines.append("")
    latex_lines.append("  \\setlength{\\tabcolsep}{2.5pt}")
    latex_lines.append("  \\renewcommand{\\arraystretch}{1.15}")
    latex_lines.append("  \\begin{tabular}{@{}l c|*{18}{r}|*{3}{r}@{}}")
    latex_lines.append("    \\toprule")
    latex_lines.append("")

    # Header row 1
    header1 = ["    Edge", ""]
    for name in DATASETS.values():
        header1.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{name}}}}}")
    header1.append("\\multicolumn{3}{c}{\\textbf{Avg}}")
    latex_lines.append(" & ".join(header1) + " \\\\")
    latex_lines.append("")

    # cmidrules
    cmidrule_parts = []
    start_col = 3
    for _ in range(len(DATASETS) + 1):
        end_col = start_col + 2
        cmidrule_parts.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col += 3
    latex_lines.append("    " + " ".join(cmidrule_parts))
    latex_lines.append("")

    # Header row 2
    header2 = ["    ", " "]
    for key in list(DATASETS.keys()) + ["avg"]:
        if key == "mmqa":
            header2.append("R@2 & R@5 & R@10")
        else:
            header2.append("R@1 & R@5 & R@10")
    latex_lines.append(" & ".join(header2) + " \\\\")
    latex_lines.append("    \\midrule")
    latex_lines.append("")

    for edge_type in EDGE_TYPES:
        # store percent-scale per dataset (Optional[float])
        per_Y = {"r@1": [], "r@5": [], "r@10": []}  # with edge
        per_N = {"r@1": [], "r@5": [], "r@10": []}  # without edge

        cells_Y: List[Optional[float]] = []
        cells_N: List[Optional[float]] = []

        # ---- dataset metrics ----
        for dataset in DATASETS.keys():
            for metric in ["r@1", "r@5", "r@10"]:
                mY = _mean(impact_data[edge_type]["with"][dataset][metric])       # [0,1] or None
                mN = _mean(impact_data[edge_type]["without"][dataset][metric])    # [0,1] or None

                vY = None if mY is None else mY * 100.0
                vN = None if mN is None else mN * 100.0

                # ✅ 핵심：若 N/Y 有一個沒有 → 兩個都當沒有
                if (vY is None) != (vN is None):
                    vY, vN = None, None

                per_Y[metric].append(vY)
                per_N[metric].append(vN)

                cells_Y.append(vY)
                cells_N.append(vN)

        # ---- Avg (ignore missing) ----
        for metric in ["r@1", "r@5", "r@10"]:
            avgY = _avg_ignore_missing(per_Y[metric])
            avgN = _avg_ignore_missing(per_N[metric])

            # 這裡其實不需要再做成對缺失，因為 per_Y/per_N 已經同步 None
            cells_Y.append(avgY)
            cells_N.append(avgN)

        # ---- formatting + bold larger (only when both exist) ----
        rowY: List[str] = []
        rowN: List[str] = []

        for vN, vY in zip(cells_N, cells_Y):
            if vN is None and vY is None:
                rowN.append("-")
                rowY.append("-")
            else:
                # 此時保證兩者都存在
                if vN > vY:
                    rowN.append(f"\\textbf{{{vN:.2f}}}")
                    rowY.append(f"{vY:.2f}")
                elif vY > vN:
                    rowN.append(f"{vN:.2f}")
                    rowY.append(f"\\textbf{{{vY:.2f}}}")
                else:
                    rowN.append(f"{vN:.2f}")
                    rowY.append(f"{vY:.2f}")

        # N first, then Y
        latex_lines.append(
            f"    \\multirow{{2}}{{*}}{{\\texttt{{{edge_type}}}}} & \\textbf{{N}} & "
            + " & ".join(rowN)
            + " \\\\"
        )
        latex_lines.append(
            f"    & \\textbf{{Y}} & "
            + " & ".join(rowY)
            + " \\\\"
        )

        # edge block spacing
        if edge_type != EDGE_TYPES[-1]:
            latex_lines.append("    \\midrule")

    latex_lines.append("    \\bottomrule")
    latex_lines.append("  \\end{tabular}")
    latex_lines.append("\\end{table*}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print(f"✅ Edge impact table generated: {output_path}")
    print(f"   Edge types analyzed: {len(EDGE_TYPES)}")
    print(f"   Datasets: {len(DATASETS)}")


if __name__ == "__main__":
    results_dir = "/user_data/TabGNN/results/edge_ablation_extended"
    output_path = "/user_data/TabGNN/results/edge_ablation_extended/edge_impact_table.tex"
    generate_latex_table(results_dir, output_path)
