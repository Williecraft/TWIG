#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate LaTeX table from edge ablation results with configurable sorting order.
Automatically highlights best (bold) and second-best (underline) for each metric.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


# Dataset names and display names
DATASETS = {
    'e2ewtq': 'E2E-WTQ',
    'feta': 'FeTaQA',
    'mimo_ch': 'MIMO(ch)',
    'mimo_en': 'MIMO(en)',
    'mmqa': 'MMQA',
    'ottqa': 'OTTQA'
}

# Edge abbreviations (bit order: 5 to 0)
EDGE_ABBREVS = ['tt', 'tc', 'tp', 'sp', 'cc', 'sc']

# ============================================================
# CONFIGURATION: Sort Order
# ============================================================
# Sort configurations by average metrics in this order (high to low)
# Valid values: 'r@1', 'r@5', 'r@10'
# Examples:
#   ['r@1', 'r@5', 'r@10']  - Sort by R@1 first, then R@5, then R@10
#   ['r@5', 'r@10', 'r@1']  - Sort by R@5 first, then R@10, then R@1
#   ['r@10', 'r@5', 'r@1']  - Sort by R@10 first, then R@5, then R@1
SORT_ORDER = ['r@10', 'r@5', 'r@1']
# ============================================================



def load_results(results_dir: str) -> Dict[str, List[dict]]:
    """Load results from all datasets."""
    all_results = {}
    
    for dataset in DATASETS.keys():
        json_path = f"{results_dir}/{dataset}/results.json"
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_results[dataset] = data.get('results', [])
        else:
            print(f"Warning: {json_path} not found")
            all_results[dataset] = []
    
    return all_results


def binary_to_edges(code: int) -> List[str]:
    """Convert binary code to edge list."""
    binary = format(code, '06b')
    edges = []
    for i, bit in enumerate(binary):
        if bit == '1':
            edges.append(EDGE_ABBREVS[i])
    return edges


def get_config_by_code(results: List[dict], code: int) -> dict:
    """Get configuration result by code."""
    for result in results:
        if result.get('code') == code:
            return result
    return None


def find_best_and_second(values: List) -> Tuple[int, int]:
    """Find indices of best and second-best values, excluding None."""
    if not values:
        return -1, -1
    
    # Filter out None values and keep track of original indices
    indexed_values = [(i, v) for i, v in enumerate(values) if v is not None]
    if not indexed_values:
        return -1, -1
    
    sorted_values = sorted(indexed_values, key=lambda x: x[1], reverse=True)
    
    best_idx = sorted_values[0][0] if len(sorted_values) > 0 else -1
    second_idx = sorted_values[1][0] if len(sorted_values) > 1 else -1
    
    return best_idx, second_idx


def format_value(value, is_best: bool, is_second: bool) -> str:
    """Format value with bold/underline if needed. Returns N/A for None."""
    if value is None:
        return "-"
    
    formatted = f"{value * 100:.2f}"
    
    if is_best:
        return f"\\textbf{{{formatted}}}"
    elif is_second:
        return f"\\underline{{{formatted}}}"
    else:
        return formatted


def generate_latex_table(results_dir: str, output_path: str):
    """
    Generate LaTeX table from results using the global SORT_ORDER configuration.
    
    Args:
        results_dir: Directory containing results
        output_path: Path to save LaTeX output
    """
    
    # Validate SORT_ORDER
    valid_metrics = {'r@1', 'r@5', 'r@10'}
    if not all(m in valid_metrics for m in SORT_ORDER):
        raise ValueError(f"Invalid metric in SORT_ORDER. Must be one of {valid_metrics}")

    
    # Load all results
    all_results = load_results(results_dir)
    
    # Get all unique codes from all datasets
    all_codes = set()
    for dataset_results in all_results.values():
        for result in dataset_results:
            all_codes.add(result.get('code'))
    
    # Build configuration data
    config_data = []
    for code in all_codes:
        config_info = {
            'code': code,
            'edges': binary_to_edges(code),
            'datasets': {}
        }
        
        # Collect results from each dataset
        for dataset in DATASETS.keys():
            result = get_config_by_code(all_results[dataset], code)
            if result:
                # MMQA uses R@2 instead of R@1 (since R@1 is always 0)
                if dataset == 'mmqa':
                    config_info['datasets'][dataset] = {
                        'r@1': result.get('recall@2', 0.0),  # Use R@2 for MMQA
                        'r@5': result.get('recall@5', 0.0),
                        'r@10': result.get('recall@10', 0.0)
                    }
                else:
                    config_info['datasets'][dataset] = {
                        'r@1': result.get('recall@1', 0.0),
                        'r@5': result.get('recall@5', 0.0),
                        'r@10': result.get('recall@10', 0.0)
                    }
            else:
                # Use None for missing data instead of 0
                config_info['datasets'][dataset] = {
                    'r@1': None,
                    'r@5': None,
                    'r@10': None
                }
        
        # Calculate averages, excluding None values
        r1_values = [config_info['datasets'][ds]['r@1'] for ds in DATASETS.keys()]
        r5_values = [config_info['datasets'][ds]['r@5'] for ds in DATASETS.keys()]
        r10_values = [config_info['datasets'][ds]['r@10'] for ds in DATASETS.keys()]
        
        # Filter out None and calculate averages
        r1_valid = [v for v in r1_values if v is not None]
        r5_valid = [v for v in r5_values if v is not None]
        r10_valid = [v for v in r10_values if v is not None]
        
        config_info['avg'] = {
            'r@1': sum(r1_valid) / len(r1_valid) if r1_valid else None,
            'r@5': sum(r5_valid) / len(r5_valid) if r5_valid else None,
            'r@10': sum(r10_valid) / len(r10_valid) if r10_valid else None
        }
        
        config_data.append(config_info)
    
    # Sort by metrics in the specified order (descending)
    # Treat None as -infinity for sorting purposes
    def sort_key(config):
        return tuple(
            config['avg'][metric] if config['avg'][metric] is not None else -float('inf')
            for metric in SORT_ORDER
        )
    
    config_data.sort(key=sort_key, reverse=True)
    
    # Find best and second-best for each column
    metrics_by_column = {}
    
    # For each dataset and metric
    for dataset in DATASETS.keys():
        for metric in ['r@1', 'r@5', 'r@10']:
            col_key = f"{dataset}_{metric}"
            values = [cfg['datasets'][dataset][metric] for cfg in config_data]
            best_idx, second_idx = find_best_and_second(values)
            metrics_by_column[col_key] = (best_idx, second_idx)
    
    # For avg metrics
    for metric in ['r@1', 'r@5', 'r@10']:
        col_key = f"avg_{metric}"
        values = [cfg['avg'][metric] for cfg in config_data]
        best_idx, second_idx = find_best_and_second(values)
        metrics_by_column[col_key] = (best_idx, second_idx)
    
    # Generate sort order description for caption
    sort_desc = " > ".join([f"Avg {m.upper()}" for m in SORT_ORDER])

    
    # Generate LaTeX
    latex_lines = []
    latex_lines.append("\\begin{table*}[t]")
    latex_lines.append("  \\footnotesize")
    latex_lines.append("  \\centering")
    latex_lines.append(f"  \\caption{{Recall metrics across different datasets, sorted by {sort_desc}. Best results are \\textbf{{bold}}, second best are \\underline{{underlined}}.}}")
    latex_lines.append("  \\label{tab:edge_ablation_results}")
    latex_lines.append("")
    latex_lines.append("  \\setlength{\\tabcolsep}{2.5pt}")
    
    # Table header
    num_cols = 6 + 6 * 3 + 3  # 6 method + 6 datasets * 3 metrics + 3 avg
    col_spec = "c" * 6 + ("ccc" * 6) + "|ccc"
    latex_lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("    \\toprule")
    latex_lines.append("")
    
    # First header row: dataset names
    header1 = "    \\multicolumn{6}{c}{Method}"
    for dataset_name in DATASETS.values():
        header1 += f" & \\multicolumn{{3}}{{c}}{{\\textbf{{{dataset_name}}}}}"
    header1 += " & \\multicolumn{3}{c}{\\textbf{Avg}} \\\\"
    latex_lines.append(header1)
    latex_lines.append("")
    
    # cmidrules
    start_col = 7
    cmidrules = []
    for i in range(7):  # 6 datasets + 1 avg
        end_col = start_col + 2
        cmidrules.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col += 3
    latex_lines.append("    " + " ".join(cmidrules))
    latex_lines.append("")
    
    # Second header row: edge types and metrics
    header2 = "    " + " & ".join(EDGE_ABBREVS)
    for i, dataset in enumerate(list(DATASETS.keys()) + ['avg']):
        if dataset == 'mmqa':
            header2 += " & R@2 & R@5 & R@10"  # MMQA uses R@2
        else:
            header2 += " & R@1 & R@5 & R@10"
    header2 += " \\\\"
    latex_lines.append(header2)
    latex_lines.append("    \\midrule")
    latex_lines.append("")
    
    # Data rows
    for row_idx, config in enumerate(config_data):
        # Edge checkmarks
        edge_cols = []
        for edge in EDGE_ABBREVS:
            if edge in config['edges']:
                edge_cols.append("v")
            else:
                edge_cols.append("")
        
        row = "    " + " & ".join(edge_cols)
        
        # Dataset metrics
        for dataset in DATASETS.keys():
            for metric in ['r@1', 'r@5', 'r@10']:
                col_key = f"{dataset}_{metric}"
                best_idx, second_idx = metrics_by_column[col_key]
                value = config['datasets'][dataset][metric]
                is_best = (row_idx == best_idx)
                is_second = (row_idx == second_idx)
                row += " & " + format_value(value, is_best, is_second)
        
        # Average metrics
        for metric in ['r@1', 'r@5', 'r@10']:
            col_key = f"avg_{metric}"
            best_idx, second_idx = metrics_by_column[col_key]
            value = config['avg'][metric]
            is_best = (row_idx == best_idx)
            is_second = (row_idx == second_idx)
            row += " & " + format_value(value, is_best, is_second)
        
        row += " \\\\"
        latex_lines.append(row)
        latex_lines.append("")
    
    latex_lines.append("    \\bottomrule")
    latex_lines.append("  \\end{tabular}")
    latex_lines.append("\\end{table*}")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"✅ LaTeX table generated: {output_path}")
    print(f"   Configurations: {len(config_data)}")
    print(f"   Datasets: {len(DATASETS)}")
    print(f"   Sort order: {sort_desc}")


if __name__ == '__main__':
    results_dir = '/user_data/TabGNN/results/edge_ablation_extended'
    output_path = '/user_data/TabGNN/results/edge_ablation_extended/results_table.tex'
    
    print(f"Current configuration:")
    print(f"  Sort order: {' > '.join([m.upper() for m in SORT_ORDER])}")
    print(f"  Output: {output_path}")
    print()
    
    # Generate table with current SORT_ORDER configuration
    generate_latex_table(results_dir, output_path)
    
    # To change sort order, modify the SORT_ORDER global variable at the top of this file
    # Examples:
    #   SORT_ORDER = ['r@5', 'r@10', 'r@1']  # Sort by R@5 first
    #   SORT_ORDER = ['r@10', 'r@5', 'r@1']  # Sort by R@10 first

