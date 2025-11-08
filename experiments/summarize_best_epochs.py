#!/usr/bin/env python3
"""
Summarize results from 'train.py' runs.

This script scans a base directory (e.g., 'results/runs') for model-group
subdirectories. For each model-group, it looks for a 'seed-42' run.

It then does the following:
1. Reads 'run_results.json' to find the 'best_val_loss', 'param_count',
   and 'latency_ms'.
2. Reads 'epoch_log.csv' to find the epoch row that most closely matches
   the 'best_val_loss'.
3. Parses the 'eval_metrics_json' from that epoch row to extract COCO
   and phenotype metrics.
4. Prints a formatted summary table of all found results.
"""

import os
import json
import csv
import argparse
import math
import numpy as np
from typing import List, Dict, Any, Optional

def _format_value(val: Any, header: str) -> str:
    """Helper to format values for the table."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    try:
        if header == "Model":
            return str(val)
        if header == "params":
            return f"{int(val):,}"
        if header == "latency":
            return f"{float(val):.2f}"
        if header == "MAPE":
            # Assumes MAPE is a float (e.g., 0.15 for 15%)
            return f"{float(val) * 100:.2f}%"
        if header in ["AP50:95", "AR50:95", "R2"]:
            return f"{float(val):.4f}"
        if header == "RMSE":
            return f"{float(val):.2f}"
    except (ValueError, TypeError):
        return "ERR"
    return str(val)

def _extract_epoch_metrics(eval_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts specific metrics from the eval_metrics_json blob."""
    metrics = {}
    
    # Extract COCO metrics
    bbox_stats = eval_data.get("bbox")
    if bbox_stats and isinstance(bbox_stats, list) and len(bbox_stats) >= 9:
        metrics["AP50:95"] = bbox_stats[0]  # AP @[ IoU=0.50:0.95 | all ]
        metrics["AR50:95"] = bbox_stats[8]  # AR @[ IoU=0.50:0.95 | all | maxDets=100 ]

    # Extract Phenotype metrics
    pheno_stats = eval_data.get("phenotype")
    if pheno_stats and isinstance(pheno_stats, dict):
        # NOTE: Grabbing metrics from the *first* phenotype found
        first_pheno_name = next(iter(pheno_stats), None)
        if first_pheno_name:
            pheno_metrics = pheno_stats[first_pheno_name]
            if isinstance(pheno_metrics, dict):
                metrics["R2"] = pheno_metrics.get("r2")
                metrics["MAPE"] = pheno_metrics.get("mape")
                metrics["RMSE"] = pheno_metrics.get("rmse")
                
    return metrics

def process_run(seed_dir: str, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Processes a single 'seed-42' run directory.
    
    Args:
        seed_dir: Path to the 'seed-42' directory.
        model_name: Name of the parent model group.

    Returns:
        A dictionary of extracted metrics or None if processing fails.
    """
    results_json_path = os.path.join(seed_dir, "run_results.json")
    epoch_log_path = os.path.join(seed_dir, "epoch_log.csv")

    if not os.path.exists(results_json_path):
        print(f"  - Skipping: 'run_results.json' not found in {seed_dir}")
        return None
    if not os.path.exists(epoch_log_path):
        print(f"  - Skipping: 'epoch_log.csv' not found in {seed_dir}")
        return None

    metrics = {"Model": model_name}
    best_val_loss = None

    try:
        # 1. Read run_results.json
        with open(results_json_path, 'r') as f:
            results_data = json.load(f)
        
        best_val_loss = results_data.get("best_val_loss")
        metrics["params"] = results_data.get("param_count")
        metrics["latency"] = results_data.get("latency_ms")

        if best_val_loss is None:
            print(f"  - Skipping: 'best_val_loss' not in {results_json_path}")
            return None

        # 2. Find best epoch in epoch_log.csv
        best_epoch_row = None
        min_diff = math.inf

        with open(epoch_log_path, 'r', newline='') as f:
            # Handle potential empty CSV
            header_line = f.readline()
            if not header_line:
                print(f"  - Skipping: 'epoch_log.csv' is empty in {seed_dir}")
                return None
            
            headers = [h.strip() for h in header_line.split(',')]
            reader = csv.DictReader(f, fieldnames=headers)
            
            for row in reader:
                val_loss_str = row.get('val_loss')
                if val_loss_str is None or val_loss_str.lower() == 'nan':
                    continue
                
                try:
                    current_val_loss = float(val_loss_str)
                    diff = abs(current_val_loss - best_val_loss)
                    
                    # Find the row with the closest val_loss
                    if diff < min_diff:
                        min_diff = diff
                        best_epoch_row = row
                        
                        # Stop if we find a near-perfect match
                        if math.isclose(diff, 0, abs_tol=1e-6):
                            break
                            
                except (ValueError, TypeError):
                    continue # Skip rows with bad data

        if best_epoch_row is None:
            print(f"  - Skipping: No matching epoch found in {epoch_log_path}")
            return None

        # 3. Extract metrics from the best epoch's JSON blob
        eval_metrics_json = best_epoch_row.get("eval_metrics_json")
        if not eval_metrics_json:
            print(f"  - Warning: 'eval_metrics_json' is empty for best epoch.")
        else:
            try:
                eval_data = json.loads(eval_metrics_json)
                epoch_metrics = _extract_epoch_metrics(eval_data)
                metrics.update(epoch_metrics)
            except json.JSONDecodeError:
                print(f"  - Warning: Failed to parse 'eval_metrics_json'.")

        return metrics

    except Exception as e:
        print(f"  - ERROR processing {seed_dir}: {e}")
        return None

def print_results_table(all_results: List[Dict[str, Any]]):
    """Prints a formatted table of all collected results."""
    if not all_results:
        print("\nNo valid 'seed-42' results found to summarize.")
        return

    print("\n--- Model Summary (from seed-42) ---")

    headers = ["Model", "AP50:95", "AR50:95", "R2", "MAPE", "RMSE", "params", "latency"]
    
    # Determine column widths
    col_widths = {h: len(h) for h in headers}
    formatted_rows = []

    for row in all_results:
        formatted_row = {}
        for h in headers:
            val_str = _format_value(row.get(h), h)
            formatted_row[h] = val_str
            col_widths[h] = max(col_widths[h], len(val_str))
        formatted_rows.append(formatted_row)

    # Print Header
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))

    # Print Rows
    for row in formatted_rows:
        row_line = " | ".join(row[h].ljust(col_widths[h]) for h in headers)
        print(row_line)

def main():
    parser = argparse.ArgumentParser(
        description="Summarize train.py results for 'seed-42' runs."
    )
    parser.add_argument(
        "-b", "--base-dir",
        nargs="?",
        default="results/runs",
        help="The base directory containing model-group subfolders."
    )
    args = parser.parse_args()

    MODEL_NAME_MAP = {
        'lettuce_model_multimodal_dataset_copypaste': 'MobileViTv2 + DF + CA',
        'lettuce_model_dataset_copypaste': 'MobileViTv2 + CA',
        'lettuce_model_multimodal_mobnetv3_dataset_copypaste': 'Baseline + DF + CA',
        'lettuce_model_mobnetv3_dataset_copypaste': 'Baseline + CA',
        'lettuce_model_multimodal_dataset_no_copypaste': 'MobileViTv2 + DF',
        'lettuce_model_dataset_no_copypaste': 'MobileViTv2',
        'lettuce_model_multimodal_mobnetv3_dataset_no_copypaste': 'Baseline + DF',
        'lettuce_model_mobnetv3_dataset_no_copypaste': 'Baseline',
    }

    if not os.path.isdir(args.base_dir):
        print(f"Error: Base directory not found: {args.base_dir}")
        return

    print(f"Scanning for 'seed-42' runs in: {args.base_dir} using defined map.")
    all_results = []

    try:
        # <<< MODIFIKASI: Iterasi melalui map, bukan memindai direktori >>>
        for folder_name, pretty_name in MODEL_NAME_MAP.items():
            model_dir_path = os.path.join(args.base_dir, folder_name)
            seed_dir = os.path.join(model_dir_path, "seed-42")
            
            if os.path.isdir(seed_dir):
                print(f"--- Processing {pretty_name} (from {folder_name}) ---")
                # Teruskan 'pretty_name' ke process_run
                result = process_run(seed_dir, pretty_name)
                if result:
                    all_results.append(result)
            else:
                # Memberi tahu Anda jika ada entri di map tapi foldernya tidak ditemukan
                if not os.path.isdir(model_dir_path):
                    print(f"--- Skipping {pretty_name} (Folder not found: {folder_name}) ---")
                else:
                    print(f"--- Skipping {pretty_name} (no 'seed-42' dir in {folder_name}) ---")

    except FileNotFoundError:
        print(f"Error: Path not found {args.base_dir}")
    except NotADirectoryError:
        print(f"Error: Path is not a directory {args.base_dir}")
    
    print_results_table(all_results)

if __name__ == "__main__":
    main()