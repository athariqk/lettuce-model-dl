#!/usr/bin/env python3
"""
Summarize results from 'train.py' ablation-type experiment runs.

This script scans a base directory (e.g., 'results/runs') for model-group
subdirectories. For each model-group, it looks for a 'seed-42' run.

Enhancement:
- MODEL_NAME_MAPS supports grouping of related folders. Each group may
    have exactly one baseline (flagged with 'baseline': True). Deltas are
    computed per-group with respect to that group's baseline.
- Backwards-compatible: If a flat MODEL_NAME_MAP is provided (legacy),
    it will be wrapped into a single group named "default".
- You can optionally override a baseline via CLI using:
        --baseline-folder group:folder_name
    or simply:
        --baseline-folder folder_name
    (the latter will search for the folder across groups).
"""

import os
import json
import csv
import argparse
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


plt.rcParams.update({
    "font.family": "serif",  # Use a serif font
    "font.serif": ["Times New Roman", "Computer Modern Roman"], # Specify font stack
    "font.size": 14,  # Base font size for ticks
    "axes.labelsize": 12,  # Font size for x and y labels
    "legend.fontsize": 12,
    "mathtext.fontset": "stix", # Use STIX for math text (looks like Times)
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


def _format_value(val: Any, header: str) -> str:
    """Helper to format values for the table."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    try:
        # Delta headers start with 'Δ' — format similar to their base metric
        if header.startswith("Δ"):
            base = header[1:]
            # AP/AR/R2 deltas are shown in percentage points
            if base in ["AP50:95", "AR50:95", "R2"]:
                return f"{float(val) * 100:.1f}"
            if base == "MAPE":
                return f"{float(val) * 100:.2f}"
            if base == "RMSE":
                return f"{float(val):.2f}"
            if base == "params":
                return f"{int(val):,}"
            if base == "latency":
                return f"{float(val):.2f}"
        # Non-delta headers
        if header == "Model":
            return str(val)
        if header == "Group":
            return str(val)
        if header == "params":
            return f"{int(val):,}"
        if header == "latency":
            return f"{float(val):.2f}"
        if header == "MAPE":
            # Assumes MAPE is a float (e.g., 0.15 for 15)
            return f"{float(val) * 100:.2f}"
        if header in ["AP50:95", "AR50:95", "R2"]:
            return f"{float(val) * 100:.1f}"
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
        with open(results_json_path, "r") as f:
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

        with open(epoch_log_path, "r", newline="") as f:
            # Handle potential empty CSV
            header_line = f.readline()
            if not header_line:
                print(f"  - Skipping: 'epoch_log.csv' is empty in {seed_dir}")
                return None

            headers = [h.strip() for h in header_line.split(",")]
            reader = csv.DictReader(f, fieldnames=headers)

            for row in reader:
                val_loss_str = row.get("val_loss")
                if val_loss_str is None or val_loss_str.lower() == "nan":
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
                    continue  # Skip rows with bad data

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


def _compute_deltas_per_group(
    all_results: List[Dict[str, Any]],
    baseline_override: Optional[str],
    model_maps: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[Dict[str, Any]]]]:
    """
    Compute absolute differences to the baseline model per group.
    - all_results: list of result dicts. Each result must include '_folder' and 'Group'.
    - baseline_override: optional string to override baseline, format:
                "group:folder" or just "folder" (will search groups).
    - model_maps: mapping group_name -> (folder_name -> info)

    Returns:
        (augmented_results, baseline_rows_by_group)
    """
    # Prepare baseline overrides per group
    overrides: Dict[str, Optional[str]] = {}
    if baseline_override:
        if ":" in baseline_override:
            g, f = baseline_override.split(":", 1)
            overrides[g] = f
        else:
            # search for folder across groups
            found_group = None
            for g, gm in model_maps.items():
                if baseline_override in gm:
                    found_group = g
                    break
            if found_group:
                overrides[found_group] = baseline_override
            else:
                # unknown override, ignore
                print(
                    f"Warning: baseline override '{baseline_override}' not found in any group; ignored."
                )

    # Organize results by group
    results_by_group: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_results:
        g = r.get("Group", "default")
        results_by_group.setdefault(g, []).append(r)

    metric_keys = ["AP50:95", "AR50:95", "R2", "RMSE", "MAPE", "params", "latency"]

    baseline_rows_by_group: Dict[str, Optional[Dict[str, Any]]] = {}
    augmented: List[Dict[str, Any]] = []

    for group_name, rows in results_by_group.items():
        # Determine baseline folder name for this group
        group_map = model_maps.get(group_name, {})
        baseline_folder = None

        # 1. check override
        if group_name in overrides:
            baseline_folder = overrides[group_name]
        else:
            # 2. scan group_map for baseline flag
            for folder_name, info in group_map.items():
                if isinstance(info, dict) and info.get("baseline"):
                    baseline_folder = folder_name
                    break

        # find baseline result row (match by '_folder')
        baseline_row = None
        if baseline_folder:
            for rr in rows:
                if rr.get("_folder") == baseline_folder:
                    baseline_row = rr
                    break

        baseline_rows_by_group[group_name] = baseline_row

        # baseline numeric values
        baseline_vals = {
            k: (baseline_row.get(k) if baseline_row else None) for k in metric_keys
        }

        for r in rows:
            newr = dict(r)  # shallow copy
            for k in metric_keys:
                a = r.get(k)
                b = baseline_vals.get(k)
                delta = None
                try:
                    if a is not None and b is not None:
                        delta = abs(float(a) - float(b))
                except (ValueError, TypeError):
                    delta = None
                newr["Δ" + k] = delta
            augmented.append(newr)

    return augmented, baseline_rows_by_group


def print_results_table(
    all_results: List[Dict[str, Any]],
    baseline_by_group: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
):
    """Prints a formatted table of all collected results, including deltas shown next to their metrics."""
    if not all_results:
        print("\nNo valid 'seed-42' results found to summarize.")
        return

    print("\n--- Model Summary (from seed-42) ---")

    # Base headers (include Group)
    metric_headers = [
        "AP50:95",
        "AR50:95",
        "R2",
        "RMSE",
        "MAPE",
        "params",
        "latency",
    ]
    headers = ["Group", "Model"] + metric_headers

    # Determine column widths
    col_widths = {h: len(h) for h in headers}
    formatted_rows = []

    for row in all_results:
        formatted_row = {}
        # Group and Model first
        for h in ["Group", "Model"]:
            val = row.get(h)
            val_str = _format_value(val, h)
            formatted_row[h] = val_str
            col_widths[h] = max(col_widths[h], len(val_str))

        # Metrics: show base value and append delta in parens if present
        for h in metric_headers:
            base_val = row.get(h)
            base_str = _format_value(base_val, h)

            delta_key = "Δ" + h
            delta_val = row.get(delta_key)
            cell_str = base_str

            # Only show delta if it's a valid numeric (not None / NaN)
            if delta_val is not None:
                try:
                    if not (isinstance(delta_val, float) and np.isnan(delta_val)):
                        # formatted delta using the delta header rules
                        delta_formatted = _format_value(delta_val, delta_key)
                        # if delta_formatted is "N/A" or "ERR", skip showing it
                        if delta_formatted not in ("N/A", "ERR"):
                            # prepend '+' or '-' sign based on comparison with baseline
                            baseline_val = baseline_by_group[row.get("Group")].get(h) if baseline_by_group and row.get("Group") in baseline_by_group else None  # type: ignore
                            if baseline_val is not None and float(base_val) < float(baseline_val):  # type: ignore
                                cell_str = (
                                    f"{base_str} (-{delta_formatted})"  # negative delta
                                )
                            else:
                                cell_str = (
                                    f"{base_str} (+{delta_formatted})"  # positive delta
                                )
                except Exception:
                    pass

            formatted_row[h] = cell_str
            col_widths[h] = max(col_widths[h], len(cell_str))

        formatted_rows.append(formatted_row)

    # Print Header
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))

    # Print Rows
    for row in formatted_rows:
        row_line = " | ".join(row[h].ljust(col_widths[h]) for h in headers)
        print(row_line)


def write_results_csv(all_results: List[Dict[str, Any]], out_path: str) -> None:
    """
    Write all_results (list of dicts) to CSV. Keys union is used as columns.
    Internal helper keys starting with '_' are skipped.
    """
    if not all_results:
        print(f"No results to write to CSV: {out_path}")
        return

    # compute union of keys, preserve Group and Model first if present
    keys = set()
    for r in all_results:
        keys.update(r.keys())
    # drop internal keys
    keys = {k for k in keys if not k.startswith("_")}
    # order: Group, Model, sorted rest
    ordered = []
    for k in ("Group", "Model"):
        if k in keys:
            ordered.append(k)
            keys.remove(k)
    ordered.extend(sorted(keys))

    # ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    filename = os.path.join(out_path, "ablation-summary.csv")
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for r in all_results:
            row = {}
            for k in ordered:
                v = r.get(k)
                # convert complex types to JSON-friendly string
                if isinstance(v, (dict, list, tuple)):
                    try:
                        row[k] = json.dumps(v, ensure_ascii=False)
                    except Exception:
                        row[k] = str(v)
                else:
                    row[k] = "" if v is None else v
            writer.writerow(row)
    print(f"Wrote CSV summary to {filename}")


def plot_grouped_metrics(all_results: List[Dict[str, Any]], out_path: str) -> None:
    """
    Create grouped bar charts for AP50:95 and R2 by Model, colored by Group.
    Saves a PNG to out_path.
    """
    if not all_results:
        print("No results to plot.")
        return

    # build DataFrame
    rows = []
    for r in all_results:
        rows.append({
            "Augmentation": r.get("Group"),
            "Model": r.get("Model"),
            "AP50:95": r.get("AP50:95"),
            "R2": r.get("R2"),
        })
    df = pd.DataFrame(rows)

    # coerce numeric columns
    df["AP50:95"] = pd.to_numeric(df["AP50:95"], errors="coerce")
    df["R2"] = pd.to_numeric(df["R2"], errors="coerce")

    if df[["AP50:95", "R2"]].dropna(how="all").empty:
        print("No numeric AP50:95 or R2 values available for plotting.")
        return

    # set plotting style
    sns.set_style("whitegrid")
    models = df["Model"].astype(str).unique()
    width = max(8, len(models) * 0.8)
    fig, axes = plt.subplots(1, 2, figsize=(width * 1.6, 5))

    # AP50:95 plot
    sns.barplot(data=df, x="Model", y="AP50:95", hue="Augmentation", ax=axes[0], errorbar=None)
    axes[0].set_title("AP50:95 by Model Configuration")
    axes[0].set_ylim(0.6, 1)
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x")

    # R2 plot
    sns.barplot(data=df, x="Model", y="R2", hue="Augmentation", ax=axes[1], errorbar=None)
    axes[1].set_title("R² by Model Configuration")
    axes[1].set_ylim(0.6, 1)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("R²")
    axes[1].tick_params(axis="x")

    # place legend only once (combine)
    axes[1].legend_.remove()

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    full_dir = os.path.abspath(out_path)
    os.makedirs(full_dir, exist_ok=True)
    filename_png = os.path.join(full_dir, "ablation-summary.png")
    filename_pdf = os.path.join(full_dir, "ablation-summary.pdf")
    fig.savefig(filename_png, dpi=300, bbox_inches='tight')
    fig.savefig(filename_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote grouped vertical bar plot to {filename_png} and {filename_pdf}")
    

def plot_latency_vs_params(all_results: List[Dict[str, Any]], out_path: str):
    if not all_results:
        print("No results to plot.")
        return

    # Build DataFrame for "Copy-Paste" group only
    rows = []
    for r in all_results:
        if r.get("Group") != "Copy-Paste":
            continue
        rows.append({
            "Extractor": r.get("Model"),
            "Params": r.get("params"),
            "GPU_ms": r.get("latency"),
        })
    df = pd.DataFrame(rows)

    # Coerce numeric columns
    df["Params"] = pd.to_numeric(df["Params"], errors="coerce")
    df["GPU_ms"] = pd.to_numeric(df["GPU_ms"], errors="coerce")
    df["Params_M"] = df["Params"] / 1e6

    if df[["Params_M", "GPU_ms"]].dropna(how="any").empty:
        print("No numeric Params or Latency values available for plotting.")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df["Params_M"], df["GPU_ms"], s=80)

    # Annotate points, fine-tuning specific labels
    for i, row in df.iterrows():
        x_offset = 0.08
        ha = 'left'
        if row["Extractor"] == "MobileViTv2 (DF)":
            x_offset = 0.01   # slight left shift to avoid boundary
            ha = 'center'
        ax.annotate(
            row["Extractor"],
            (row["Params_M"] + x_offset, row["GPU_ms"] + 0.2),
            fontsize=13,
            ha=ha
        )

    # Linear fit (for visual trend)
    coeffs = np.polyfit(df["Params_M"], df["GPU_ms"], 1)
    xfit = np.linspace(min(df["Params_M"]) - 0.5, max(df["Params_M"]) + 0.5, 100)
    yfit = np.polyval(coeffs, xfit)
    ax.plot(xfit, yfit)

    # Axis labels and grid
    ax.set_xlabel("# Params (M)", fontsize=16)
    ax.set_ylabel("GPU inference latency (ms)", fontsize=16)
    ax.set_title("Model size vs GPU latency", fontsize=18)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.8)

    # Adjust margins slightly to keep labels within bounds
    plt.subplots_adjust(right=0.93)
    ax.margins(x=0.08)

    # Tight layout and save
    plt.tight_layout()
    full_dir = os.path.abspath(out_path)
    os.makedirs(full_dir, exist_ok=True)
    
    filename_png = os.path.join(full_dir, "ablation-params-vs-latency.png")
    filename_pdf = os.path.join(full_dir, "ablation-params-vs-latency.pdf")
    
    fig.savefig(filename_png, dpi=300, bbox_inches='tight')
    fig.savefig(filename_pdf, bbox_inches='tight')
    
    plt.close(fig)
    print(f"Wrote params vs latency plot to {filename_png} (and .pdf)")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize train.py results for 'seed-42' runs."
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        nargs="?",
        default="results/runs",
        help="The base directory containing model-group subfolders.",
    )
    parser.add_argument(
        "--baseline-folder",
        nargs="?",
        default=None,
        help="(Optional) Override baseline. Format: 'group:folder_name' or just 'folder_name'.",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Optional path to write summaries (default: <base-dir>)",
    )
    args = parser.parse_args()

    # MODEL_NAME_MAPS: map group_name -> (folder_name -> info)
    # info can be either a pretty name string, or a dict: {"name": "Pretty Name", "baseline": True}
    MODEL_NAME_MAPS = {
        "Copy-Paste": {
            "lettuce_model_mobnetv3_dataset_copypaste": {
                "name": "Baseline",
                "baseline": True,
            },
            "lettuce_model_multimodal_mobnetv3_dataset_copypaste": "MobileNetV3 (DF)",
            "lettuce_model_dataset_copypaste": "MobileViTv2",
            "lettuce_model_multimodal_dataset_copypaste": "MobileViTv2 (DF)",
        },
        "No Copy-Paste": {
            "lettuce_model_mobnetv3_dataset_no_copypaste": {
                "name": "Baseline",
                "baseline": True,
            },
            "lettuce_model_multimodal_mobnetv3_dataset_no_copypaste": "MobileNetV3 (DF)",
            "lettuce_model_dataset_no_copypaste": "MobileViTv2",
            "lettuce_model_multimodal_dataset_no_copypaste": "MobileViTv2 (DF)",
        },
    }

    # Backwards compatibility: accept a flat MODEL_NAME_MAP if desired
    # (not used here, but code supports it by wrapping into a single group).

    if not os.path.isdir(args.base_dir):
        print(f"Error: Base directory not found: {args.base_dir}")
        return

    print(f"Scanning for 'seed-42' runs in: {args.base_dir}")
    all_results = []

    try:
        # Iterate groups
        for group_name, group_map in MODEL_NAME_MAPS.items():
            for folder_name, info in group_map.items():
                if isinstance(info, dict):
                    pretty_name = info.get("name")
                else:
                    pretty_name = str(info)
                model_dir_path = os.path.join(args.base_dir, folder_name)
                seed_dir = os.path.join(model_dir_path, "seed-42")

                if os.path.isdir(seed_dir):
                    print(
                        f"--- Processing {pretty_name} (from {folder_name}) in group '{group_name}' ---"
                    )
                    result = process_run(
                        seed_dir, pretty_name if pretty_name else "???"
                    )
                    if result:
                        # annotate with group and folder to allow per-group baseline matching
                        result["Group"] = group_name
                        result["_folder"] = folder_name
                        all_results.append(result)
                else:
                    # Notify if entry in map but folder missing
                    if not os.path.isdir(model_dir_path):
                        print(
                            f"--- Skipping {pretty_name} (Folder not found: {folder_name}) in group '{group_name}' ---"
                        )
                    else:
                        print(
                            f"--- Skipping {pretty_name} (no 'seed-42' dir in {folder_name}) in group '{group_name}' ---"
                        )

    except FileNotFoundError:
        print(f"Error: Path not found {args.base_dir}")
    except NotADirectoryError:
        print(f"Error: Path is not a directory {args.base_dir}")

    # Compute deltas per-group
    augmented_results, baseline_rows_by_group = _compute_deltas_per_group(
        all_results, args.baseline_folder, MODEL_NAME_MAPS
    )

    any_baseline = any(b is not None for b in baseline_rows_by_group.values())
    if any_baseline:
        print("\nUsing baselines for deltas per group:")
        for g, b in baseline_rows_by_group.items():
            if b:
                print(
                    f" - Group '{g}': baseline = {b.get('Model')} (folder: {b.get('_folder')})"
                )
            else:
                print(f" - Group '{g}': NO baseline found")
    else:
        print(
            "\nNo baseline found for any group (no deltas will be shown). To set baselines, either:"
        )
        print(
            " - mark an entry in MODEL_NAME_MAPS[group] as {'name': 'Pretty', 'baseline': True}"
        )
        print(" - or pass --baseline-folder group:folder_name (or just folder_name)")

    print_results_table(augmented_results, baseline_rows_by_group)

    # Save CSV if requested (or default location)
    out_path = args.out if getattr(args, "out", None) else args.base_dir
    write_results_csv(augmented_results, out_path)

    # Save grouped bar chart (AP50:95 and R2) if requested (or default location)
    try:
        plot_grouped_metrics(augmented_results, out_path)
        plot_latency_vs_params(augmented_results, out_path)
    except Exception as e:
        print(f"Warning: Failed to create plot: {e}")


if __name__ == "__main__":
    main()
