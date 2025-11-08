#!/usr/bin/env python3

"""
Analyzes and compares multiple model configurations from multi-seed training runs.

This script scans a base directory for all model configuration subfolders.
For each configuration folder, it finds all 'seed-*' sub-runs.
It extracts the best-performing metrics from each seed, based on the
'best_val_loss' in 'run_results.json' and the corresponding 'epoch_log.csv'.

It then performs two levels of analysis:

1.  **Descriptive Statistics:** Aggregates metrics for each model configuration
    (mean, std, 95% CI) to show the performance and stability of each model.

2.  **Significance Testing:** Performs a pairwise Mann-Whitney U test
    between all model configurations for each metric to determine if the
    differences between models are statistically significant (p < 0.05).

All results are printed to the console and saved to a structured report
(CSV and Markdown files) in the specified --output directory.
"""

import argparse
import json
import pandas as pd
import numpy as np
import sys
import itertools
from pathlib import Path
from scipy.stats import t, mannwhitneyu
from typing import Dict, Any, Union, List, Optional, Callable

import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper Functions (No changes here) ---

def flatten_metrics(data: Union[Dict, List, Any], prefix: str = '') -> Dict[str, Any]:
    """
    Recursively flattens a nested dictionary or list of metrics.
    """
    flat_metrics = {}
    
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}_{key}" if prefix else key
            flat_metrics.update(flatten_metrics(value, new_prefix))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            flat_metrics.update(flatten_metrics(value, f"{prefix}_{i}"))
    else:
        if prefix:
            flat_metrics[prefix] = data
        else:
            flat_metrics["value"] = data
            
    return flat_metrics

def process_seed_run(seed_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Extracts the best-epoch metrics from a single seed run directory.
    
    Uses run_results.json to find the best_val_loss, then finds the
    matching row in epoch_log.csv to extract all metrics.
    """
    log_file = seed_dir / 'epoch_log.csv'
    results_json_file = seed_dir / 'run_results.json'

    if not log_file.exists():
        print(f"  Warning: 'epoch_log.csv' not found in {seed_dir}. Skipping.", file=sys.stderr)
        return None
    
    if not results_json_file.exists():
        print(f"  Warning: 'run_results.json' not found in {seed_dir}. Skipping.", file=sys.stderr)
        return None

    # Load run_results.json to get the authoritative best_val_loss
    try:
        with open(results_json_file, 'r') as f:
            run_results = json.load(f)
        
        if 'best_val_loss' not in run_results:
            print(f"  Warning: 'best_val_loss' not found in {results_json_file}. Skipping.", file=sys.stderr)
            return None
        
        best_val_loss_from_json = run_results['best_val_loss']
    except Exception as e:
        print(f"  Error reading {results_json_file}: {e}", file=sys.stderr)
        return None

    # Read the log file
    try:
        df = pd.read_csv(log_file)
    except Exception as e:
        print(f"  Error reading {log_file}: {e}", file=sys.stderr)
        return None
    
    if 'val_loss' not in df.columns:
        print(f"  Warning: 'val_loss' column not found in {log_file}. Skipping.", file=sys.stderr)
        return None

    # Find the best epoch by matching the val_loss from run_results.json
    best_rows = df[np.isclose(df['val_loss'], best_val_loss_from_json)]

    if best_rows.empty:
        print(f"  Warning: 'best_val_loss' ({best_val_loss_from_json:.6f}) from JSON not found in {log_file}. Skipping.", file=sys.stderr)
        return None
    
    best_row = best_rows.iloc[0] # Take the first match
    
    epoch = best_row['epoch']
    val_loss = best_row['val_loss']

    # Extract and parse the metrics JSON
    if 'eval_metrics_json' not in best_row:
        print(f"  Warning: 'eval_metrics_json' not found. Skipping metrics.", file=sys.stderr)
        return None
        
    json_string = best_row['eval_metrics_json']
    
    try:
        metrics_data = json.loads(json_string)
    except json.JSONDecodeError:
        try:
            metrics_data = json.loads(json_string.replace('""', '"'))
        except Exception as e:
            print(f"  Warning: Could not parse JSON in {seed_dir}: {e}. Skipping.", file=sys.stderr)
            return None

    # Flatten the nested metrics
    flat_metrics = flatten_metrics(metrics_data)
    flat_metrics['seed'] = seed_dir.name
    flat_metrics['best_epoch'] = epoch
    flat_metrics['best_val_loss'] = val_loss
    
    return flat_metrics

# --- Analysis Functions (Refactored to return DataFrames) ---

def generate_descriptive_stats(master_df: pd.DataFrame, log: Callable) -> Optional[pd.DataFrame]:
    """
    Calculates and prints descriptive statistics (mean, std, CI)
    grouped by model configuration, using the provided log function.
    
    Returns a combined DataFrame of all stats.
    """
    log("\n" + "="*80)
    log(" Descriptive Statistics per Model Configuration")
    log("="*80)
    
    # Get all numeric metric columns, excluding housekeeping ones
    numeric_cols = master_df.select_dtypes(include=np.number).columns
    metric_cols = [col for col in numeric_cols if col not in ['best_epoch']]
    
    # Use groupby to calculate stats for each config
    grouped = master_df.groupby('model_config')
    
    all_stats_dfs = []
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000, 'display.float_format', '{:,.4f}'.format):
        for config_name, group_df in grouped:
            log(f"\n--- Configuration: {config_name} (n={len(group_df)}) ---")
            
            # Use the logic from the old script
            desc = group_df[metric_cols].describe().T
            n = len(group_df)
            
            if n > 1:
                sem = desc['std'] / np.sqrt(n)
                desc['sem'] = sem
                
                # t-distribution is more accurate for small n
                ci_margin = sem * t.ppf((1 + 0.95) / 2., n - 1)
                desc['95%_ci_low'] = desc['mean'] - ci_margin
                desc['95%_ci_high'] = desc['mean'] + ci_margin
                stat_cols = ['mean', 'std', 'sem', '95%_ci_low', '95%_ci_high', 'min', 'max']
            else:
                stat_cols = ['mean', 'min', 'max']

            log_df = desc[stat_cols].copy()
            log(log_df.to_string())
            
            log_df['model_config'] = config_name
            all_stats_dfs.append(log_df)

    if not all_stats_dfs:
        return None
        
    combined_stats_df = pd.concat(all_stats_dfs)
    combined_stats_df = combined_stats_df.set_index('model_config', append=True).reorder_levels([1, 0])
    combined_stats_df.index.names = ['model_config', 'metric']
    return combined_stats_df

def generate_significance_tests(master_df: pd.DataFrame, log: Callable) -> Optional[pd.DataFrame]:
    """
    Runs pairwise Mann-Whitney U tests between all model configurations
    and reports the p-values, using the provided log function.
    
    Returns a combined DataFrame of all test results.
    """
    log("\n" + "="*80)
    log(" Pairwise Statistical Significance (Mann-Whitney U test)")
    log("="*80)
    log(" p-value < 0.05 indicates a statistically significant difference.")
    log(" NOTE: With n=4, only very large differences will be significant.")
    log("       Always check the 95% CIs from the summary above.")
    
    configs = master_df['model_config'].unique()
    if len(configs) < 2:
        log("\nNot enough model configurations (min 2) to run a comparison.")
        return None

    # Get all numeric metric columns
    numeric_cols = master_df.select_dtypes(include=np.number).columns
    # --- THIS IS THE FIX ---
    # We remove 'best_val_loss' from the exclusion list, so it gets tested.
    metric_cols = [col for col in numeric_cols if col not in ['best_epoch']]
    # --- END FIX ---

    all_results_dfs = []

    # Get all unique pairs of configurations
    for (config_a, config_b) in itertools.combinations(configs, 2):
        log(f"\n--- Comparing: '{config_a}' vs. '{config_b}' ---")
        
        results = []
        data_a_all = master_df[master_df['model_config'] == config_a]
        data_b_all = master_df[master_df['model_config'] == config_b]

        for metric in metric_cols:
            data_a = data_a_all[metric].dropna()
            data_b = data_b_all[metric].dropna()
            
            # Need at least one data point in each group to run test
            if len(data_a) == 0 or len(data_b) == 0:
                continue

            try:
                # alternative='two-sided' is the default, for (A != B)
                stat, p_value = mannwhitneyu(data_a, data_b, alternative='two-sided')
                
                # Report which one is better (lower is better for loss/rmse, higher for r2)
                mean_a = data_a.mean()
                mean_b = data_b.mean()
                
                is_loss_metric = 'loss' in metric.lower() or 'rmse' in metric.lower() or 'mape' in metric.lower()
                
                if is_loss_metric:
                    better = config_a if mean_a < mean_b else config_b
                else: # Assuming higher is better (e.g., r2)
                    better = config_a if mean_a > mean_b else config_b
                
                results.append({
                    'metric': metric, 
                    'p_value': p_value,
                    f'mean_{config_a}': mean_a,
                    f'mean_{config_b}': mean_b,
                    'better_mean': better
                })
            except ValueError as e:
                # This can happen if all values are identical
                print(f"  Skipping {metric}: {e}", file=sys.stderr)
                
        if not results:
            log("  No common metrics found to compare.")
            continue
            
        # Create a DataFrame and highlight significant results
        results_df = pd.DataFrame(results).sort_values('p_value')
        
        # Add comparison info for the combined DataFrame
        results_df['comparison'] = f"{config_a}_vs_{config_b}"
        all_results_dfs.append(results_df)
        
        def highlight_significant(row):
            return ['font-weight: bold; background-color: #e0ffe0'] * len(row) if row.p_value < 0.05 else [''] * len(row)

        styled_df = results_df.style.apply(highlight_significant, axis=1).format({
            'p_value': '{:.4f}',
            f'mean_{config_a}': '{:.4f}',
            f'mean_{config_b}': '{:.4f}',
        })
        
        with pd.option_context('display.max_rows', 100, 'display.max_columns', None, 'display.width', 1000):
            log(styled_df.to_string())

    if not all_results_dfs:
        return None
        
    combined_sig_df = pd.concat(all_results_dfs, ignore_index=True)
    return combined_sig_df

def generate_markdown_report(
    master_df: pd.DataFrame, 
    stats_df: Optional[pd.DataFrame], 
    sig_df: Optional[pd.DataFrame], 
    report_path: Path
):
    """
    Generates a human-readable Markdown report from all analysis DataFrames.
    """
    print(f"\nGenerating Markdown report at {report_path}...")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Model Configuration Analysis Report\n\n")
            
            f.write(f"Report generated on: {pd.Timestamp.now()}\n\n")

            f.write("## 1. Master Data (Best Epoch From Every Run)\n\n")
            f.write("This table shows the raw metrics extracted from the best-performing epoch (by val_loss) for every seed run.\n\n")
            f.write(master_df.to_markdown(index=False, floatfmt=",.4f"))
            
            f.write("\n\n## 2. Descriptive Statistics per Configuration\n\n")
            f.write("This table summarizes the performance and stability of each model configuration across all its seed runs.\n\n")
            if stats_df is not None:
                f.write(stats_df.to_markdown(floatfmt=",.4f"))
            else:
                f.write("No statistics generated.\n")
                
            f.write("\n\n## 3. Pairwise Statistical Significance (Mann-Whitney U Test)\n\n")
            f.write("This table shows the p-value for each metric when comparing model configurations. \n")
            f.write("**A p-value < 0.05 (highlighted with `***`) indicates a statistically significant difference.**\n\n")
            if sig_df is not None:
                # Add a 'significant' column for easier reading in Markdown
                sig_df_md = sig_df.copy()
                sig_df_md['significant'] = sig_df_md['p_value'].apply(lambda x: '***' if x < 0.05 else '')
                
                # Reorder to put 'significant' next to p-value
                cols = sig_df_md.columns.tolist()
                cols.insert(cols.index('p_value') + 1, cols.pop(cols.index('significant')))
                sig_df_md = sig_df_md[cols]
                
                f.write(sig_df_md.to_markdown(index=False, floatfmt=",.4f"))
            else:
                f.write("No significance tests run.\n")
        print("Markdown report generation successful.")
    except Exception as e:
        print(f"Error generating Markdown report: {e}", file=sys.stderr)


def generate_significance_heatmap(
    sig_df: pd.DataFrame, 
    master_df: pd.DataFrame,  # We now need master_df to get all config names
    output_dir: Path, 
    log: Callable
):
    """
    Generates a separate p-value matrix heatmap for each metric.
    
    Each heatmap shows a grid of Model vs. Model comparisons.
    """
    print(f"\nGenerating significance heatmaps (one per metric)...")

    # Create a sub-directory for the heatmaps
    heatmap_dir = output_dir / "significance_heatmaps"
    try:
        heatmap_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"  Warning: Could not create heatmap directory {heatmap_dir}. Error: {e}", file=sys.stderr)
        return

    all_metrics = sig_df['metric'].unique()
    all_configs = master_df['model_config'].unique()
    
    if len(all_configs) < 2:
        print("  ...Skipped: Need at least 2 configs for a matrix.")
        return

    generated_count = 0
    for metric in all_metrics:
        print(f"  - Generating heatmap for: {metric}")
        try:
            # Filter sig_df for the current metric
            metric_sig_df = sig_df[sig_df['metric'] == metric]
            
            # Create an empty NxN matrix, initialized with NaN
            matrix = pd.DataFrame(np.nan, index=all_configs, columns=all_configs)
            
            # Fill in the p-values from the significance test results
            for (config_a, config_b) in itertools.combinations(all_configs, 2):
                comparison_name_1 = f"{config_a}_vs_{config_b}"
                comparison_name_2 = f"{config_b}_vs_{config_a}" # Check both possible names
                
                p_row = metric_sig_df[
                    (metric_sig_df['comparison'] == comparison_name_1) |
                    (metric_sig_df['comparison'] == comparison_name_2)
                ]
                
                if not p_row.empty:
                    p_value = p_row.iloc[0]['p_value']
                    matrix.loc[config_a, config_b] = p_value
                    matrix.loc[config_b, config_a] = p_value
            
            # --- Plotting the Matrix ---
            fig_size = max(8, len(all_configs) * 1.5) # Dynamic size
            plt.figure(figsize=(fig_size, fig_size - 1)) # Slightly wider than tall
            
            ax = sns.heatmap(
                matrix,
                annot=True,          # Show p-values in cells
                fmt=".3f",           # Format to 3 decimal places
                cmap="RdYlGn_r",     # Green = significant (low p), Red = not (high p)
                vmin=0.0,
                vmax=0.1,            # Cap color scale at 0.1 for better contrast
                linewidths=.5,
                cbar_kws={'label': 'p-value (capped at 0.1)'},
                square=True
            )
            
            ax.set_title(f"Pairwise Significance (p-value) for: {metric}", fontsize=16)
            ax.set_xlabel("Model Configuration", fontsize=12)
            ax.set_ylabel("Model Configuration", fontsize=12)
            ax.set_facecolor('gainsboro') # Gray for diagonal (NaN)
            
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Sanitize metric name for a safe filename
            safe_metric_name = "".join(c for c in metric if c.isalnum() or c in ('_', '-')).rstrip()
            save_path = heatmap_dir / f"p_value_matrix_{safe_metric_name}.png"
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close() # Close the figure to free up memory
            generated_count += 1

        except Exception as e:
            print(f"    Error generating heatmap for {metric}: {e}", file=sys.stderr)
    
    if generated_count > 0:
        print(f"Saved {generated_count} heatmaps to {heatmap_dir}")

# --- Main execution logic (Refactored to support directory output) ---

def run_analysis(base_dir_str: str, output_dir_str: Optional[str]):
    """
    Main analysis function.
    
    :param base_dir_str: The path to the base experiments directory.
    :param output_dir_str: Optional path to an output directory for reports.
    """
    
    # --- Set up logging ---
    # We'll use print() for user-facing progress/file save messages.
    # We'll use log() for dumping data tables to the console.
    # If an output_dir is specified, we'll disable the log() function.
    
    def do_nothing_log(*args, **kwargs):
        """A log function that does nothing."""
        pass

    output_dir: Optional[Path] = None
    if output_dir_str:
        output_dir = Path(output_dir_str)
        log = do_nothing_log  # Silence data table dumps to console
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving reports to: {output_dir.resolve()}")
        except Exception as e:
            print(f"Warning: Could not create output directory {output_dir}. Reports will not be saved. Error: {e}", file=sys.stderr)
            output_dir = None
    else:
        # No output dir, print data tables to console
        log = print
        print("No --output directory specified. Printing reports to console.")

    base_dir = Path(base_dir_str)
    if not base_dir.is_dir():
        print(f"Error: Base directory not found: {base_dir_str}", file=sys.stderr)
        sys.exit(1)
        
    # Find all subdirectories in the base_dir
    try:
        config_dirs = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    except Exception as e:
        print(f"Error reading directory {base_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    if not config_dirs:
        print(f"Error: No model configuration subfolders found in {base_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(config_dirs)} model configurations:")
    for d in config_dirs:
        print(f"- {d.name}")

    all_runs_data = []
    
    # Loop over each model configuration
    for config_dir in config_dirs:
        # Use log() for console/file, but print() for transient progress
        print(f"\nProcessing configuration: {config_dir.name}...")
        config_name = config_dir.name
        
        seed_dirs = [d for d in config_dir.glob('seed-*') if d.is_dir()]
        
        if not seed_dirs:
            print(f"  Warning: No 'seed-*' directories found in {config_dir}. Skipping.", file=sys.stderr)
            continue
            
        print(f"  Found {len(seed_dirs)} seed runs.")

        # Loop over each seed in that configuration
        for seed_dir in seed_dirs:
            metrics = process_seed_run(seed_dir)
            if metrics:
                metrics['model_config'] = config_name
                all_runs_data.append(metrics)
    
    if not all_runs_data:
        print("\nNo data successfully extracted from any run. Exiting.")
        sys.exit(1)

    # Create the master DataFrame
    master_df = pd.DataFrame(all_runs_data)
    
    # Reorder columns
    cols = master_df.columns.tolist()
    housekeeping_cols = ['model_config', 'seed', 'best_epoch', 'best_val_loss']
    metric_cols = sorted([c for c in cols if c not in housekeeping_cols])
    master_df = master_df[housekeeping_cols + metric_cols]
    
    log("\n" + "="*80)
    log(" Master Data (Best epoch from every run)")
    log("="*80)
    with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1000, 'display.float_format', '{:,.4f}'.format):
        log(master_df.to_string())
        
    # Save master data CSV
    if output_dir:
        try:
            master_df.to_csv(output_dir / "master_data.csv", index=False)
            print(f"Saved master_data.csv to {output_dir}")
        except Exception as e:
            print(f"Error saving master_data.csv: {e}", file=sys.stderr)

    # --- Analysis ---
    
    # 1. Descriptive Stats
    stats_df = generate_descriptive_stats(master_df, log)
    
    # Save descriptive stats CSV
    if output_dir and stats_df is not None:
        try:
            stats_df.to_csv(output_dir / "descriptive_stats.csv")
            print(f"Saved descriptive_stats.csv to {output_dir}")
        except Exception as e:
            print(f"Error saving descriptive_stats.csv: {e}", file=sys.stderr)
    
    # 2. Significance Testing
    sig_df = generate_significance_tests(master_df, log)
    
    # Save significance tests CSV
    if output_dir and sig_df is not None:
        try:
            sig_df.to_csv(output_dir / "significance_tests.csv", index=False)
            print(f"Saved significance_tests.csv to {output_dir}")
        except Exception as e:
            print(f"Error saving significance_tests.csv: {e}", file=sys.stderr)

    # 3. Generate Markdown Report
    if output_dir:
        generate_markdown_report(master_df, stats_df, sig_df, output_dir / "analysis_report.md")

    # 4. Generate Significance Heatmap
    if output_dir and sig_df is not None:
        # Pass master_df to get the full list of configs
        generate_significance_heatmap(sig_df, master_df, output_dir, log)


def main():
    """
    Parses command line arguments and starts the analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze and compare metrics from multi-seed, multi-configuration training runs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "-d", "--base_dir",
        type=str,
        help="The base directory containing all model configuration subfolders."
             "\nExample: /path/to/experiments/"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional: Path to a directory to save structured reports (CSV, MD)."
    )
    
    args = parser.parse_args()
    
    # Pass the output directory string (or None) to the main analysis function
    run_analysis(args.base_dir, args.output)


if __name__ == "__main__":
    main()