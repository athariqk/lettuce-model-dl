import pandas as pd
import argparse
from pathlib import Path

# --- User-defined Mappings ---

# Map internal config names to "pretty" extractor names for the table
CONFIG_MAP = {
    'lettuce_model_multimodal_dataset_copypaste': 'MobileViTv2 + DF + CA',
    'lettuce_model_dataset_copypaste': 'MobileViTv2 + CA',
    'lettuce_model_multimodal_mobnetv3_dataset_copypaste': 'MobileNetV3 + DF + CA',
    'lettuce_model_mobnetv3_dataset_copypaste': 'MobileNetV3 + CA',
    'lettuce_model_multimodal_dataset_no_copypaste': 'MobileViTv2 + DF',
    'lettuce_model_dataset_no_copypaste': 'MobileViTv2',
    'lettuce_model_multimodal_mobnetv3_dataset_no_copypaste': 'MobileNetV3 + DF',
    'lettuce_model_mobnetv3_dataset_no_copypaste': 'MobileNetV3',
}

# Map internal metric names to "pretty" column headers
# We'll use this to select *and* order the columns
METRIC_MAP = {
    'bbox_0': 'Bbox AP',
    'bbox_8': 'Bbox AR',  # This is AR @ IoU=0.50:0.95 (Max=100)
    'phenotype_fresh_weight_r2': 'R²',
    'phenotype_fresh_weight_rmse': 'RMSE',
    'phenotype_fresh_weight_mape': 'MAPE (%)',
}

# List of metrics that should be multiplied by 100 for the table
METRICS_TO_MULTIPLY = [
    'phenotype_fresh_weight_mape',
]

# --- End User-defined Mappings ---


def format_thesis_table(stats_file: Path, output_dir: Path):
    """
    Reads the descriptive_stats.csv file, formats it, and saves a
    publication-ready table as CSV and Markdown.
    """
    print(f"Reading statistics from: {stats_file}")
    if not stats_file.exists():
        print(f"Error: File not found: {stats_file}")
        print("Please run 'compare_model_configs.py' first to generate this file.")
        return

    try:
        # Read the CSV, using the first two columns as the index
        df = pd.read_csv(stats_file, index_col=[0, 1])
    except Exception as e:
        print(f"Error reading {stats_file}. Is it a valid CSV from the compare script? Error: {e}")
        return

    # Filter to keep only the metrics we care about for the table
    df = df[df.index.get_level_values('metric').isin(METRIC_MAP.keys())]
    
    # Reset index to turn 'model_config' and 'metric' into columns
    df = df.reset_index()

    # Apply the "pretty name" mappings
    df['Extractor'] = df['model_config'].map(CONFIG_MAP)
    df['Metric'] = df['metric'].map(METRIC_MAP)
    
    # Handle any unmapped configs or metrics
    df = df.dropna(subset=['Extractor', 'Metric'])
    if df.empty:
        print("Error: No data to format. Do your CONFIG_MAP and METRIC_MAP match the stats file?")
        return

    # --- Value Formatting ---
    # 1. Scale metrics that need to be percentages
    df['mean_scaled'] = df.apply(
        lambda row: row['mean'] * 100 if row['metric'] in METRICS_TO_MULTIPLY else row['mean'],
        axis=1
    )
    df['std_scaled'] = df.apply(
        lambda row: row['std'] * 100 if row['metric'] in METRICS_TO_MULTIPLY else row['std'],
        axis=1
    )

    # 2. Format as "Mean ± SD" string
    # We use .3f for raw floats (R2/AP) and .1f for scaled percentages (NRMSE/MAPE)
    df['Value'] = df.apply(
        lambda row: f"{row['mean_scaled']:.1f} ± {row['std_scaled']:.1f}" if row['metric'] in METRICS_TO_MULTIPLY 
        else f"{row['mean_scaled']:.3f} ± {row['std_scaled']:.3f}",
        axis=1
    )
    
    # --- Pivoting ---
    print("Formatting table...")
    # Pivot the table to get Extractors as rows and Metrics as columns
    final_table = df.pivot(index='Extractor', columns='Metric', values='Value')

    # --- Re-ordering ---
    # Ensure rows and columns are in a logical, not alphabetical, order
    
    # Get the "pretty" row order from the CONFIG_MAP's values
    # We use list(dict.fromkeys(...)) to get unique values in insertion order
    row_order = list(dict.fromkeys(CONFIG_MAP[key] for key in CONFIG_MAP if key in df['model_config'].unique()))
    
    # Get the "pretty" column order from the METRIC_MAP's values
    col_order = list(METRIC_MAP.values())

    # Reindex the table to match our desired order
    final_table = final_table.reindex(index=row_order, columns=col_order)
    
    # --- Saving ---
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "thesis_table.csv"
        md_path = output_dir / "thesis_table.md"

        final_table.to_csv(csv_path)
        final_table.to_markdown(md_path)
        
        print(f"\nSuccessfully created thesis table!")
        print(f"  - CSV: {csv_path}")
        print(f"  - MD:  {md_path}")
        
        print("\n--- Final Table (Markdown) ---")
        print(final_table.to_markdown())
        
    except Exception as e:
        print(f"Error saving files: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Format the descriptive_stats.csv into a publication-ready thesis table.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "stats_file",
        type=str,
        help="Path to the 'descriptive_stats.csv' file generated by 'compare_model_configs.py'."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=".",
        help="Directory to save the 'thesis_table.csv' and 'thesis_table.md' files (default: current directory)."
    )
    args = parser.parse_args()
    
    format_thesis_table(Path(args.stats_file), Path(args.output))

if __name__ == "__main__":
    main()