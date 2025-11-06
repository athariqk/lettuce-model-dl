#!/bin/bash

# --- 1. VALIDATE AND ASSIGN ARGUMENTS ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <output_dir> <dataset_cp_root> <dataset_no_cp_root>"
    echo "Example: $0 results/ablation_study data/cp_dataset data/no_cp_dataset"
    exit 1
fi

# Assign positional arguments to variables
OUTPUT_DIR="$1"
DATASET_CP_ROOT="$2"
DATASET_NO_CP_ROOT="$3"

echo "--- Configuration ---"
echo "Base Output Dir: $OUTPUT_DIR"
echo "Copy-Paste Dataset: $DATASET_CP_ROOT"
echo "No Copy-Paste Dataset: $DATASET_NO_CP_ROOT"
echo "---------------------"
# --- End Configuration ---

# --- 2. DEFINE EXPERIMENT ARRAYS ---
models=("lettuce_model" "lettuce_model_multimodal" "lettuce_model_mobnetv3" "lettuce_model_multimodal_mobnetv3")
data_paths=("$DATASET_CP_ROOT" "$DATASET_NO_CP_ROOT")
seeds=(42 77 123 2003)

# --- 3. RUN EXPERIMENT LOOP ---
for model in "${models[@]}"; do
    for data_path in "${data_paths[@]}"; do
        for seed in "${seeds[@]}"; do
            
            # Extract the base name of the data path for the output directory name
            data_basename=$(basename "$data_path")

            # Construct the output directory name
            output_subdir="${model}_${data_basename}"

            # Construct the full output path
            # Note: Your train.py script expects the seed *in* the output path
            # but also takes a --seed argument. This matches your Python logic.
            full_output_path="${OUTPUT_DIR}/${output_subdir}/seed-${seed}"

            echo "Running experiment with model: $model and data path: $data_path"
            echo "Output will be saved to: $full_output_path"

            # Run the training script
            # We create the directory just in case
            mkdir -p "$full_output_path"
            
            torchrun --nproc_per_node=2 train.py \
                --data-path "$data_path" \
                --dataset lettuce_rgbd \
                --model "$model" \
                --epochs 50 \
                --aspect-ratio-group-factor 3 \
                --opt adamw \
                --lr-scheduler cosineannealinglr \
                --lr 0.001 \
                --batch-size 32 \
                --weight-decay 0.05 \
                --data-augmentation lettuce_rgbd \
                --use-v2 \
                --output-dir "$full_output_path" \
                --trainable-backbone-layers 2 \
                --k-folds 0 \
                --phenotype-loss-weight 0.1 \
                --phenotype-names fresh_weight \
                --seed "$seed" \
                --save-metrics \
                --measure-latency \
                --resume

            echo "Finished experiment with model: $model, data path: $data_path and seed $seed"
            echo "--------------------------------------------------"
        done
    done
done

echo "Ablation study complete."