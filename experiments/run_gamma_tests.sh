#!/bin/bash

# --- 1. VALIDATE AND ASSIGN ARGUMENTS ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <output_dir> <dataset_root> <nproc>"
    echo "Example: $0 results/gamma_tests data/cp_dataset 2"
    exit 1
fi

# Assign positional arguments to variables
OUTPUT_DIR="$1"
DATASET_ROOT="$2"
NPROC="$3"

# --- 2. DEFINE EXPERIMENT ARRAYS ---
gammas=(0.1 0.3 0.5 0.7 0.9)
seeds=(42 77 123)

# --- 3. RUN EXPERIMENT LOOP ---
for gamma in "${gammas[@]}"; do
        for seed in "${seeds[@]}"; do
            
            # Extract the base name of the data path for the output directory name
            data_basename=$(basename "$DATASET_ROOT")

            # Construct the output directory name
            output_subdir="${gamma}_${DATASET_ROOT}"

            # Construct the full output path
            # Note: Your train.py script expects the seed *in* the output path
            # but also takes a --seed argument. This matches your Python logic.
            full_output_path="${OUTPUT_DIR}/${output_subdir}/seed-${seed}"

            echo "Running experiment with gamma: $gamma and data path: $DATASET_ROOT"
            echo "Output will be saved to: $full_output_path"

            # Run the training script
            # We create the directory just in case
            mkdir -p "$full_output_path"
            
            torchrun --nproc_per_node="$NPROC" train.py \
                --data-path "$DATASET_ROOT" \
                --dataset lettuce_rgbd \
                --model lettuce_model \
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
                --phenotype-loss-weight "$gamma" \
                --phenotype-names fresh_weight \
                --seed "$seed" \
                --save-metrics \
                --measure-latency \
                --resume

            echo "Finished experiment with gamma: $gamma, data path: $DATASET_ROOT and seed $seed"
            echo "--------------------------------------------------"
        done
    done
done

echo "Gamma value tests complete."