#!/bin/bash

# Script to run only preprocessing (binary masks + patches)

# Exit on error
set -e

# Change to project directory
cd /home/wjcho/hp-skin-01

# Activate conda environment
echo "=== Activating conda environment ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hp-skin-01

# Run preprocessing
echo "=== Step 1: Creating binary masks ==="
python src/preprocessing/create_binary_masks.py \
    --data_root dataset \
    --log_file preprocessing_logs.csv

echo "=== Step 2: Creating patches ==="
python src/preprocessing/create_patches.py \
    --data_root dataset \
    --output_dir patches \
    --patch_size 384 \
    --tissue_threshold 0.1 \
    --test_size 0.1 \
    --val_size 0.1

echo "Preprocessing complete!"
echo "Binary masks saved in: dataset/*/Binary_Mask/"
echo "Patches saved in: patches/"
echo "Data splits saved in: patches/data_splits.json"