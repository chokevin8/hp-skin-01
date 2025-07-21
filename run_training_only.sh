#!/bin/bash

# Script to run ONLY training (skip preprocessing if already done)

# Exit on error
set -e

# Change to project directory
cd /home/wjcho/hp-skin-01

# Activate conda environment
echo "=== Activating conda environment ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hp-skin-01

# Check if patches already exist
if [ -d "patches" ] && [ -f "patches/data_splits.json" ]; then
    echo "=== Patches already exist, skipping preprocessing ==="
else
    echo "ERROR: No patches found! Run preprocessing first with:"
    echo "  bash run_preprocessing_only.sh"
    exit 1
fi

# Train models
echo "=== Training U-Net with ResNet50 ==="
python src/training/train.py \
    --encoder resnet50 \
    --wandb_project epidermis-segmentation \
    --experiment_name unet_resnet50_baseline

echo "=== Training U-Net with EfficientNet-B3 ==="
python src/training/train.py \
    --encoder efficientnet-b3 \
    --wandb_project epidermis-segmentation \
    --experiment_name unet_efficientnetb3_baseline

echo "Training complete!"