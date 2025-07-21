#!/bin/bash

# Script to train ALL models (U-Net and DeepLabV3+ with ResNet50 and EfficientNet-B3)

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

# Train U-Net models
echo "===================================================="
echo "=== PHASE 1: Training U-Net Models ==="
echo "===================================================="

echo -e "\n=== Training U-Net with ResNet50 ==="
python src/training/train_universal.py \
    --architecture unet \
    --encoder resnet50 \
    --wandb_project epidermis-segmentation \
    --experiment_name unet_resnet50_baseline

echo -e "\n=== Training U-Net with EfficientNet-B3 ==="
python src/training/train_universal.py \
    --architecture unet \
    --encoder efficientnet-b3 \
    --wandb_project epidermis-segmentation \
    --experiment_name unet_efficientnetb3_baseline

# Train DeepLabV3+ models
echo -e "\n===================================================="
echo "=== PHASE 2: Training DeepLabV3+ Models ==="
echo "===================================================="

echo -e "\n=== Training DeepLabV3+ with ResNet50 ==="
python src/training/train_universal.py \
    --architecture deeplabv3plus \
    --encoder resnet50 \
    --wandb_project epidermis-segmentation \
    --experiment_name deeplabv3plus_resnet50_baseline

echo -e "\n=== Training DeepLabV3+ with EfficientNet-B3 ==="
python src/training/train_universal.py \
    --architecture deeplabv3plus \
    --encoder efficientnet-b3 \
    --wandb_project epidermis-segmentation \
    --experiment_name deeplabv3plus_efficientnetb3_baseline

echo -e "\n===================================================="
echo "=== All models trained successfully! ==="
echo "===================================================="

# Summary
echo -e "\nTrained models:"
echo "1. U-Net + ResNet50"
echo "2. U-Net + EfficientNet-B3"
echo "3. DeepLabV3+ + ResNet50"
echo "4. DeepLabV3+ + EfficientNet-B3"
echo -e "\nCheck experiments directory for checkpoints and logs."