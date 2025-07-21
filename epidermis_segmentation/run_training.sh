#!/bin/bash

# Script to run epidermis segmentation training

# Activate conda environment if needed
# conda activate epidermis

# Run preprocessing if not done
echo "=== Step 1: Creating binary masks ==="
python src/preprocessing/create_binary_masks.py \
    --data_root dataset \
    --min_epidermis_ratio 0.01 \
    --log_file preprocessing_logs.csv

echo "=== Step 2: Creating patches ==="
python src/preprocessing/create_patches.py \
    --data_root dataset \
    --output_dir patches \
    --patch_size 384 \
    --tissue_threshold 0.1 \
    --test_size 0.1 \
    --val_size 0.1

# Train models
echo "=== Step 3: Training U-Net with ResNet50 ==="
python src/training/train.py \
    --encoder resnet50 \
    --wandb_project epidermis-segmentation \
    --experiment_name unet_resnet50_baseline

echo "=== Step 4: Training U-Net with EfficientNet-B3 ==="
python src/training/train.py \
    --encoder efficientnet-b3 \
    --wandb_project epidermis-segmentation \
    --experiment_name unet_efficientnetb3_baseline

echo "Training complete!"