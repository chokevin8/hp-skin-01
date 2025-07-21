#!/bin/bash

# Script to evaluate trained epidermis segmentation models

# Exit on error
set -e

# Change to project directory
cd /home/wjcho/hp-skin-01

# Activate conda environment
echo "=== Activating conda environment ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hp-skin-01

# Evaluate ResNet50 model
echo "=== Evaluating U-Net with ResNet50 ==="
python src/evaluation/evaluate.py \
    --checkpoint experiments/unet_resnet50_baseline/checkpoints/best.pth \
    --encoder resnet50 \
    --output_dir evaluation_results/resnet50 \
    --visualize \
    --num_visualize 20

# Evaluate EfficientNet-B3 model
echo "=== Evaluating U-Net with EfficientNet-B3 ==="
python src/evaluation/evaluate.py \
    --checkpoint experiments/unet_efficientnetb3_baseline/checkpoints/best.pth \
    --encoder efficientnet-b3 \
    --output_dir evaluation_results/efficientnetb3 \
    --visualize \
    --num_visualize 20

echo "Evaluation complete! Results saved in evaluation_results/"