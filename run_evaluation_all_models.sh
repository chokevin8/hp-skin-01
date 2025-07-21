#!/bin/bash

# Script to evaluate ALL trained models

# Exit on error
set -e

# Change to project directory
cd /home/wjcho/hp-skin-01

# Activate conda environment
echo "=== Activating conda environment ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hp-skin-01

# Function to evaluate a model
evaluate_model() {
    local architecture=$1
    local encoder=$2
    local checkpoint_path=$3
    local output_dir=$4
    
    echo -e "\n=== Evaluating ${architecture^^} with ${encoder} ==="
    
    if [ -f "$checkpoint_path" ]; then
        python src/evaluation/evaluate_universal.py \
            --checkpoint "$checkpoint_path" \
            --architecture "$architecture" \
            --encoder "$encoder" \
            --output_dir "$output_dir" \
            --batch_size 32 \
            --num_visualizations 50
    else
        echo "WARNING: Checkpoint not found at $checkpoint_path"
        echo "Skipping evaluation for ${architecture^^} with ${encoder}"
    fi
}

# Evaluate U-Net models
echo "===================================================="
echo "=== PHASE 1: Evaluating U-Net Models ==="
echo "===================================================="

evaluate_model "unet" "resnet50" \
    "experiments/unet_resnet50_baseline/checkpoints/best.pth" \
    "evaluation_results/unet_resnet50"

evaluate_model "unet" "efficientnet-b3" \
    "experiments/unet_efficientnetb3_baseline/checkpoints/best.pth" \
    "evaluation_results/unet_efficientnetb3"

# Evaluate DeepLabV3+ models
echo -e "\n===================================================="
echo "=== PHASE 2: Evaluating DeepLabV3+ Models ==="
echo "===================================================="

evaluate_model "deeplabv3plus" "resnet50" \
    "experiments/deeplabv3plus_resnet50_baseline/checkpoints/best.pth" \
    "evaluation_results/deeplabv3plus_resnet50"

evaluate_model "deeplabv3plus" "efficientnet-b3" \
    "experiments/deeplabv3plus_efficientnetb3_baseline/checkpoints/best.pth" \
    "evaluation_results/deeplabv3plus_efficientnetb3"

echo -e "\n===================================================="
echo "=== All evaluations complete! ==="
echo "===================================================="

# Create comparison report
echo -e "\n=== Creating comparison report ==="
python -c "
import json
import os
import pandas as pd

results = []
models = [
    ('U-Net', 'ResNet50', 'evaluation_results/unet_resnet50'),
    ('U-Net', 'EfficientNet-B3', 'evaluation_results/unet_efficientnetb3'),
    ('DeepLabV3+', 'ResNet50', 'evaluation_results/deeplabv3plus_resnet50'),
    ('DeepLabV3+', 'EfficientNet-B3', 'evaluation_results/deeplabv3plus_efficientnetb3'),
]

for arch, encoder, result_dir in models:
    result_file = os.path.join(result_dir, 'evaluation_results.json')
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
            metrics = data['metrics']
            results.append({
                'Architecture': arch,
                'Encoder': encoder,
                'Dice': metrics['dice'],
                'IoU': metrics['iou'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'Has_Epi_Acc': metrics['has_epidermis_accuracy']
            })

if results:
    df = pd.DataFrame(results)
    print('\nModel Comparison:')
    print(df.to_string(index=False, float_format='%.4f'))
    df.to_csv('evaluation_results/model_comparison.csv', index=False)
    print('\nComparison saved to: evaluation_results/model_comparison.csv')
else:
    print('No evaluation results found!')
"

echo -e "\nCheck evaluation_results/ directory for detailed results and visualizations."