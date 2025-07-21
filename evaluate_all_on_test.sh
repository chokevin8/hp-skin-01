#!/bin/bash

# Script to evaluate ALL models on TEST SET ONLY
# Reports epidermis-only Dice score as primary metric

# Exit on error
set -e

# Change to project directory
cd /home/wjcho/hp-skin-01

# Activate conda environment
echo "=== Activating conda environment ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hp-skin-01

# Function to evaluate a single model
evaluate_test_model() {
    local architecture=$1
    local encoder=$2
    local experiment_name=$3
    local checkpoint_path="experiments/${experiment_name}/checkpoints/best.pth"
    local output_dir="test_results/${experiment_name}"
    
    echo -e "\n=============================================="
    echo "Evaluating: ${architecture^^} + ${encoder}"
    echo "=============================================="
    
    # Check if checkpoint exists
    if [ ! -f "$checkpoint_path" ]; then
        echo "WARNING: Checkpoint not found at $checkpoint_path"
        echo "Skipping ${architecture^^} + ${encoder}"
        return 1
    fi
    
    # Run evaluation
    python src/evaluation/evaluate_test_set.py \
        --checkpoint "$checkpoint_path" \
        --architecture "$architecture" \
        --encoder "$encoder" \
        --output_dir "$output_dir" \
        --batch_size 32 \
        --num_workers 8
    
    # Read and display the Dice score
    if [ -f "$output_dir/test_results.json" ]; then
        dice_score=$(python -c "
import json
with open('$output_dir/test_results.json', 'r') as f:
    data = json.load(f)
    print(f\"{data['test_dice_epidermis']:.4f}\")
")
        echo ">>> TEST SET DICE (Epidermis-only): $dice_score"
    fi
    
    return 0
}

# Create results directory
mkdir -p test_results

echo "===================================================="
echo "EVALUATING ALL MODELS ON TEST SET"
echo "===================================================="
echo "Note: Calculating epidermis-only Dice score"
echo "      (ignoring background pixels)"
echo ""

# Track results
declare -A results

# Evaluate U-Net models
echo -e "\n### U-Net Models ###"

if evaluate_test_model "unet" "resnet50" "unet_resnet50_baseline"; then
    dice=$(python -c "import json; print(json.load(open('test_results/unet_resnet50_baseline/test_results.json'))['test_dice_epidermis'])")
    results["U-Net + ResNet50"]=$dice
fi

if evaluate_test_model "unet" "efficientnet-b3" "unet_efficientnetb3_baseline"; then
    dice=$(python -c "import json; print(json.load(open('test_results/unet_efficientnetb3_baseline/test_results.json'))['test_dice_epidermis'])")
    results["U-Net + EfficientNet-B3"]=$dice
fi

# Evaluate DeepLabV3+ models
echo -e "\n### DeepLabV3+ Models ###"

if evaluate_test_model "deeplabv3plus" "resnet50" "deeplabv3plus_resnet50_baseline"; then
    dice=$(python -c "import json; print(json.load(open('test_results/deeplabv3plus_resnet50_baseline/test_results.json'))['test_dice_epidermis'])")
    results["DeepLabV3+ + ResNet50"]=$dice
fi

if evaluate_test_model "deeplabv3plus" "efficientnet-b3" "deeplabv3plus_efficientnetb3_baseline"; then
    dice=$(python -c "import json; print(json.load(open('test_results/deeplabv3plus_efficientnetb3_baseline/test_results.json'))['test_dice_epidermis'])")
    results["DeepLabV3+ + EfficientNet-B3"]=$dice
fi

# Create final summary
echo -e "\n===================================================="
echo "FINAL TEST SET RESULTS SUMMARY"
echo "===================================================="
echo "Epidermis-only Dice Scores:"
echo ""

# Create summary file
summary_file="test_results/test_set_summary.txt"
{
    echo "TEST SET EVALUATION SUMMARY"
    echo "=========================="
    echo "Epidermis-only Dice Scores:"
    echo ""
} > $summary_file

# Display and save results
for model in "U-Net + ResNet50" "U-Net + EfficientNet-B3" "DeepLabV3+ + ResNet50" "DeepLabV3+ + EfficientNet-B3"; do
    if [[ -n "${results[$model]}" ]]; then
        printf "%-30s: %.4f\n" "$model" "${results[$model]}"
        printf "%-30s: %.4f\n" "$model" "${results[$model]}" >> $summary_file
    else
        printf "%-30s: Not evaluated\n" "$model"
        printf "%-30s: Not evaluated\n" "$model" >> $summary_file
    fi
done

echo ""
echo "Detailed results saved in: test_results/"
echo "Summary saved to: $summary_file"

# Create comparison CSV
python -c "
import json
import pandas as pd
import os

data = []
models = [
    ('unet', 'resnet50', 'unet_resnet50_baseline'),
    ('unet', 'efficientnet-b3', 'unet_efficientnetb3_baseline'),
    ('deeplabv3plus', 'resnet50', 'deeplabv3plus_resnet50_baseline'),
    ('deeplabv3plus', 'efficientnet-b3', 'deeplabv3plus_efficientnetb3_baseline'),
]

for arch, encoder, exp_name in models:
    results_file = f'test_results/{exp_name}/test_results.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
            data.append({
                'Model': f'{arch.upper()} + {encoder}',
                'Test_Dice': results['test_dice_epidermis'],
                'Test_IoU': results['test_iou'],
                'Test_Precision': results['test_precision'],
                'Test_Recall': results['test_recall'],
                'Num_Patches': results['num_test_patches']
            })

if data:
    df = pd.DataFrame(data)
    df.to_csv('test_results/test_set_comparison.csv', index=False)
    print('\nComparison CSV saved to: test_results/test_set_comparison.csv')
"