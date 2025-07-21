#!/bin/bash

# Test script to verify training pipeline before full training

# Exit on error
set -e

# Change to project directory
cd /home/wjcho/hp-skin-01

# Activate conda environment
echo "=== Activating conda environment ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hp-skin-01

# Run test
echo "=== Testing training pipeline ==="
python test_training_pipeline.py

echo "=== Test complete ==="