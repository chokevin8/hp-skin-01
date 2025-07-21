# Epidermis Segmentation Project

This project implements deep learning models for epidermis segmentation in skin whole slide images using U-Net and DeepLabv3+ architectures.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data:**
   Place your datasets in the following structure:
   ```
   dataset/
   ├── Histo-Seg/
   │   ├── WSI/          # Original WSI images
   │   └── Mask/         # Multiclass masks
   └── Queensland/
       ├── WSI/          # Original WSI images
       └── Mask/         # Multiclass masks
   ```

3. **Run training pipeline:**
   ```bash
   bash run_training.sh
   ```

## Pipeline Overview

1. **Binary Mask Generation**: Extracts epidermis pixels from multiclass masks
2. **Patch Extraction**: Creates 384×384 patches with tissue segmentation
3. **Model Training**: Trains U-Net models with different encoders
4. **Evaluation**: Computes Dice, IoU, and other metrics

## Models

- U-Net with ResNet50 encoder
- U-Net with EfficientNet-B3 encoder
- DeepLabv3+ (coming soon)

## Monitoring

Training progress is logged to Weights & Biases. View your experiments at [wandb.ai](https://wandb.ai).

## Results

Best models are saved in `experiments/*/checkpoints/best.pth`.

See `CLAUDE_EPIDERMIS.md` for detailed documentation.