# HP-Skin-01: Epidermis Segmentation Project

This project implements deep learning models for epidermis segmentation in skin whole slide images using U-Net and DeepLabv3+ architectures.

## Quick Start

### 1. Setup Environment

```bash
# Create and setup conda environment
bash setup_environment.sh

# Activate environment
conda activate hp-skin-01
```

### 2. Run Complete Pipeline

```bash
# Run preprocessing and training
bash run_training.sh

# Evaluate trained models
bash run_evaluation.sh
```

## Dataset Structure

Place your datasets in the following structure:
```
dataset/
├── Histo-Seg/
│   ├── WSI/          # .jpg files (20x resolution)
│   └── Mask/         # .jpg files (multiclass masks)
└── Queensland/
    ├── WSI/          # .tif files (10x resolution)
    └── Mask/         # .png files (multiclass masks)
```

## Pipeline Overview

1. **Binary Mask Generation**: Extracts epidermis pixels from multiclass masks
   - Histo-Seg: RGB(112, 48, 160)
   - Queensland: RGB(73, 0, 106)

2. **Patch Extraction**: Creates 384×384 patches with tissue segmentation
   - Non-overlapping patches
   - Tissue segmentation to remove background
   - Paired WSI-mask patches

3. **Model Training**: Trains U-Net models with different encoders
   - ResNet50 encoder
   - EfficientNet-B3 encoder
   - Pure Dice loss
   - Wandb integration for experiment tracking

4. **Evaluation**: Computes Dice, IoU, and other metrics on test set

## Models

- **U-Net with ResNet50**: Balanced performance
- **U-Net with EfficientNet-B3**: Higher accuracy, more parameters
- **DeepLabv3+**: Coming soon

## File Formats

- **Histo-Seg**: .jpg for both WSI and masks
- **Queensland**: .tif for WSI (requires OpenSlide), .png for masks

## Results

- Models are saved in `experiments/*/checkpoints/`
- Evaluation results in `evaluation_results/`
- Training progress tracked on Weights & Biases

## Troubleshooting

### OpenSlide Installation (for .tif support)
```bash
# Ubuntu/Debian
sudo apt-get install openslide-tools

# macOS
brew install openslide
```

### GPU Memory Issues
- Reduce batch size in `configs/training_config.yaml`
- Enable gradient accumulation
- Use mixed precision training

### Environment Issues
```bash
# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import segmentation_models_pytorch as smp; print('SMP installed')"
```

## Documentation

See `CLAUDE.md` for detailed project documentation and implementation details.

## License

This project is for research purposes only.