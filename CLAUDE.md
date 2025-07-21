# HP-Skin-01: Epidermis Segmentation Project

## Project Overview

This project develops a deep learning pipeline for epidermis segmentation in skin whole slide images (WSI) using supervised semantic segmentation. The approach uses U-Net and DeepLabv3+ models with pretrained encoders to achieve accurate epidermis segmentation across multiple histopathology datasets.

## Environment Setup

```bash
# Create conda environment
bash setup_environment.sh

# Activate environment
conda activate hp-skin-01
```

## Dataset Information

### Histo-Seg Dataset
- **Resolution**: 20x magnification
- **Structure**: `dataset/Histo-Seg/WSI/` and `dataset/Histo-Seg/Mask/`
- **File Format**: `.jpg` for both WSI and Mask
- **Mask Type**: Multiclass segmentation masks
- **Epidermis Pixel Value**: RGB(112, 48, 160)
- **Binary Mask Output**: `dataset/Histo-Seg/Binary_Mask/`

### Queensland Dataset
- **Resolution**: 10x magnification
- **Structure**: `dataset/Queensland/WSI/` and `dataset/Queensland/Mask/`
- **File Format**: `.tif` for WSI, `.png` for Mask
- **Mask Type**: Multiclass segmentation masks
- **Epidermis Pixel Value**: RGB(73, 0, 106)
- **Binary Mask Output**: `dataset/Queensland/Binary_Mask/`
- **Cancer Types**: BCC, IEC, SCC

## Data Preprocessing Pipeline

### Step 1: Binary Mask Generation
```python
# Extract epidermis-only binary masks from multiclass masks
# Histo-Seg: RGB(112, 48, 160) → 1, others → 0
# Queensland: RGB(73, 0, 106) → 1, others → 0
# Skip images with <1% epidermis pixels
```

### Step 2: WSI Tissue Segmentation
Using CLAM-inspired tissue segmentation:
```python
tissue_seg_params = {
    'seg_level': -1,      # Lowest resolution for speed
    'sthresh': 8,         # Saturation threshold
    'mthresh': 7,         # Mean threshold
    'close': 4,           # Morphological closing
    'use_otsu': False,    # Manual thresholding
    'a_t': 100,          # Area threshold
    'a_h': 16,           # Hole area threshold
    'max_n_holes': 8     # Maximum holes allowed
}
```

### Step 3: Patch Extraction
- **Patch Size**: 384×384 pixels
- **Overlap**: 0% (non-overlapping)
- **Background Removal**: Skip patches with >90% white background
- **Output**: Paired WSI-mask patches

## Model Architecture

### U-Net Implementation (Primary)
```python
# Using segmentation_models_pytorch (SMP)
models = {
    'unet_resnet50': {
        'encoder_name': 'resnet50',
        'encoder_weights': 'imagenet',
        'in_channels': 3,
        'classes': 1,
        'activation': 'sigmoid'
    },
    'unet_efficientnet_b3': {
        'encoder_name': 'efficientnet-b3',
        'encoder_weights': 'imagenet',
        'in_channels': 3,
        'classes': 1,
        'activation': 'sigmoid'
    }
}
```

### DeepLabv3+ Implementation (Secondary)
```python
# To be implemented after U-Net validation
deeplabv3plus_config = {
    'encoder_name': 'resnet50',
    'encoder_weights': 'imagenet',
    'encoder_output_stride': 16,
    'decoder_channels': 256,
    'classes': 1
}
```

## Training Configuration

### Hyperparameters
```python
training_config = {
    # Data
    'patch_size': 384,
    'batch_size': 32,  # Optimized for 48GB GPU
    'num_workers': 16,  # Increased for better data loading
    
    # Model
    'loss': 'DiceLoss',  # Pure Dice loss - epidermis class only
    'optimizer': 'AdamW',
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    
    # Training
    'epochs': 200,
    'early_stopping_patience': 15,
    'save_best_only': True,
    'monitor_metric': 'val_dice',
    
    # Augmentation
    'augmentations': {
        'HorizontalFlip': {'p': 0.5}
    },
    
    # Wandb
    'wandb_api_key': '056cc8f5fd5428b2d91107a96c0da5f5ae1c3476',
    'wandb_project': 'epidermis-segmentation',
    'wandb_entity': 'chokevin8',
    'log_frequency': 10  # Log every 10 epochs
}
```

### Class Imbalance Handling

Since most patches contain no epidermis after tissue segmentation, we handle imbalance via:

1. **Weighted Random Sampling** (Default):
   ```yaml
   use_weighted_sampling: true
   oversample_factor: null  # Auto-calculate for 50:50 balance
   ```
   - If patches are 10:1 (without:with epidermis), auto-sets oversample_factor=10.0
   - Creates balanced batches with ~50% epidermis patches
   - Can manually set oversample_factor if needed

2. **Alternative - Focal Loss**:
   ```yaml
   loss: focal
   loss_params:
     alpha: 0.25  # Weight for epidermis class
     gamma: 2.0   # Focus on hard examples
   ```

### Data Split Strategy
- **Train**: 80%
- **Validation**: 10%
- **Test**: 10%
- **Stratification**: By dataset source (maintain proportional representation)

### Learning Rate Schedule
```python
scheduler_config = {
    'type': 'ReduceLROnPlateau',
    'mode': 'max',  # Maximize Dice
    'factor': 0.5,
    'patience': 10,
    'min_lr': 1e-6
}
```

## Evaluation Metrics

### Primary Metric
- **Dice Coefficient**: Main metric for model selection and early stopping

### Secondary Metrics
```python
metrics = {
    'dice': DiceMetric(),
    'iou': JaccardIndex(),
    'sensitivity': Recall(),  # True Positive Rate
    'specificity': Specificity(),  # True Negative Rate
    'precision': Precision()
}
```

### Per-Dataset Evaluation
Track performance separately for:
- Histo-Seg (20x)
- Queensland (10x)
- Combined dataset

## Running the Pipeline

### 1. Setup Environment
```bash
bash setup_environment.sh
conda activate hp-skin-01
```

### 2. Run Complete Pipeline
```bash
bash run_training.sh
```

This will:
1. Create binary masks from multiclass masks
2. Extract patches with tissue segmentation
3. Train U-Net models with both encoders
4. Save best models based on validation Dice

### 3. Evaluate Models
```bash
bash run_evaluation.sh
```

## Project Structure

```
hp-skin-01/
├── dataset/                   # Input data (already exists)
│   ├── Histo-Seg/
│   │   ├── WSI/              # .jpg files
│   │   ├── Mask/             # .jpg files
│   │   └── Binary_Mask/      # Generated
│   └── Queensland/
│       ├── WSI/              # .tif files
│       ├── Mask/             # .png files
│       └── Binary_Mask/      # Generated
├── patches/                  # Generated patches
│   ├── Histo-Seg/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── Queensland/
│       ├── train/
│       ├── val/
│       └── test/
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── create_binary_masks.py
│   │   ├── create_patches.py
│   │   └── tissue_segmentation.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── epidermis_dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet_smp.py
│   │   └── deeplabv3plus_smp.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── losses.py
│   │   └── metrics.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   └── visualize.py
│   └── utils/
│       ├── __init__.py
│       ├── data_utils.py
│       └── training_utils.py
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── experiments/              # Training outputs
├── checkpoints/             # Model checkpoints
├── logs/                    # Training logs
├── results/                 # Evaluation results
├── setup_environment.sh     # Conda setup script
├── run_training.sh         # Training pipeline
├── run_evaluation.sh       # Evaluation script
├── requirements.txt        # Python dependencies
├── CLAUDE.md              # This file
└── README.md
```

## Key Implementation Details

### File Format Handling
- **Histo-Seg**: Uses `.jpg` for both WSI and masks
- **Queensland**: Uses `.tif` for WSI (requires openslide) and `.png` for masks
- Automatic format detection based on file extension

### Memory Optimization
- Use mixed precision training (fp16) if GPU memory is limited
- Implement gradient accumulation for effective larger batch sizes
- Cache patches to disk to reduce repeated I/O

### Multi-Resolution Handling
- Native resolution training (Histo-Seg 20x, Queensland 10x)
- Future: Implement resolution normalization if needed
- Track resolution-specific performance metrics

### Quality Control
- Log skipped images (<1% epidermis) to `preprocessing_logs.csv`
- Validate patch quality before training
- Monitor class distribution in patches

### Inference Pipeline
- Sliding window inference for full WSI
- Patch aggregation with averaging at overlaps
- Post-processing options (ready for future morphological operations)

## Success Criteria

1. **Model Performance**
   - Validation Dice > 0.85
   - Test Dice > 0.80
   - Consistent performance across both datasets

2. **Training Stability**
   - Smooth convergence
   - No overfitting (train-val gap < 0.1)
   - Reproducible results

3. **Inference Efficiency**
   - Process 1000×1000 image in < 2 seconds
   - Full WSI inference in reasonable time

## Notes & Considerations

- **Resolution Difference**: Currently training on native resolutions. Monitor if this causes issues.
- **File Format Support**: Code handles .jpg, .png, and .tif formats automatically.
- **Class Imbalance**: Most patches contain no epidermis. Handled via weighted sampling (auto-balanced).
- **Patch Selection**: Consider importance sampling in future iterations.
- **Augmentation**: Starting conservative with horizontal flips only. Can expand if needed.
- **Multi-GPU**: Ready for DDP implementation if needed.

## Troubleshooting

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi
```

### OpenSlide Issues (for .tif files)
```bash
# Install system dependencies if needed
sudo apt-get install openslide-tools
```

### Memory Issues
- Reduce batch size in training config
- Enable gradient accumulation
- Use mixed precision training

## Future Enhancements

1. **Advanced Augmentations**: Elastic deformations, rotations (if appropriate)
2. **Ensemble Methods**: Combine predictions from multiple models
3. **Active Learning**: Focus on hard-to-segment regions
4. **Foundation Models**: Explore SAM or other foundation models for medical imaging
5. **3D Context**: If z-stack data becomes available
6. **Resolution Normalization**: Standardize Histo-Seg to 10x for consistent training

Last Updated: 2025-01-17