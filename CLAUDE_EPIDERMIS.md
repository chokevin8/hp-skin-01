# Epidermis Segmentation Project

## Project Overview

This project develops a deep learning pipeline for epidermis segmentation in skin whole slide images (WSI) using supervised semantic segmentation. The approach uses U-Net and DeepLabv3+ models with pretrained encoders to achieve accurate epidermis segmentation across multiple histopathology datasets.

## Dataset Information

### Histo-Seg Dataset
- **Resolution**: 20x magnification
- **Structure**: `dataset/Histo-Seg/WSI/` and `dataset/Histo-Seg/Mask/`
- **Mask Type**: Multiclass segmentation masks
- **Epidermis Pixel Value**: RGB(112, 48, 160)
- **Binary Mask Output**: `dataset/Histo-Seg/Binary_Mask/`

### Queensland Dataset
- **Resolution**: 10x magnification
- **Structure**: `dataset/Queensland/WSI/` and `dataset/Queensland/Mask/`
- **Mask Type**: Multiclass segmentation masks
- **Epidermis Pixel Value**: RGB(73, 0, 106)
- **Binary Mask Output**: `dataset/Queensland/Binary_Mask/`

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
    'batch_size': 16,
    'num_workers': 8,
    
    # Model
    'loss': 'DiceLoss',  # Pure Dice loss
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
    'wandb_entity': 'your-entity',  # Update this
    'log_frequency': 10  # Log every 10 epochs
}
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

## Implementation Timeline

### Phase 1: Data Preparation (Day 1-2)
- [x] Create project structure
- [ ] Implement binary mask generation (`src/preprocessing/create_binary_masks.py`)
- [ ] Implement WSI patching (`src/preprocessing/create_patches.py`)
- [ ] Validate preprocessing pipeline

### Phase 2: Model Development (Day 3-4)
- [ ] Create dataset class (`src/datasets/epidermis_dataset.py`)
- [ ] Implement U-Net models (`src/models/unet_smp.py`)
- [ ] Setup training pipeline (`src/training/train.py`)
- [ ] Integrate wandb logging

### Phase 3: Training & Evaluation (Day 5-6)
- [ ] Train U-Net + ResNet50
- [ ] Train U-Net + EfficientNet-B3
- [ ] Evaluate models on test set
- [ ] Generate prediction visualizations

### Phase 4: Advanced Models (Day 7)
- [ ] Implement DeepLabv3+
- [ ] Compare all models
- [ ] Select best performing model
- [ ] Create inference pipeline

## Project Structure

```
epidermis_segmentation/
├── dataset/
│   ├── Histo-Seg/
│   │   ├── WSI/
│   │   ├── Mask/
│   │   └── Binary_Mask/        # Generated
│   └── Queensland/
│       ├── WSI/
│       ├── Mask/
│       └── Binary_Mask/        # Generated
├── patches/                    # Generated patches
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
├── checkpoints/               # Model checkpoints
├── logs/                      # Training logs
├── results/                   # Evaluation results
├── requirements.txt
├── CLAUDE.md                  # HER2 project documentation
├── CLAUDE_EPIDERMIS.md       # This file
└── README.md
```

## Dependencies

```
# Core
python>=3.8
torch>=2.0.0
torchvision>=0.15.0

# Segmentation
segmentation-models-pytorch>=0.3.3
albumentations>=1.3.0

# WSI Processing
openslide-python>=1.3.0
opencv-python>=4.8.0
scikit-image>=0.21.0
Pillow>=10.0.0

# Experiment Tracking
wandb>=0.15.0

# Metrics & Visualization
torchmetrics>=1.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Utils
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
pyyaml>=6.0
```

## Key Implementation Details

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
- **Class Imbalance**: Epidermis typically occupies small portion of WSI. Dice loss helps mitigate this.
- **Patch Selection**: Consider importance sampling in future iterations.
- **Augmentation**: Starting conservative with horizontal flips only. Can expand if needed.
- **Multi-GPU**: Ready for DDP implementation if needed.

## Future Enhancements

1. **Advanced Augmentations**: Elastic deformations, rotations (if appropriate)
2. **Ensemble Methods**: Combine predictions from multiple models
3. **Active Learning**: Focus on hard-to-segment regions
4. **Foundation Models**: Explore SAM or other foundation models for medical imaging
5. **3D Context**: If z-stack data becomes available

Last Updated: 2025-01-17