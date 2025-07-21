#!/usr/bin/env python
"""
Test script to verify training pipeline works without errors.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from src.datasets.epidermis_dataset import EpidermisDataModule
from src.models.unet_smp import create_unet_models
from src.training.losses import get_loss_function
from src.training.metrics_simple import SegmentationMetrics, DiceMetric


def main():
    print("=== Testing Epidermis Segmentation Training Pipeline ===")
    
    # Test 1: Data loading
    print("\n1. Testing data loading...")
    try:
        data_module = EpidermisDataModule(
            patch_root='patches',
            split_file='patches/data_splits.json',
            batch_size=4,  # Small batch for testing
            num_workers=2,
            patch_size=384,
            use_weighted_sampling=True,
            oversample_factor=None  # Auto-calculate
        )
        data_module.setup()
        
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        print(f"✓ Train samples: {len(data_module.train_dataset)}")
        print(f"✓ Val samples: {len(data_module.val_dataset)}")
        print(f"✓ Class distribution: {data_module.train_dataset.class_distribution}")
        
        # Get one batch
        batch = next(iter(train_loader))
        print(f"✓ Batch shapes - Image: {batch['image'].shape}, Mask: {batch['mask'].shape}")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False
    
    # Test 2: Model creation
    print("\n2. Testing model creation...")
    try:
        # Test both encoders
        for encoder in ['resnet50', 'efficientnet-b3']:
            models = create_unet_models({'encoders': [encoder]})
            model = models[f'unet_{encoder}']
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            print(f"✓ U-Net with {encoder} created and moved to {device}")
        
        # Use ResNet50 for remaining tests
        models = create_unet_models({'encoders': ['resnet50']})
        model = models['unet_resnet50']
        model = model.to(device)
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    # Test 3: Loss function
    print("\n3. Testing loss function...")
    try:
        criterion = get_loss_function('dice', smooth=1e-6, apply_sigmoid=True)
        print("✓ Loss function created")
    except Exception as e:
        print(f"✗ Loss function creation failed: {e}")
        return False
    
    # Test 4: Metrics
    print("\n4. Testing metrics...")
    try:
        metrics = SegmentationMetrics(device=str(device))
        dice_metric = DiceMetric()
        print("✓ Metrics created")
    except Exception as e:
        print(f"✗ Metrics creation failed: {e}")
        return False
    
    # Test 5: Forward pass
    print("\n5. Testing forward pass...")
    try:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        model.train()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        print(f"✓ Forward pass successful - Loss: {loss.item():.4f}")
        print(f"✓ Output shape: {outputs.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Test 6: Metrics update
    print("\n6. Testing metrics update...")
    try:
        metrics.update(outputs, masks)
        metric_values = metrics.compute()
        print(f"✓ Metrics computed: {metric_values}")
    except Exception as e:
        print(f"✗ Metrics update failed: {e}")
        return False
    
    # Test 7: Backward pass
    print("\n7. Testing backward pass...")
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("✓ Backward pass successful")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        return False
    
    print("\n=== All tests passed! Training pipeline is ready. ===")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)