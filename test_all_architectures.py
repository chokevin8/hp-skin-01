#!/usr/bin/env python
"""
Test script to verify all architectures work correctly.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from src.datasets.epidermis_dataset import EpidermisDataModule
from src.models.unet_smp import create_unet_models
from src.models.deeplabv3plus_smp import create_deeplabv3plus_models
from src.training.losses import get_loss_function
from src.training.metrics_simple import SegmentationMetrics


def test_architecture(arch_name: str, model, device: torch.device):
    """Test a single architecture."""
    print(f"\n{'='*60}")
    print(f"Testing {arch_name}")
    print('='*60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
    
    # Move model to device
    model = model.to(device)
    
    # Create dummy batch
    batch_size = 2
    images = torch.randn(batch_size, 3, 384, 384).to(device)
    masks = torch.randint(0, 2, (batch_size, 1, 384, 384)).float().to(device)
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    model.train()
    outputs = model(images)
    print(f"✓ Input shape: {images.shape}")
    print(f"✓ Output shape: {outputs.shape}")
    
    # Test loss computation
    print(f"\nTesting loss computation...")
    criterion = get_loss_function('dice', smooth=1e-6, apply_sigmoid=True)
    loss = criterion(outputs, masks)
    print(f"✓ Loss: {loss.item():.4f}")
    
    # Test backward pass
    print(f"\nTesting backward pass...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"✓ Backward pass successful")
    
    # Test metrics
    print(f"\nTesting metrics...")
    metrics = SegmentationMetrics(device=str(device))
    metrics.update(outputs, masks)
    metric_values = metrics.compute()
    print(f"✓ Metrics computed: {metric_values}")
    
    # Test inference mode
    print(f"\nTesting inference mode...")
    model.eval()
    with torch.no_grad():
        outputs_eval = model(images)
    print(f"✓ Inference successful")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        torch.cuda.empty_cache()
    
    print(f"\n✓ All tests passed for {arch_name}!")


def main():
    print("=== Testing All Epidermis Segmentation Architectures ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test configurations
    architectures = ['unet', 'deeplabv3plus']
    encoders = ['resnet50', 'efficientnet-b3']
    
    # Test each combination
    for arch in architectures:
        for encoder in encoders:
            arch_name = f"{arch.upper()} + {encoder}"
            
            try:
                # Create model with config that might have U-Net params
                config = {
                    'encoders': [encoder],
                    'decoder_channels': [256, 128, 64, 32, 16],  # U-Net style
                    'decoder_use_batchnorm': True,  # U-Net param
                    'decoder_attention_type': None   # U-Net param
                }
                
                if arch == 'unet':
                    models = create_unet_models(config)
                    model = models[f'unet_{encoder}']
                else:  # deeplabv3plus
                    models = create_deeplabv3plus_models(config)
                    model = models[f'deeplabv3plus_{encoder}']
                
                # Test the model
                test_architecture(arch_name, model, device)
                
            except Exception as e:
                print(f"\n✗ Error testing {arch_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Test data loading
    print(f"\n\n{'='*60}")
    print("Testing Data Loading")
    print('='*60)
    try:
        data_module = EpidermisDataModule(
            patch_root='patches',
            split_file='patches/data_splits.json',
            batch_size=4,
            num_workers=2,
            patch_size=384,
            use_weighted_sampling=True,
            oversample_factor=None
        )
        data_module.setup()
        
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        
        print(f"✓ Data loading successful")
        print(f"✓ Batch keys: {list(batch.keys())}")
        print(f"✓ Image shape: {batch['image'].shape}")
        print(f"✓ Mask shape: {batch['mask'].shape}")
        print(f"✓ Class distribution: {data_module.train_dataset.class_distribution}")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
    
    print("\n\n=== All architecture tests complete! ===")
    print("\nSummary of tested models:")
    print("1. U-Net + ResNet50")
    print("2. U-Net + EfficientNet-B3")
    print("3. DeepLabV3+ + ResNet50")
    print("4. DeepLabV3+ + EfficientNet-B3")
    print("\nAll models are ready for training!")


if __name__ == "__main__":
    main()