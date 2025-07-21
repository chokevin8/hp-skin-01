#!/usr/bin/env python
"""
Show valid parameters for each model architecture.
"""

import inspect
import segmentation_models_pytorch as smp
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from src.models.unet_smp import UNetModel
from src.models.deeplabv3plus_smp import DeepLabV3PlusModel


def show_parameters():
    """Show parameters accepted by each architecture."""
    print("=== Model Parameter Reference ===\n")
    
    # U-Net parameters
    print("U-Net (via UNetModel):")
    print("-" * 40)
    unet_sig = inspect.signature(UNetModel.__init__)
    for param_name, param in unet_sig.parameters.items():
        if param_name not in ['self', 'kwargs']:
            default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
            print(f"  {param_name}: {default}")
    
    # DeepLabV3+ parameters
    print("\n\nDeepLabV3+ (via DeepLabV3PlusModel):")
    print("-" * 40)
    dlv3_sig = inspect.signature(DeepLabV3PlusModel.__init__)
    for param_name, param in dlv3_sig.parameters.items():
        if param_name not in ['self', 'kwargs']:
            default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
            print(f"  {param_name}: {default}")
    
    # Show key differences
    print("\n\nKey Differences:")
    print("-" * 40)
    print("1. decoder_channels:")
    print("   - U-Net: List of ints [256, 128, 64, 32, 16]")
    print("   - DeepLabV3+: Single int (256)")
    print("\n2. U-Net only parameters:")
    print("   - decoder_use_batchnorm")
    print("   - decoder_attention_type")
    print("\n3. DeepLabV3+ only parameters:")
    print("   - encoder_output_stride")
    print("   - decoder_atrous_rates")
    print("   - upsampling")
    
    # Show what happens with shared config
    print("\n\nShared Config Handling:")
    print("-" * 40)
    print("When using the same config file:")
    print("- U-Net: Uses all its parameters, ignores DeepLabV3+ specific ones")
    print("- DeepLabV3+: Filters out U-Net specific parameters, converts decoder_channels")


if __name__ == "__main__":
    show_parameters()