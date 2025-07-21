"""
DeepLabV3+ model implementation using segmentation_models_pytorch.

Implements DeepLabV3+ with various encoder backbones for epidermis segmentation.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict, Optional


class DeepLabV3PlusModel(nn.Module):
    """DeepLabV3+ model wrapper for epidermis segmentation."""
    
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        encoder_depth: int = 5,
        encoder_output_stride: int = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36),
        upsampling: int = 4,
        aux_params: Optional[Dict] = None
    ):
        """
        Initialize DeepLabV3+ model.
        
        Args:
            encoder_name: Name of encoder ('resnet50', 'efficientnet-b3', etc.)
            encoder_weights: Pretrained weights to use
            in_channels: Number of input channels
            classes: Number of output classes (1 for binary)
            encoder_depth: Depth of encoder
            encoder_output_stride: Output stride of encoder
            decoder_channels: Number of channels in decoder
            decoder_atrous_rates: Atrous rates for ASPP module
            upsampling: Upsampling factor in decoder
            aux_params: Auxiliary output parameters (for deep supervision)
        """
        super().__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            encoder_depth=encoder_depth,
            encoder_output_stride=encoder_output_stride,
            decoder_channels=decoder_channels,
            decoder_atrous_rates=decoder_atrous_rates,
            upsampling=upsampling,
            aux_params=aux_params
        )
        
        # Store encoder for access
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.segmentation_head = self.model.segmentation_head
        
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def get_params_groups(self, lr: float):
        """
        Get parameter groups for differential learning rates.
        
        Args:
            lr: Base learning rate
            
        Returns:
            List of parameter groups
        """
        # Encoder with lower learning rate
        encoder_params = []
        decoder_params = []
        
        for name, param in self.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        
        return [
            {'params': encoder_params, 'lr': lr * 0.1},  # 10x lower for encoder
            {'params': decoder_params, 'lr': lr}
        ]


def create_deeplabv3plus_models(config: Optional[Dict] = None) -> Dict[str, nn.Module]:
    """
    Create DeepLabV3+ models with different encoders.
    
    Args:
        config: Model configuration
        
    Returns:
        Dictionary of models with different encoders
    """
    if config is None:
        config = {}
    
    models = {}
    
    # Default configuration
    default_config = {
        'encoder_weights': 'imagenet',
        'in_channels': 3,
        'classes': 1,  # Binary segmentation
        'encoder_depth': 5,
        'encoder_output_stride': 16,
        'decoder_channels': 256,
        'decoder_atrous_rates': (12, 24, 36),
        'upsampling': 4
    }
    
    # Extract encoders list before updating config
    encoders = config.get('encoders', ['resnet50', 'efficientnet-b3'])
    
    # Valid parameters for DeepLabV3+
    valid_params = {
        'encoder_weights', 'in_channels', 'classes', 'encoder_depth',
        'encoder_output_stride', 'decoder_channels', 'decoder_atrous_rates',
        'upsampling', 'aux_params'
    }
    
    # Filter and process config parameters
    model_params = {}
    for k, v in config.items():
        if k == 'encoders':
            continue
        elif k in valid_params:
            # Special handling for decoder_channels
            if k == 'decoder_channels' and isinstance(v, list):
                # Use the first value if it's a list (from U-Net config)
                model_params[k] = v[0]
            else:
                model_params[k] = v
        # Silently ignore invalid parameters like decoder_use_batchnorm
    
    default_config.update(model_params)
    
    # Create models with different encoders
    for encoder_name in encoders:
        model_config = default_config.copy()
        model_config['encoder_name'] = encoder_name
        
        models[f'deeplabv3plus_{encoder_name}'] = DeepLabV3PlusModel(**model_config)
    
    return models


def test_deeplabv3plus():
    """Test DeepLabV3+ model creation and forward pass."""
    import time
    
    # Test with different encoders
    encoders = ['resnet50', 'efficientnet-b3']
    
    for encoder in encoders:
        print(f"\nTesting DeepLabV3+ with {encoder}:")
        print("-" * 50)
        
        # Create model
        models = create_deeplabv3plus_models({'encoders': [encoder]})
        model = models[f'deeplabv3plus_{encoder}']
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 384, 384).to(device)
        
        # Measure inference time
        with torch.no_grad():
            # Warmup
            _ = model(input_tensor)
            
            # Time measurement
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            output = model(input_tensor)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
        
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB" if torch.cuda.is_available() else "N/A")


if __name__ == "__main__":
    test_deeplabv3plus()