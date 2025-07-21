"""
U-Net models using segmentation_models_pytorch.

Implements U-Net with different pretrained encoders for epidermis segmentation.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict, Optional, List
import logging


class UNetModel(nn.Module):
    """U-Net wrapper for epidermis segmentation."""
    
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        decoder_use_batchnorm: bool = True,
        decoder_attention_type: Optional[str] = None
    ):
        """
        Initialize U-Net model.
        
        Args:
            encoder_name: Name of encoder ('resnet50', 'efficientnet-b3', etc.)
            encoder_weights: Pretrained weights to use
            in_channels: Number of input channels
            classes: Number of output classes (1 for binary)
            activation: Final activation function
            decoder_channels: Number of channels in decoder blocks
            decoder_use_batchnorm: Whether to use batch normalization
            decoder_attention_type: Attention type (None, 'scse')
        """
        super().__init__()
        
        self.encoder_name = encoder_name
        self.logger = logging.getLogger(__name__)
        
        # Create U-Net model
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            decoder_channels=decoder_channels,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_attention_type=decoder_attention_type
        )
        
        self.logger.info(
            f"Initialized U-Net with {encoder_name} encoder, "
            f"input channels: {in_channels}, output classes: {classes}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def get_params_groups(self, lr: float = 1e-4) -> List[Dict]:
        """
        Get parameter groups for differential learning rates.
        
        Args:
            lr: Base learning rate
            
        Returns:
            List of parameter groups
        """
        # Encoder parameters (pretrained) - lower learning rate
        encoder_params = []
        for name, param in self.model.encoder.named_parameters():
            encoder_params.append(param)
        
        # Decoder parameters (from scratch) - higher learning rate
        decoder_params = []
        for name, param in self.model.decoder.named_parameters():
            decoder_params.append(param)
        
        # Segmentation head parameters
        head_params = []
        for name, param in self.model.segmentation_head.named_parameters():
            head_params.append(param)
        
        param_groups = [
            {'params': encoder_params, 'lr': lr * 0.1},  # 10x lower LR for encoder
            {'params': decoder_params, 'lr': lr},
            {'params': head_params, 'lr': lr}
        ]
        
        return param_groups


def create_unet_models(config: Dict) -> Dict[str, UNetModel]:
    """
    Create multiple U-Net models with different encoders.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Dictionary of models
    """
    models = {}
    
    # Default configuration
    default_config = {
        'in_channels': 3,
        'classes': 1,
        'activation': None,  # Will use sigmoid in loss
        'decoder_channels': [256, 128, 64, 32, 16],
        'decoder_use_batchnorm': True,
        'decoder_attention_type': None
    }
    
    # Extract encoders list before updating config
    encoders = config.get('encoders', ['resnet50', 'efficientnet-b3'])
    
    # Update with provided config (excluding 'encoders' key)
    model_params = {k: v for k, v in config.items() if k != 'encoders'}
    default_config.update(model_params)
    
    # Create models with different encoders
    for encoder_name in encoders:
        model_config = default_config.copy()
        model_config['encoder_name'] = encoder_name
        
        models[f'unet_{encoder_name}'] = UNetModel(**model_config)
    
    return models


class UNetEnsemble(nn.Module):
    """Ensemble of U-Net models for improved performance."""
    
    def __init__(
        self,
        models: List[UNetModel],
        mode: str = 'mean'
    ):
        """
        Initialize ensemble.
        
        Args:
            models: List of U-Net models
            mode: Ensemble mode ('mean', 'weighted')
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.mode = mode
        
        if mode == 'weighted':
            # Learnable weights for each model
            self.weights = nn.Parameter(
                torch.ones(len(models)) / len(models)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        outputs = []
        
        for model in self.models:
            outputs.append(model(x))
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=0)
        
        if self.mode == 'mean':
            return outputs.mean(dim=0)
        elif self.mode == 'weighted':
            weights = torch.softmax(self.weights, dim=0)
            weights = weights.view(-1, 1, 1, 1, 1)
            return (outputs * weights).sum(dim=0)
        else:
            raise ValueError(f"Unknown ensemble mode: {self.mode}")


def get_model_info(encoder_name: str) -> Dict:
    """
    Get information about a specific encoder.
    
    Args:
        encoder_name: Name of the encoder
        
    Returns:
        Dictionary with encoder information
    """
    try:
        # Get encoder module
        encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights='imagenet'
        )
        
        # Get output channels for each stage
        out_channels = encoder.out_channels
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(
            p.numel() for p in encoder.parameters() if p.requires_grad
        )
        
        info = {
            'encoder_name': encoder_name,
            'out_channels': out_channels,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'pretrained': True
        }
        
        return info
        
    except Exception as e:
        logging.error(f"Error getting encoder info for {encoder_name}: {e}")
        return {}


if __name__ == "__main__":
    # Test model creation
    config = {
        'encoders': ['resnet50', 'efficientnet-b3'],
        'in_channels': 3,
        'classes': 1
    }
    
    models = create_unet_models(config)
    
    for name, model in models.items():
        print(f"\n{name}:")
        info = get_model_info(model.encoder_name)
        print(f"  Encoder channels: {info.get('out_channels', 'N/A')}")
        print(f"  Total parameters: {info.get('total_params', 0):,}")
        
        # Test forward pass
        x = torch.randn(2, 3, 384, 384)
        y = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")