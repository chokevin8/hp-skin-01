"""
Enhanced loss functions for epidermis segmentation.

Provides alternatives to pure Dice loss if concerned about false positives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class CombinedDiceBCELoss(nn.Module):
    """Combined Dice and BCE loss for better handling of imbalanced data."""
    
    def __init__(
        self,
        dice_weight: float = 0.8,
        bce_weight: float = 0.2,
        smooth: float = 1e-6,
        pos_weight: Optional[float] = None
    ):
        """
        Initialize combined loss.
        
        Args:
            dice_weight: Weight for Dice loss component
            bce_weight: Weight for BCE loss component
            smooth: Smoothing factor for Dice
            pos_weight: Positive class weight for BCE (if None, no weighting)
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.pos_weight = pos_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss."""
        # Apply sigmoid if needed
        pred_sigmoid = torch.sigmoid(pred)
        
        # Dice loss
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        
        # BCE loss
        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor([self.pos_weight], device=pred.device)
            bce_loss = F.binary_cross_entropy_with_logits(
                pred, target, pos_weight=pos_weight_tensor
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(pred, target)
        
        # Combined loss
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        
        return total_loss


class WeightedDiceLoss(nn.Module):
    """Dice loss with different weights for patches with/without epidermis."""
    
    def __init__(
        self,
        smooth: float = 1e-6,
        epidermis_weight: float = 1.0,
        no_epidermis_weight: float = 0.5
    ):
        """
        Initialize weighted Dice loss.
        
        Args:
            smooth: Smoothing factor
            epidermis_weight: Weight for patches with epidermis
            no_epidermis_weight: Weight for patches without epidermis
        """
        super().__init__()
        self.smooth = smooth
        self.epidermis_weight = epidermis_weight
        self.no_epidermis_weight = no_epidermis_weight
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        has_epidermis: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate weighted Dice loss.
        
        Args:
            pred: Predictions (logits)
            target: Ground truth
            has_epidermis: Boolean tensor indicating if patch has epidermis
        """
        batch_size = pred.shape[0]
        pred_sigmoid = torch.sigmoid(pred)
        total_loss = 0.0
        
        for i in range(batch_size):
            # Calculate Dice for this sample
            pred_flat = pred_sigmoid[i].view(-1)
            target_flat = target[i].view(-1)
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1 - dice
            
            # Apply weight based on whether patch has epidermis
            if has_epidermis is not None and not has_epidermis[i]:
                weight = self.no_epidermis_weight
            else:
                weight = self.epidermis_weight
            
            total_loss += weight * dice_loss
        
        return total_loss / batch_size


class FocalDiceLoss(nn.Module):
    """Focal variant of Dice loss to focus on hard examples."""
    
    def __init__(self, smooth: float = 1e-6, gamma: float = 2.0):
        """
        Initialize focal Dice loss.
        
        Args:
            smooth: Smoothing factor
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.smooth = smooth
        self.gamma = gamma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate focal Dice loss."""
        pred_sigmoid = torch.sigmoid(pred)
        
        # Calculate Dice
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Apply focal weighting
        focal_weight = (1 - dice) ** self.gamma
        focal_dice_loss = focal_weight * (1 - dice)
        
        return focal_dice_loss


class TverskyLoss(nn.Module):
    """
    Tversky loss with adjustable false positive/negative penalties.
    
    Useful when you want to control the trade-off between precision and recall.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1e-6
    ):
        """
        Initialize Tversky loss.
        
        Args:
            alpha: Weight for false positives (higher = more FP penalty)
            beta: Weight for false negatives (higher = more FN penalty)
            smooth: Smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Tversky loss."""
        pred_sigmoid = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        # Calculate TP, FP, FN
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky


def get_enhanced_loss_function(
    loss_type: str = 'dice',
    **kwargs
) -> nn.Module:
    """
    Get enhanced loss function by name.
    
    Args:
        loss_type: Type of loss ('dice', 'dice_bce', 'weighted_dice', 'focal_dice', 'tversky')
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function module
    """
    if loss_type == 'dice':
        # Use original Dice loss
        from .losses import DiceLoss
        return DiceLoss(**kwargs)
    elif loss_type == 'dice_bce':
        return CombinedDiceBCELoss(**kwargs)
    elif loss_type == 'weighted_dice':
        return WeightedDiceLoss(**kwargs)
    elif loss_type == 'focal_dice':
        return FocalDiceLoss(**kwargs)
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Example configurations for different scenarios
LOSS_CONFIGS = {
    'balanced': {
        'loss_type': 'dice_bce',
        'dice_weight': 0.8,
        'bce_weight': 0.2,
        'pos_weight': 2.0  # Since epidermis is less common
    },
    'conservative': {
        'loss_type': 'tversky',
        'alpha': 0.7,  # Higher penalty for false positives
        'beta': 0.3    # Lower penalty for false negatives
    },
    'aggressive': {
        'loss_type': 'tversky',
        'alpha': 0.3,  # Lower penalty for false positives
        'beta': 0.7    # Higher penalty for false negatives
    },
    'focal': {
        'loss_type': 'focal_dice',
        'gamma': 2.0   # Focus on hard examples
    },
    'weighted': {
        'loss_type': 'weighted_dice',
        'epidermis_weight': 1.0,
        'no_epidermis_weight': 0.3  # Less emphasis on no-epidermis patches
    }
}