"""
Loss functions for epidermis segmentation.

Implements Dice loss and combined losses for binary segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation - EPIDERMIS (foreground) class only.
    
    Loss = 1 - 2 * |X ∩ Y| / (|X| + |Y|)
    
    Note: This implementation correctly focuses on epidermis segmentation only.
    Background pixels are ignored in the loss calculation.
    """
    
    def __init__(
        self,
        smooth: float = 1e-6,
        apply_sigmoid: bool = True
    ):
        """
        Initialize Dice loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            apply_sigmoid: Whether to apply sigmoid to predictions
        """
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Dice loss.
        
        Args:
            pred: Predicted masks (B, 1, H, W)
            target: Ground truth masks (B, 1, H, W)
            
        Returns:
            Dice loss value
        """
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return loss (1 - Dice)
        return 1.0 - dice


class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss with better gradient properties.
    
    Uses squared terms in denominator for smoother gradients.
    """
    
    def __init__(
        self,
        smooth: float = 1e-6,
        apply_sigmoid: bool = True,
        reduction: str = 'mean'
    ):
        """
        Initialize Soft Dice loss.
        
        Args:
            smooth: Smoothing factor
            apply_sigmoid: Whether to apply sigmoid
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Soft Dice loss.
        
        Args:
            pred: Predicted masks (B, 1, H, W)
            target: Ground truth masks (B, 1, H, W)
            
        Returns:
            Soft Dice loss value
        """
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)
        
        # Calculate per-sample loss
        batch_size = pred.shape[0]
        losses = []
        
        for i in range(batch_size):
            pred_i = pred[i].view(-1)
            target_i = target[i].view(-1)
            
            # Soft Dice formula
            intersection = (pred_i * target_i).sum()
            pred_sum = (pred_i * pred_i).sum()
            target_sum = (target_i * target_i).sum()
            
            dice = (2.0 * intersection + self.smooth) / (
                pred_sum + target_sum + self.smooth
            )
            
            losses.append(1.0 - dice)
        
        losses = torch.stack(losses)
        
        # Apply reduction
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Useful for epidermis segmentation where most patches contain no epidermis.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (epidermis)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Focal Loss.
        
        Args:
            pred: Predicted masks (B, 1, H, W) - logits
            target: Ground truth masks (B, 1, H, W)
            
        Returns:
            Focal loss value
        """
        # Apply sigmoid to get probabilities
        prob = torch.sigmoid(pred)
        
        # Calculate binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Calculate p_t
        p_t = prob * target + (1 - prob) * (1 - target)
        
        # Calculate alpha_t
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Calculate focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Apply focal weighting
        focal_loss = focal_weight * bce
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FocalDiceLoss(nn.Module):
    """
    Focal Dice loss that focuses on hard samples.
    
    Applies focal weighting to Dice loss.
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        smooth: float = 1e-6,
        apply_sigmoid: bool = True
    ):
        """
        Initialize Focal Dice loss.
        
        Args:
            gamma: Focal parameter (higher = more focus on hard samples)
            smooth: Smoothing factor
            apply_sigmoid: Whether to apply sigmoid
        """
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid
        self.dice_loss = DiceLoss(smooth, apply_sigmoid=False)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Focal Dice loss.
        
        Args:
            pred: Predicted masks
            target: Ground truth masks
            
        Returns:
            Focal Dice loss value
        """
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)
        
        # Calculate Dice loss
        dice_loss = self.dice_loss(pred, target)
        
        # Apply focal weighting
        focal_weight = torch.pow(dice_loss, self.gamma)
        
        return focal_weight * dice_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function (Dice + BCE).
    
    Useful for better convergence in some cases.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1e-6,
        apply_sigmoid: bool = True
    ):
        """
        Initialize combined loss.
        
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            smooth: Smoothing factor for Dice
            apply_sigmoid: Whether to apply sigmoid
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        self.dice_loss = DiceLoss(smooth, apply_sigmoid)
        if apply_sigmoid:
            self.bce_loss = nn.BCEWithLogitsLoss()
        else:
            self.bce_loss = nn.BCELoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            pred: Predicted masks
            target: Ground truth masks
            
        Returns:
            Combined loss value
        """
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        return self.dice_weight * dice + self.bce_weight * bce


def get_loss_function(
    loss_name: str = 'dice',
    **kwargs
) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_name: Name of loss function
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function module
    """
    loss_functions = {
        'dice': DiceLoss,
        'soft_dice': SoftDiceLoss,
        'focal': FocalLoss,
        'focal_dice': FocalDiceLoss,
        'combined': CombinedLoss
    }
    
    if loss_name not in loss_functions:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Available: {list(loss_functions.keys())}"
        )
    
    return loss_functions[loss_name](**kwargs)