"""
Simple metrics implementation for epidermis segmentation.

Uses manual calculation for metrics to avoid API compatibility issues.
"""

import torch
import numpy as np
from typing import Dict, List, Optional


class SimpleSegmentationMetrics:
    """Simple metrics for segmentation evaluation without torchmetrics dependency."""
    
    def __init__(
        self,
        threshold: float = 0.5,
        device: str = 'cuda'
    ):
        """
        Initialize metrics.
        
        Args:
            threshold: Threshold for binary prediction
            device: Device to place metrics on
        """
        self.threshold = threshold
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.tn = 0  # True Negatives
        self.fn = 0  # False Negatives
        self.total = 0
    
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        apply_sigmoid: bool = True
    ):
        """
        Update metrics with predictions and targets.
        
        Args:
            preds: Predicted masks (B, 1, H, W) - logits
            targets: Ground truth masks (B, 1, H, W) - binary (0 or 1)
            apply_sigmoid: Whether to apply sigmoid to predictions
        """
        if apply_sigmoid:
            preds = torch.sigmoid(preds)
        
        # Convert to binary
        preds_binary = (preds > self.threshold).float()
        targets = targets.float()
        
        # Calculate confusion matrix elements
        self.tp += ((preds_binary == 1) & (targets == 1)).sum().item()
        self.fp += ((preds_binary == 1) & (targets == 0)).sum().item()
        self.tn += ((preds_binary == 0) & (targets == 0)).sum().item()
        self.fn += ((preds_binary == 0) & (targets == 1)).sum().item()
        self.total += targets.numel()
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics - focusing on epidermis (foreground) class.
        
        Returns:
            Dictionary of metric values
        """
        eps = 1e-8
        
        # Basic metrics
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        specificity = self.tn / (self.tn + self.fp + eps)
        accuracy = (self.tp + self.tn) / (self.total + eps)
        
        # Dice coefficient
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)
        
        # IoU (Jaccard)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        
        return {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'accuracy': accuracy
        }


class DiceMetric:
    """
    Custom Dice metric implementation for more control.
    
    Useful for per-sample or per-dataset evaluation.
    """
    
    def __init__(self, smooth: float = 1e-6):
        """
        Initialize Dice metric.
        
        Args:
            smooth: Smoothing factor
        """
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.dice_scores = []
        self.sample_info = []
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_info: Optional[Dict] = None
    ):
        """
        Update metric with a batch.
        
        Args:
            pred: Predicted masks (B, 1, H, W)
            target: Ground truth masks (B, 1, H, W)
            sample_info: Optional sample information
        """
        pred = torch.sigmoid(pred)
        batch_size = pred.shape[0]
        
        for i in range(batch_size):
            # Calculate Dice for each sample
            pred_i = (pred[i] > 0.5).float().view(-1)
            target_i = target[i].view(-1)
            
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            
            self.dice_scores.append(dice.cpu().item())
            
            if sample_info:
                self.sample_info.append({
                    'patch_id': sample_info['patch_id'][i],
                    'dataset': sample_info['dataset'][i],
                    'dice': dice.cpu().item()
                })
    
    def compute(self) -> Dict[str, float]:
        """
        Compute overall and per-dataset metrics.
        
        Returns:
            Dictionary with overall and per-dataset Dice scores
        """
        if not self.dice_scores:
            return {'dice': 0.0}
        
        results = {
            'dice': np.mean(self.dice_scores),
            'dice_std': np.std(self.dice_scores)
        }
        
        # Per-dataset metrics if available
        if self.sample_info:
            datasets = set(info['dataset'] for info in self.sample_info)
            for dataset in datasets:
                dataset_scores = [
                    info['dice'] for info in self.sample_info 
                    if info['dataset'] == dataset
                ]
                results[f'dice_{dataset}'] = np.mean(dataset_scores)
                results[f'dice_{dataset}_std'] = np.std(dataset_scores)
        
        return results


# Use simple metrics as default
SegmentationMetrics = SimpleSegmentationMetrics