"""
Metrics for epidermis segmentation evaluation.

Implements Dice, IoU, and other metrics for binary segmentation.
"""

import torch
import torchmetrics
from typing import Dict, List, Optional
import numpy as np


class SegmentationMetrics:
    """Collection of metrics for segmentation evaluation."""
    
    def __init__(
        self,
        num_classes: int = 1,
        threshold: float = 0.5,
        device: str = 'cuda'
    ):
        """
        Initialize metrics.
        
        Args:
            num_classes: Number of classes (1 for binary)
            threshold: Threshold for binary prediction
            device: Device to place metrics on
        """
        self.num_classes = num_classes
        self.threshold = threshold
        self.device = device
        
        # Initialize metrics
        self.metrics = {
            'dice': torchmetrics.Dice(
                num_classes=2,  # Binary: background + foreground
                average='macro',
                threshold=threshold
            ).to(device),
            
            'iou': torchmetrics.JaccardIndex(
                task='binary',
                num_classes=2,
                average='macro',
                threshold=threshold
            ).to(device),
            
            'precision': torchmetrics.Precision(
                task='binary',
                num_classes=2,
                average='macro',
                threshold=threshold
            ).to(device),
            
            'recall': torchmetrics.Recall(
                task='binary',
                num_classes=2,
                average='macro',
                threshold=threshold
            ).to(device),
            
            'specificity': torchmetrics.Specificity(
                task='binary',
                num_classes=2,
                average='macro',
                threshold=threshold
            ).to(device),
            
            'accuracy': torchmetrics.Accuracy(
                task='binary',
                num_classes=2,
                average='macro',
                threshold=threshold
            ).to(device)
        }
    
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        apply_sigmoid: bool = True
    ):
        """
        Update metrics with predictions and targets.
        
        Args:
            preds: Predicted masks (B, 1, H, W)
            targets: Ground truth masks (B, 1, H, W)
            apply_sigmoid: Whether to apply sigmoid to predictions
        """
        if apply_sigmoid:
            preds = torch.sigmoid(preds)
        
        # Convert to binary format expected by metrics
        preds_binary = (preds > self.threshold).long()
        targets_binary = targets.long()
        
        # Update each metric
        for metric in self.metrics.values():
            metric.update(preds_binary, targets_binary)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric values
        """
        results = {}
        for name, metric in self.metrics.items():
            value = metric.compute()
            # Extract foreground class value for binary segmentation
            if hasattr(value, 'shape') and len(value.shape) > 0:
                results[name] = value[1].item()  # Foreground class
            else:
                results[name] = value.item()
        
        return results
    
    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()


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


class EpidermisRatioMetric:
    """
    Metric to track epidermis ratio predictions.
    
    Useful for understanding model behavior.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize metric.
        
        Args:
            threshold: Threshold for binary prediction
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.pred_ratios = []
        self.true_ratios = []
        self.errors = []
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        true_ratios: Optional[List[float]] = None
    ):
        """
        Update metric.
        
        Args:
            pred: Predicted masks
            target: Ground truth masks
            true_ratios: Known epidermis ratios
        """
        pred = torch.sigmoid(pred)
        batch_size = pred.shape[0]
        
        for i in range(batch_size):
            # Calculate predicted ratio
            pred_binary = (pred[i] > self.threshold).float()
            pred_ratio = pred_binary.mean().cpu().item()
            
            # Calculate true ratio
            true_ratio = target[i].float().mean().cpu().item()
            
            self.pred_ratios.append(pred_ratio)
            self.true_ratios.append(true_ratio)
            self.errors.append(abs(pred_ratio - true_ratio))
    
    def compute(self) -> Dict[str, float]:
        """Compute ratio metrics."""
        if not self.pred_ratios:
            return {}
        
        return {
            'mean_pred_ratio': np.mean(self.pred_ratios),
            'mean_true_ratio': np.mean(self.true_ratios),
            'mean_ratio_error': np.mean(self.errors),
            'std_ratio_error': np.std(self.errors)
        }