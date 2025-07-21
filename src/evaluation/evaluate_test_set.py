"""
Evaluation script specifically for TEST SET with epidermis-only Dice score.

This script:
1. Uses ONLY the test dataset split
2. Calculates epidermis-only Dice score (ignoring background)
3. Provides clear final metric output
4. Handles checkpoint loading robustly
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Optional
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.epidermis_dataset import EpidermisDataModule
from models.unet_smp import create_unet_models
from models.deeplabv3plus_smp import create_deeplabv3plus_models
from training.metrics_simple import SimpleSegmentationMetrics


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'test_evaluation.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_model_from_checkpoint(
    architecture: str,
    encoder: str,
    checkpoint_path: str,
    device: torch.device,
    logger: logging.Logger
) -> nn.Module:
    """
    Robustly load model from checkpoint.
    
    Args:
        architecture: 'unet' or 'deeplabv3plus'
        encoder: 'resnet50' or 'efficientnet-b3'
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        logger: Logger instance
        
    Returns:
        Loaded model in eval mode
    """
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Create model
    if architecture == 'unet':
        models = create_unet_models({'encoders': [encoder]})
        model = models[f'unet_{encoder}']
    elif architecture == 'deeplabv3plus':
        models = create_deeplabv3plus_models({'encoders': [encoder]})
        model = models[f'deeplabv3plus_{encoder}']
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Best val Dice from checkpoint: {checkpoint.get('best_val_dice', 'unknown')}")
    else:
        # Try loading directly (old format)
        model.load_state_dict(checkpoint)
        logger.warning("Loaded checkpoint in old format (no metadata)")
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    return model


def calculate_epidermis_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Calculate Dice score for epidermis class only.
    
    Args:
        pred: Predicted logits (B, 1, H, W)
        target: Ground truth masks (B, 1, H, W) with values 0 or 1
        threshold: Threshold for binary prediction
        
    Returns:
        Dice score for epidermis pixels only
    """
    # Apply sigmoid and threshold
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # Calculate intersection and union for EPIDERMIS (value=1) only
    tp = (pred_flat * target_flat).sum()  # True Positives
    fp = (pred_flat * (1 - target_flat)).sum()  # False Positives
    fn = ((1 - pred_flat) * target_flat).sum()  # False Negatives
    
    # Dice coefficient
    smooth = 1e-8
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    
    return dice.item()


def evaluate_on_test_set(
    model: nn.Module,
    test_loader,
    device: torch.device,
    output_dir: str,
    logger: logging.Logger,
    save_predictions: bool = True
) -> Dict:
    """
    Evaluate model on test set with detailed metrics.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        output_dir: Directory to save results
        logger: Logger instance
        save_predictions: Whether to save per-patch predictions
        
    Returns:
        Dictionary of results
    """
    model.eval()
    
    # Initialize metrics
    metrics = SimpleSegmentationMetrics(device=str(device))
    
    # Storage for per-sample results
    all_dice_scores = []
    all_predictions = []
    
    # Progress tracking
    total_patches = len(test_loader.dataset)
    logger.info(f"Evaluating on {total_patches} test patches...")
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating TEST set')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            batch_size = images.shape[0]
            
            # Forward pass
            outputs = model(images)
            
            # Update global metrics
            metrics.update(outputs, masks, apply_sigmoid=True)
            
            # Calculate per-sample Dice scores
            for i in range(batch_size):
                dice_score = calculate_epidermis_dice(
                    outputs[i:i+1],
                    masks[i:i+1]
                )
                all_dice_scores.append(dice_score)
                
                # Store prediction info
                if save_predictions:
                    pred_sigmoid = torch.sigmoid(outputs[i])
                    pred_binary = (pred_sigmoid > 0.5).float()
                    
                    # Determine has_epidermis from epidermis_ratio or mask
                    if 'epidermis_ratio' in batch:
                        epidermis_ratio = batch['epidermis_ratio'][i]
                        # Handle both tensor and float
                        if hasattr(epidermis_ratio, 'item'):
                            epidermis_ratio = epidermis_ratio.item()
                        has_epidermis = epidermis_ratio > 0.01
                    else:
                        # Calculate from mask if epidermis_ratio not available
                        epidermis_ratio = masks[i].mean().item()
                        has_epidermis = epidermis_ratio > 0.01
                    
                    pred_info = {
                        'patch_id': batch['patch_id'][i],
                        'dataset': batch['dataset'][i],
                        'has_epidermis': has_epidermis,
                        'epidermis_ratio_true': masks[i].mean().item(),
                        'epidermis_ratio_pred': pred_binary.mean().item(),
                        'dice_score': dice_score
                    }
                    all_predictions.append(pred_info)
            
            # Update progress bar
            current_dice = np.mean(all_dice_scores)
            pbar.set_postfix({'Dice': f'{current_dice:.4f}'})
    
    # Compute final metrics
    final_metrics = metrics.compute()
    
    # Calculate statistics
    dice_scores = np.array(all_dice_scores)
    final_results = {
        # Primary metric
        'test_dice_epidermis': float(np.mean(dice_scores)),
        'test_dice_std': float(np.std(dice_scores)),
        'test_dice_median': float(np.median(dice_scores)),
        
        # Other metrics from SimpleSegmentationMetrics
        'test_iou': final_metrics['iou'],
        'test_precision': final_metrics['precision'],
        'test_recall': final_metrics['recall'],
        'test_specificity': final_metrics['specificity'],
        'test_accuracy': final_metrics['accuracy'],
        
        # Sample counts
        'num_test_patches': total_patches,
        'num_patches_with_epidermis': sum(1 for p in all_predictions if p['has_epidermis']) if all_predictions else 0,
        'num_patches_without_epidermis': sum(1 for p in all_predictions if not p['has_epidermis']) if all_predictions else 0,
        
        # Per-dataset breakdown
        'per_dataset_dice': {}
    }
    
    # Calculate per-dataset metrics
    if save_predictions:
        df = pd.DataFrame(all_predictions)
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            dataset_dice = dataset_df['dice_score'].mean()
            final_results['per_dataset_dice'][dataset] = {
                'dice': float(dataset_dice),
                'num_patches': len(dataset_df)
            }
    
    # Save results
    results_file = os.path.join(output_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    # Save predictions CSV
    if save_predictions:
        pred_file = os.path.join(output_dir, 'test_predictions.csv')
        df.to_csv(pred_file, index=False)
        logger.info(f"Predictions saved to: {pred_file}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"Total test patches: {total_patches}")
    logger.info(f"Patches with epidermis: {final_results['num_patches_with_epidermis']}")
    logger.info(f"Patches without epidermis: {final_results['num_patches_without_epidermis']}")
    logger.info("-"*60)
    logger.info(f"EPIDERMIS-ONLY DICE SCORE: {final_results['test_dice_epidermis']:.4f} ± {final_results['test_dice_std']:.4f}")
    logger.info(f"Median Dice: {final_results['test_dice_median']:.4f}")
    logger.info(f"IoU: {final_results['test_iou']:.4f}")
    logger.info(f"Precision: {final_results['test_precision']:.4f}")
    logger.info(f"Recall: {final_results['test_recall']:.4f}")
    logger.info("-"*60)
    
    for dataset, stats in final_results['per_dataset_dice'].items():
        logger.info(f"{dataset}: Dice={stats['dice']:.4f} (n={stats['num_patches']})")
    
    logger.info("="*60)
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on TEST set")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (best.pth)'
    )
    parser.add_argument(
        '--architecture',
        type=str,
        required=True,
        choices=['unet', 'deeplabv3plus'],
        help='Model architecture'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        required=True,
        choices=['resnet50', 'efficientnet-b3'],
        help='Encoder backbone'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of data loading workers'
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.output_dir)
    logger.info(f"Evaluating {args.architecture.upper()} + {args.encoder}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model_from_checkpoint(
        args.architecture,
        args.encoder,
        args.checkpoint,
        device,
        logger
    )
    
    # Setup TEST data only
    logger.info("Loading TEST dataset...")
    data_module = EpidermisDataModule(
        patch_root='patches',
        split_file='patches/data_splits.json',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=384,
        use_weighted_sampling=False  # No sampling for evaluation
    )
    data_module.setup()
    
    # Get TEST loader specifically
    test_loader = data_module.test_dataloader()
    test_size = len(data_module.test_dataset)
    logger.info(f"Test dataset size: {test_size} patches")
    
    # Verify we're using test split
    if hasattr(data_module.test_dataset, 'split'):
        assert data_module.test_dataset.split == 'test', "Not using test split!"
    
    # Run evaluation
    results = evaluate_on_test_set(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=args.output_dir,
        logger=logger,
        save_predictions=True
    )
    
    logger.info("Evaluation complete!")
    
    # Return dice score for scripts
    return results['test_dice_epidermis']


if __name__ == "__main__":
    dice_score = main()
    # Exit with dice score * 100 as exit code (for scripts)
    exit(min(int(dice_score * 100), 99))