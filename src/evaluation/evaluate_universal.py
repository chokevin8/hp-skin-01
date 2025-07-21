"""
Universal evaluation script for epidermis segmentation models.

Supports U-Net and DeepLabV3+ architectures with various encoders.
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
import cv2
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.epidermis_dataset import EpidermisDataModule
from models.unet_smp import create_unet_models
from models.deeplabv3plus_smp import create_deeplabv3plus_models
from training.metrics_simple import SegmentationMetrics, DiceMetric


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'evaluation.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_model(architecture: str, encoder: str, checkpoint_path: str, device: torch.device):
    """
    Create and load model from checkpoint.
    
    Args:
        architecture: Model architecture ('unet' or 'deeplabv3plus')
        encoder: Encoder name
        checkpoint_path: Path to checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


class ModelEvaluator:
    """Evaluator for segmentation models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: str,
        threshold: float = 0.5
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to run evaluation on
            output_dir: Directory to save outputs
            threshold: Threshold for binary prediction
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.predictions_dir = os.path.join(output_dir, 'predictions')
        self.visualizations_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # Metrics
        self.metrics = SegmentationMetrics(device=device)
        self.dice_metric = DiceMetric()
        
    def evaluate(
        self,
        dataloader,
        save_predictions: bool = True,
        num_visualizations: int = 20
    ) -> Dict:
        """
        Evaluate model on dataset.
        
        Args:
            dataloader: DataLoader to evaluate on
            save_predictions: Whether to save predictions
            num_visualizations: Number of samples to visualize
            
        Returns:
            Dictionary of evaluation results
        """
        self.model.eval()
        self.metrics.reset()
        self.dice_metric.reset()
        
        all_predictions = []
        visualizations_saved = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Evaluating')
            
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Update metrics
                self.metrics.update(outputs, masks)
                self.dice_metric.update(
                    outputs, masks,
                    sample_info={
                        'patch_id': batch['patch_id'],
                        'dataset': batch['dataset']
                    }
                )
                
                # Process predictions
                predictions = torch.sigmoid(outputs)
                predictions_binary = (predictions > self.threshold).float()
                
                # Save predictions and visualizations
                for i in range(len(batch['patch_id'])):
                    patch_id = batch['patch_id'][i]
                    dataset = batch['dataset'][i]
                    
                    # Get epidermis_ratio and determine has_epidermis
                    epidermis_ratio = batch.get('epidermis_ratio', [None])[i]
                    if epidermis_ratio is not None:
                        has_epidermis = epidermis_ratio > 0.01
                    else:
                        # Calculate from mask if not available
                        epidermis_ratio = masks[i].mean().item()
                        has_epidermis = epidermis_ratio > 0.01
                    
                    # Calculate predicted epidermis ratio
                    pred_ratio = predictions_binary[i].mean().item()
                    
                    # Store prediction info
                    pred_info = {
                        'patch_id': patch_id,
                        'dataset': dataset,
                        'has_epidermis': has_epidermis,
                        'true_epidermis_ratio': epidermis_ratio,
                        'pred_epidermis_ratio': pred_ratio,
                        'dice_score': self._calculate_dice(
                            predictions_binary[i:i+1],
                            masks[i:i+1]
                        )
                    }
                    all_predictions.append(pred_info)
                    
                    # Save visualizations for first N samples
                    if visualizations_saved < num_visualizations:
                        self._save_visualization(
                            images[i].cpu(),
                            masks[i].cpu(),
                            predictions[i].cpu(),
                            predictions_binary[i].cpu(),
                            pred_info,
                            visualizations_saved
                        )
                        visualizations_saved += 1
                
                pbar.set_postfix({'samples': len(all_predictions)})
        
        # Compute final metrics
        metrics = self.metrics.compute()
        dice_metrics = self.dice_metric.compute()
        metrics.update(dice_metrics)
        
        # Add confusion matrix for has_epidermis classification
        true_has_epi = [p['has_epidermis'] for p in all_predictions]
        pred_has_epi = [p['pred_epidermis_ratio'] > 0.01 for p in all_predictions]
        
        cm = confusion_matrix(true_has_epi, pred_has_epi)
        metrics['has_epidermis_accuracy'] = (cm[0,0] + cm[1,1]) / cm.sum()
        metrics['has_epidermis_precision'] = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0
        metrics['has_epidermis_recall'] = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
        
        # Save results
        results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'num_samples': len(all_predictions),
            'predictions': all_predictions if save_predictions else None
        }
        
        # Save to JSON
        results_file = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save predictions CSV
        if save_predictions:
            pred_df = pd.DataFrame(all_predictions)
            pred_df.to_csv(
                os.path.join(self.output_dir, 'predictions.csv'),
                index=False
            )
        
        # Create summary plots
        self._create_summary_plots(all_predictions, metrics)
        
        return results
    
    def _calculate_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Dice score for a single sample."""
        smooth = 1e-6
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.item()
    
    def _save_visualization(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        prediction: torch.Tensor,
        prediction_binary: torch.Tensor,
        pred_info: Dict,
        index: int
    ):
        """Save visualization of prediction."""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        
        # Plot original image
        axes[0].imshow(image.permute(1, 2, 0))
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Plot ground truth
        axes[1].imshow(mask.squeeze(), cmap='gray')
        axes[1].set_title(f'Ground Truth\nRatio: {pred_info["true_epidermis_ratio"]:.3f}')
        axes[1].axis('off')
        
        # Plot prediction probability
        axes[2].imshow(prediction.squeeze(), cmap='hot', vmin=0, vmax=1)
        axes[2].set_title('Prediction (Probability)')
        axes[2].axis('off')
        
        # Plot binary prediction
        axes[3].imshow(prediction_binary.squeeze(), cmap='gray')
        axes[3].set_title(f'Prediction (Binary)\nDice: {pred_info["dice_score"]:.3f}')
        axes[3].axis('off')
        
        # Add overall title
        dataset = pred_info['dataset']
        patch_id = pred_info['patch_id']
        plt.suptitle(f'{dataset} - {patch_id}')
        
        # Save
        plt.tight_layout()
        save_path = os.path.join(
            self.visualizations_dir,
            f'vis_{index:03d}_{dataset}_{patch_id}.png'
        )
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_summary_plots(self, predictions: List[Dict], metrics: Dict):
        """Create summary plots."""
        # Dice score distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Overall Dice distribution
        dice_scores = [p['dice_score'] for p in predictions]
        axes[0, 0].hist(dice_scores, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(dice_scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(dice_scores):.3f}')
        axes[0, 0].set_xlabel('Dice Score')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Dice Score Distribution')
        axes[0, 0].legend()
        
        # Dice by dataset
        datasets = list(set(p['dataset'] for p in predictions))
        dice_by_dataset = {
            ds: [p['dice_score'] for p in predictions if p['dataset'] == ds]
            for ds in datasets
        }
        
        positions = range(len(datasets))
        axes[0, 1].boxplot(
            [dice_by_dataset[ds] for ds in datasets],
            labels=datasets,
            showmeans=True
        )
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].set_title('Dice Score by Dataset')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Epidermis ratio: predicted vs true
        true_ratios = [p['true_epidermis_ratio'] for p in predictions]
        pred_ratios = [p['pred_epidermis_ratio'] for p in predictions]
        
        axes[1, 0].scatter(true_ratios, pred_ratios, alpha=0.5, s=10)
        axes[1, 0].plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
        axes[1, 0].set_xlabel('True Epidermis Ratio')
        axes[1, 0].set_ylabel('Predicted Epidermis Ratio')
        axes[1, 0].set_title('Epidermis Ratio: Predicted vs True')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confusion matrix for has_epidermis
        true_has_epi = [p['has_epidermis'] for p in predictions]
        pred_has_epi = [p['pred_epidermis_ratio'] > 0.01 for p in predictions]
        cm = confusion_matrix(true_has_epi, pred_has_epi)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Epidermis', 'Has Epidermis'],
            yticklabels=['No Epidermis', 'Has Epidermis'],
            ax=axes[1, 1]
        )
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('True')
        axes[1, 1].set_title('Has Epidermis Classification')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, 'evaluation_summary.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        # Print summary statistics
        self.logger.info("\n" + "="*50)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Total samples: {len(predictions)}")
        self.logger.info(f"Overall Dice: {metrics['dice']:.4f} ± {metrics.get('dice_std', 0):.4f}")
        
        for dataset in datasets:
            dataset_dice = dice_by_dataset[dataset]
            self.logger.info(
                f"{dataset} Dice: {np.mean(dataset_dice):.4f} ± {np.std(dataset_dice):.4f} "
                f"(n={len(dataset_dice)})"
            )
        
        self.logger.info(f"\nHas Epidermis Classification:")
        self.logger.info(f"  Accuracy: {metrics['has_epidermis_accuracy']:.4f}")
        self.logger.info(f"  Precision: {metrics['has_epidermis_precision']:.4f}")
        self.logger.info(f"  Recall: {metrics['has_epidermis_recall']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate epidermis segmentation model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--architecture',
        type=str,
        default='unet',
        choices=['unet', 'deeplabv3plus'],
        help='Model architecture'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='resnet50',
        choices=['resnet50', 'efficientnet-b3'],
        help='Encoder used in model'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
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
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for binary prediction'
    )
    parser.add_argument(
        '--num_visualizations',
        type=int,
        default=50,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['val', 'test'],
        help='Which split to evaluate on'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info(f"Evaluating {args.architecture.upper()} with {args.encoder} encoder")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    model = create_model(args.architecture, args.encoder, args.checkpoint, device)
    
    # Setup data
    logger.info("Setting up data...")
    data_module = EpidermisDataModule(
        patch_root='patches',
        split_file='patches/data_splits.json',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=384,
        use_weighted_sampling=False  # No sampling for evaluation
    )
    data_module.setup()
    
    # Get appropriate dataloader
    if args.split == 'val':
        dataloader = data_module.val_dataloader()
        num_samples = len(data_module.val_dataset)
    else:
        dataloader = data_module.test_dataloader()
        num_samples = len(data_module.test_dataset)
    
    logger.info(f"Evaluating on {num_samples} {args.split} samples")
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        output_dir=args.output_dir,
        threshold=args.threshold
    )
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluator.evaluate(
        dataloader=dataloader,
        save_predictions=True,
        num_visualizations=args.num_visualizations
    )
    
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()