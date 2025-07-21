"""
Evaluation script for epidermis segmentation models.

Evaluates trained models on test set and generates visualizations.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.datasets.epidermis_dataset import EpidermisDataModule
from src.models.unet_smp import UNetModel
from src.training.metrics import SegmentationMetrics, DiceMetric


class ModelEvaluator:
    """Evaluator for segmentation models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda',
        threshold: float = 0.5
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            device: Device to use
            threshold: Threshold for binary prediction
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        
        # Metrics
        self.metrics = SegmentationMetrics(device=self.device)
        self.dice_metric = DiceMetric()
    
    def evaluate(
        self,
        dataloader,
        save_predictions: bool = False,
        output_dir: str = None
    ) -> Dict:
        """
        Evaluate model on dataset.
        
        Args:
            dataloader: Test dataloader
            save_predictions: Whether to save predictions
            output_dir: Directory to save predictions
            
        Returns:
            Dictionary of metrics
        """
        self.metrics.reset()
        self.dice_metric.reset()
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
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
                
                # Save predictions if requested
                if save_predictions:
                    preds = torch.sigmoid(outputs)
                    for i in range(len(batch['patch_id'])):
                        all_predictions.append({
                            'patch_id': batch['patch_id'][i],
                            'image_id': batch['image_id'][i],
                            'dataset': batch['dataset'][i],
                            'epidermis_ratio_true': batch['epidermis_ratio'][i],
                            'epidermis_ratio_pred': (preds[i] > self.threshold).float().mean().item(),
                            'dice': self._calculate_dice_single(preds[i], masks[i])
                        })
        
        # Compute metrics
        metrics = self.metrics.compute()
        dice_metrics = self.dice_metric.compute()
        metrics.update(dice_metrics)
        
        # Save predictions if requested
        if save_predictions and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as CSV
            df = pd.DataFrame(all_predictions)
            df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
            
            # Save summary statistics
            summary = {
                'overall_metrics': metrics,
                'per_dataset_stats': df.groupby('dataset')['dice'].agg(['mean', 'std']).to_dict(),
                'threshold': self.threshold
            }
            
            with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
        
        return metrics
    
    def _calculate_dice_single(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Dice for single sample."""
        pred_binary = (pred > self.threshold).float()
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum()
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return dice.item()
    
    def visualize_predictions(
        self,
        dataloader,
        num_samples: int = 16,
        output_dir: str = None
    ):
        """
        Visualize model predictions.
        
        Args:
            dataloader: Test dataloader
            num_samples: Number of samples to visualize
            output_dir: Directory to save visualizations
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        samples_visualized = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if samples_visualized >= num_samples:
                    break
                
                # Move to device
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                preds = torch.sigmoid(outputs)
                
                # Visualize each sample in batch
                for i in range(min(len(images), num_samples - samples_visualized)):
                    # Denormalize image
                    img = images[i].cpu().numpy().transpose(1, 2, 0)
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img = np.clip(img, 0, 1)
                    
                    # Get masks
                    true_mask = masks[i, 0].cpu().numpy()
                    pred_mask = (preds[i, 0] > self.threshold).cpu().numpy()
                    
                    # Create figure
                    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                    
                    # Original image
                    axes[0].imshow(img)
                    axes[0].set_title('Original Image')
                    axes[0].axis('off')
                    
                    # True mask
                    axes[1].imshow(true_mask, cmap='gray')
                    axes[1].set_title('True Mask')
                    axes[1].axis('off')
                    
                    # Predicted mask
                    axes[2].imshow(pred_mask, cmap='gray')
                    axes[2].set_title('Predicted Mask')
                    axes[2].axis('off')
                    
                    # Overlay
                    overlay = img.copy()
                    overlay[pred_mask > 0] = [1, 0, 0]  # Red for predictions
                    axes[3].imshow(overlay)
                    axes[3].set_title('Overlay')
                    axes[3].axis('off')
                    
                    plt.tight_layout()
                    
                    if output_dir:
                        save_path = os.path.join(
                            output_dir,
                            f"prediction_{batch['patch_id'][i]}.png"
                        )
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    
                    plt.close()
                    
                    samples_visualized += 1


def plot_metrics_comparison(
    results: Dict[str, Dict],
    output_path: str
):
    """
    Plot comparison of metrics across models.
    
    Args:
        results: Dictionary of model results
        output_path: Path to save plot
    """
    # Extract metrics
    models = list(results.keys())
    metrics_names = ['dice', 'iou', 'precision', 'recall', 'specificity']
    
    # Create DataFrame
    data = []
    for model in models:
        for metric in metrics_names:
            if metric in results[model]:
                data.append({
                    'Model': model,
                    'Metric': metric.upper(),
                    'Value': results[model][metric]
                })
    
    df = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Metric', y='Value', hue='Model')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.legend(title='Model')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate epidermis segmentation model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='resnet50',
        help='Encoder used in model'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='patches',
        help='Root directory of patches'
    )
    parser.add_argument(
        '--split_file',
        type=str,
        default='patches/data_splits.json',
        help='Path to data splits file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations'
    )
    parser.add_argument(
        '--num_visualize',
        type=int,
        default=16,
        help='Number of samples to visualize'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup data
    print("Setting up data...")
    data_module = EpidermisDataModule(
        patch_root=args.data_root,
        split_file=args.split_file,
        batch_size=args.batch_size,
        num_workers=4
    )
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    print(f"Test samples: {len(data_module.test_dataset)}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = UNetModel(encoder_name=args.encoder)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = ModelEvaluator(model)
    
    # Evaluate
    print("Evaluating model...")
    metrics = evaluator.evaluate(
        test_loader,
        save_predictions=True,
        output_dir=args.output_dir
    )
    
    # Print results
    print("\n=== Evaluation Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize predictions
    if args.visualize:
        print("\nGenerating visualizations...")
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        evaluator.visualize_predictions(
            test_loader,
            num_samples=args.num_visualize,
            output_dir=viz_dir
        )
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()