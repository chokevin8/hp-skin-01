"""
Training script for epidermis segmentation.

Trains U-Net models with wandb logging and experiment tracking.
"""

import os
import sys
import argparse
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import wandb
from typing import Dict, Optional, Tuple
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.epidermis_dataset import EpidermisDataModule
from models.unet_smp import create_unet_models
from training.losses import get_loss_function
from training.metrics_simple import SegmentationMetrics, DiceMetric


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class EpidermisTrainer:
    """Trainer for epidermis segmentation models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        checkpoint_dir: str,
        wandb_run = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            checkpoint_dir: Directory to save checkpoints
            wandb_run: Wandb run object
        """
        self.model = model
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.wandb_run = wandb_run
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = get_loss_function(
            config.get('loss', 'dice'),
            **config.get('loss_params', {})
        )
        
        # Optimizer
        if hasattr(model, 'get_params_groups'):
            param_groups = model.get_params_groups(config['learning_rate'])
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=config.get('weight_decay', 1e-5)
            )
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 1e-5)
            )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize Dice
            factor=config.get('lr_factor', 0.5),
            patience=config.get('lr_patience', 10),
            min_lr=config.get('min_lr', 1e-6)
        )
        
        # Metrics
        self.train_metrics = SegmentationMetrics(device=self.device)
        self.val_metrics = SegmentationMetrics(device=self.device)
        self.dice_metric = DiceMetric()
        
        # Training state
        self.current_epoch = 0
        self.best_val_dice = 0.0
        self.patience_counter = 0
        
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        self.train_metrics.reset()
        
        pbar = tqdm(dataloader, desc=f'Epoch {self.current_epoch} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            self.train_metrics.update(outputs, masks)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to wandb
            if self.wandb_run and batch_idx % 10 == 0:
                self.wandb_run.log({
                    'train/batch_loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        # Compute epoch metrics
        metrics = self.train_metrics.compute()
        metrics['loss'] = epoch_loss / len(dataloader)
        
        return metrics
    
    def validate_epoch(self, dataloader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_loss = 0.0
        self.val_metrics.reset()
        self.dice_metric.reset()
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f'Epoch {self.current_epoch} [Val]')
            
            for batch in pbar:
                # Move to device
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                epoch_loss += loss.item()
                self.val_metrics.update(outputs, masks)
                self.dice_metric.update(
                    outputs, masks,
                    sample_info={
                        'patch_id': batch['patch_id'],
                        'dataset': batch['dataset']
                    }
                )
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        metrics = self.val_metrics.compute()
        dice_metrics = self.dice_metric.compute()
        metrics.update(dice_metrics)
        metrics['loss'] = epoch_loss / len(dataloader)
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with Dice: {self.best_val_dice:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_dice = checkpoint['best_val_dice']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int
    ):
        """
        Train the model.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs to train
        """
        early_stopping_patience = self.config.get('early_stopping_patience', 15)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.logger.info(
                f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Dice: {train_metrics['dice']:.4f}"
            )
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            self.logger.info(
                f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Dice: {val_metrics['dice']:.4f}"
            )
            
            # Per-dataset metrics
            for key, value in val_metrics.items():
                if key.startswith('dice_') and key.endswith('Seg'):
                    self.logger.info(f"  {key}: {value:.4f}")
            
            # Update learning rate
            self.scheduler.step(val_metrics['dice'])
            
            # Check for improvement
            is_best = val_metrics['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Log to wandb
            if self.wandb_run:
                log_dict = {
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/dice': train_metrics['dice'],
                    'val/loss': val_metrics['loss'],
                    'val/dice': val_metrics['dice'],
                    'val/iou': val_metrics['iou'],
                    'val/precision': val_metrics['precision'],
                    'val/recall': val_metrics['recall'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                
                # Add per-dataset metrics
                for key, value in val_metrics.items():
                    if key not in ['loss', 'dice', 'iou', 'precision', 'recall']:
                        log_dict[f'val/{key}'] = value
                
                self.wandb_run.log(log_dict)
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after {early_stopping_patience} epochs"
                )
                break
        
        self.logger.info(f"Training complete. Best Dice: {self.best_val_dice:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train epidermis segmentation model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='resnet50',
        choices=['resnet50', 'efficientnet-b3'],
        help='Encoder to use for U-Net'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='epidermis-segmentation',
        help='Wandb project name'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Experiment name for wandb'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'patch_root': 'patches',
            'split_file': 'patches/data_splits.json',
            'batch_size': 32,  # Optimized for 48GB GPU
            'num_workers': 16,  # Optimized for better data loading
            'patch_size': 384,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'num_epochs': 200,
            'early_stopping_patience': 15,
            'gradient_clip': 1.0,
            'loss': 'dice',
            'loss_params': {'smooth': 1e-6, 'apply_sigmoid': True},
            'lr_factor': 0.5,
            'lr_patience': 10,
            'min_lr': 1e-6,
            'use_weighted_sampling': True,
            'oversample_factor': None,  # Auto-calculate for 50:50 balance
            'wandb_api_key': '056cc8f5fd5428b2d91107a96c0da5f5ae1c3476',
            'wandb_entity': 'chokevin8'
        }
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = args.experiment_name or f"unet_{args.encoder}_{timestamp}"
    experiment_dir = os.path.join('experiments', experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(experiment_dir)
    logger.info(f"Starting experiment: {experiment_name}")
    
    # Initialize wandb
    wandb.login(key=config['wandb_api_key'])
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=config.get('wandb_entity', 'chokevin8'),
        name=experiment_name,
        config=config,
        dir=experiment_dir
    )
    
    # Setup data
    logger.info("Setting up data...")
    data_module = EpidermisDataModule(
        patch_root=config['patch_root'],
        split_file=config['split_file'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        patch_size=config['patch_size'],
        use_weighted_sampling=config.get('use_weighted_sampling', True),
        oversample_factor=config.get('oversample_factor', 2.0)
    )
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    logger.info(f"Train samples: {len(data_module.train_dataset)}")
    logger.info(f"Val samples: {len(data_module.val_dataset)}")
    
    # Log class distribution and balancing strategy
    train_dist = data_module.train_dataset.class_distribution
    logger.info(f"Training class distribution: {train_dist}")
    if config.get('use_weighted_sampling', True):
        oversample = config.get('oversample_factor')
        if oversample is None:
            logger.info("Using weighted sampling with auto-calculated oversample factor for 50:50 balance")
        else:
            logger.info(f"Using weighted sampling with manual oversample factor: {oversample}")
    else:
        logger.info("Using standard random sampling")
    
    # Create model
    logger.info(f"Creating U-Net model with {args.encoder} encoder...")
    models = create_unet_models({'encoders': [args.encoder]})
    model = models[f'unet_{args.encoder}']
    
    # Create trainer
    trainer = EpidermisTrainer(
        model=model,
        config=config,
        checkpoint_dir=checkpoint_dir,
        wandb_run=wandb_run
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train model
    logger.info("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs']
    )
    
    # Finish wandb run
    wandb.finish()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()