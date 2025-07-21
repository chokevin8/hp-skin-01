"""
PyTorch Dataset for epidermis segmentation.

Loads paired WSI and mask patches for training/validation/testing.
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Optional, Tuple
import logging


class EpidermisDataset(Dataset):
    """Dataset for epidermis segmentation from WSI patches."""
    
    def __init__(
        self,
        patch_root: str,
        split_file: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        patch_size: int = 384
    ):
        """
        Initialize dataset.
        
        Args:
            patch_root: Root directory containing patches
            split_file: Path to JSON file with data splits
            split: Which split to use ('train', 'val', 'test')
            transform: Albumentations transform pipeline
            patch_size: Expected patch size
        """
        self.patch_root = patch_root
        self.split = split
        self.patch_size = patch_size
        self.logger = logging.getLogger(__name__)
        
        # Load split information
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        self.patches = split_data['patch_splits'][split]
        self.logger.info(f"Loaded {len(self.patches)} patches for {split} split")
        
        # Calculate class distribution for balancing
        self.class_distribution = self._calculate_class_distribution()
        
        # Default transform if none provided
        if transform is None:
            if split == 'train':
                self.transform = self._get_train_transform()
            else:
                self.transform = self._get_val_transform()
        else:
            self.transform = transform
    
    def _get_train_transform(self) -> A.Compose:
        """Get default training augmentations."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def _get_val_transform(self) -> A.Compose:
        """Get default validation/test transform."""
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def _calculate_class_distribution(self) -> Dict[str, int]:
        """Calculate distribution of patches with/without epidermis."""
        with_epidermis = 0
        without_epidermis = 0
        
        for patch in self.patches:
            if patch['epidermis_ratio'] > 0:
                with_epidermis += 1
            else:
                without_epidermis += 1
        
        distribution = {
            'with_epidermis': with_epidermis,
            'without_epidermis': without_epidermis,
            'total': len(self.patches)
        }
        
        self.logger.info(f"Class distribution - With epidermis: {with_epidermis}, Without epidermis: {without_epidermis}")
        return distribution
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Index of sample
            
        Returns:
            Dictionary with 'image', 'mask', and metadata
        """
        patch_info = self.patches[idx]
        
        # Extract dataset name from image_id
        dataset_name = 'Histo-Seg' if 'Histo-Seg' in patch_info['image_id'] else 'Queensland'
        
        # Construct paths
        wsi_path = os.path.join(
            self.patch_root,
            dataset_name,
            'wsi',
            f"{patch_info['patch_id']}.png"
        )
        mask_path = os.path.join(
            self.patch_root,
            dataset_name,
            'mask',
            f"{patch_info['patch_id']}.png"
        )
        
        # Load images
        try:
            image = np.array(Image.open(wsi_path).convert('RGB'))
            mask = np.array(Image.open(mask_path).convert('L'))
        except Exception as e:
            self.logger.error(f"Error loading patch {patch_info['patch_id']}: {e}")
            # Return black patch as fallback
            image = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
            mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        
        # Convert mask to binary (0 or 1)
        mask = (mask > 127).astype(np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure mask has correct shape for loss computation
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension
        
        return {
            'image': image,
            'mask': mask,
            'patch_id': patch_info['patch_id'],
            'image_id': patch_info['image_id'],
            'epidermis_ratio': patch_info['epidermis_ratio'],
            'dataset': dataset_name
        }
    
    def get_weighted_sampler(self, oversample_factor: Optional[float] = None) -> WeightedRandomSampler:
        """
        Create weighted sampler to balance epidermis/non-epidermis patches.
        
        Args:
            oversample_factor: Factor to oversample epidermis patches. 
                             If None, automatically calculates to achieve 50:50 balance.
            
        Returns:
            WeightedRandomSampler for balanced sampling
        """
        # Calculate actual class imbalance if oversample_factor not provided
        if oversample_factor is None:
            with_epi = self.class_distribution['with_epidermis']
            without_epi = self.class_distribution['without_epidermis']
            
            if with_epi > 0:
                # Calculate factor to achieve 50:50 balance
                oversample_factor = without_epi / with_epi
                self.logger.info(f"Auto-calculated oversample factor: {oversample_factor:.2f} "
                               f"(ratio without:with = {without_epi}:{with_epi})")
            else:
                # No epidermis patches at all - shouldn't happen but handle gracefully
                oversample_factor = 1.0
                self.logger.warning("No patches with epidermis found! Using oversample factor 1.0")
        
        # Calculate weights for each sample
        weights = []
        
        for patch in self.patches:
            if patch['epidermis_ratio'] > 0:
                # Higher weight for patches with epidermis
                weights.append(oversample_factor)
            else:
                # Lower weight for patches without epidermis
                weights.append(1.0)
        
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )


class EpidermisDataModule:
    """Data module for managing train/val/test datasets."""
    
    def __init__(
        self,
        patch_root: str,
        split_file: str,
        batch_size: int = 16,
        num_workers: int = 8,
        patch_size: int = 384,
        train_transform: Optional[A.Compose] = None,
        val_transform: Optional[A.Compose] = None,
        use_weighted_sampling: bool = True,
        oversample_factor: Optional[float] = None
    ):
        """
        Initialize data module.
        
        Args:
            patch_root: Root directory containing patches
            split_file: Path to JSON file with data splits
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            patch_size: Expected patch size
            train_transform: Custom training transforms
            val_transform: Custom validation/test transforms
            use_weighted_sampling: Use weighted sampling for training
            oversample_factor: Factor to oversample epidermis patches (None = auto-calculate for 50:50 balance)
        """
        self.patch_root = patch_root
        self.split_file = split_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.use_weighted_sampling = use_weighted_sampling
        self.oversample_factor = oversample_factor
    
    def setup(self):
        """Setup datasets."""
        self.train_dataset = EpidermisDataset(
            self.patch_root,
            self.split_file,
            split='train',
            transform=self.train_transform,
            patch_size=self.patch_size
        )
        
        self.val_dataset = EpidermisDataset(
            self.patch_root,
            self.split_file,
            split='val',
            transform=self.val_transform,
            patch_size=self.patch_size
        )
        
        self.test_dataset = EpidermisDataset(
            self.patch_root,
            self.split_file,
            split='test',
            transform=self.val_transform,
            patch_size=self.patch_size
        )
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get training dataloader with optional weighted sampling."""
        if self.use_weighted_sampling:
            # Use weighted sampler for balanced training
            sampler = self.train_dataset.get_weighted_sampler(self.oversample_factor)
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,  # Use sampler instead of shuffle
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True
            )
        else:
            # Standard random sampling
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True
            )
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get validation dataloader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get test dataloader."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def get_sample_batch(self, n_samples: int = 4) -> Dict[str, torch.Tensor]:
        """Get a sample batch for visualization."""
        indices = np.random.choice(len(self.train_dataset), n_samples, replace=False)
        batch = []
        
        for idx in indices:
            batch.append(self.train_dataset[idx])
        
        # Stack into batch
        images = torch.stack([b['image'] for b in batch])
        masks = torch.stack([b['mask'] for b in batch])
        
        return {
            'images': images,
            'masks': masks,
            'patch_ids': [b['patch_id'] for b in batch],
            'epidermis_ratios': [b['epidermis_ratio'] for b in batch]
        }