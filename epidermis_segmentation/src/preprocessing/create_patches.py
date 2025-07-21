"""
Create patches from WSI and corresponding masks with tissue segmentation.

This script extracts non-overlapping patches from whole slide images,
filtering out background regions and creating paired WSI-mask patches.
"""

import os
import json
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import argparse
import logging
from typing import Tuple, List, Dict, Optional
import h5py
from sklearn.model_selection import train_test_split

from tissue_segmentation import TissueSegmenter


def setup_logging(log_file: str = 'patching_logs.txt'):
    """Setup logging for patching process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class WSIPatcher:
    """Extract patches from WSI and mask pairs."""
    
    def __init__(
        self,
        patch_size: int = 384,
        overlap: float = 0.0,
        tissue_threshold: float = 0.1,
        segmenter_params: Optional[Dict] = None
    ):
        """
        Initialize WSI patcher.
        
        Args:
            patch_size: Size of patches to extract
            overlap: Overlap ratio between patches (0.0 for non-overlapping)
            tissue_threshold: Minimum tissue ratio to keep patch
            segmenter_params: Parameters for tissue segmentation
        """
        self.patch_size = patch_size
        self.stride = int(patch_size * (1 - overlap))
        self.tissue_threshold = tissue_threshold
        self.segmenter = TissueSegmenter(**(segmenter_params or {}))
        self.logger = logging.getLogger(__name__)
    
    def extract_patches(
        self,
        wsi_path: str,
        mask_path: str,
        save_dir: str,
        image_id: str
    ) -> Dict:
        """
        Extract patches from WSI and mask pair.
        
        Args:
            wsi_path: Path to WSI image
            mask_path: Path to binary mask
            save_dir: Directory to save patches
            image_id: Unique identifier for the image
            
        Returns:
            Dictionary with extraction metadata
        """
        # Load images
        try:
            wsi = np.array(Image.open(wsi_path))
            mask = np.array(Image.open(mask_path))
            
            # Ensure mask is binary
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            mask = (mask > 127).astype(np.uint8) * 255
            
        except Exception as e:
            self.logger.error(f"Error loading images: {e}")
            return {'status': 'error', 'error': str(e)}
        
        # Check dimensions match
        if wsi.shape[:2] != mask.shape[:2]:
            self.logger.error(
                f"Dimension mismatch: WSI {wsi.shape[:2]} vs Mask {mask.shape[:2]}"
            )
            return {'status': 'error', 'error': 'dimension_mismatch'}
        
        # Extract patches
        h, w = wsi.shape[:2]
        patch_coords = []
        valid_patches = 0
        total_patches = 0
        
        # Create save directories
        wsi_save_dir = os.path.join(save_dir, 'wsi')
        mask_save_dir = os.path.join(save_dir, 'mask')
        os.makedirs(wsi_save_dir, exist_ok=True)
        os.makedirs(mask_save_dir, exist_ok=True)
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                # Extract patch
                wsi_patch = wsi[y:y+self.patch_size, x:x+self.patch_size]
                mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                
                total_patches += 1
                
                # Check if patch contains sufficient tissue
                if self.segmenter.is_tissue_patch(wsi_patch, self.tissue_threshold):
                    # Save patches
                    patch_id = f"{image_id}_y{y}_x{x}"
                    
                    wsi_patch_path = os.path.join(
                        wsi_save_dir, f"{patch_id}.png"
                    )
                    mask_patch_path = os.path.join(
                        mask_save_dir, f"{patch_id}.png"
                    )
                    
                    Image.fromarray(wsi_patch).save(wsi_patch_path)
                    Image.fromarray(mask_patch).save(mask_patch_path)
                    
                    # Store coordinates
                    patch_coords.append({
                        'patch_id': patch_id,
                        'x': x,
                        'y': y,
                        'epidermis_ratio': np.sum(mask_patch > 0) / (self.patch_size ** 2)
                    })
                    
                    valid_patches += 1
        
        # Save metadata
        metadata = {
            'image_id': image_id,
            'wsi_path': wsi_path,
            'mask_path': mask_path,
            'image_size': [h, w],
            'patch_size': self.patch_size,
            'stride': self.stride,
            'total_patches': total_patches,
            'valid_patches': valid_patches,
            'patch_coords': patch_coords
        }
        
        metadata_path = os.path.join(save_dir, f"{image_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(
            f"Extracted {valid_patches}/{total_patches} valid patches from {image_id}"
        )
        
        return metadata


def create_split_file(
    patch_metadata: List[Dict],
    output_path: str,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42
):
    """
    Create train/val/test split file.
    
    Args:
        patch_metadata: List of patch metadata dictionaries
        output_path: Path to save split file
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed
    """
    # Extract unique image IDs and their dataset source
    image_info = []
    for meta in patch_metadata:
        dataset = 'Histo-Seg' if 'Histo-Seg' in meta['wsi_path'] else 'Queensland'
        image_info.append({
            'image_id': meta['image_id'],
            'dataset': dataset,
            'n_patches': meta['valid_patches']
        })
    
    df = pd.DataFrame(image_info)
    
    # Stratified split by dataset
    train_val_ids, test_ids = train_test_split(
        df['image_id'],
        test_size=test_size,
        stratify=df['dataset'],
        random_state=random_state
    )
    
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size / (1 - test_size),
        stratify=df[df['image_id'].isin(train_val_ids)]['dataset'],
        random_state=random_state
    )
    
    # Create split dictionary
    splits = {
        'train': train_ids.tolist(),
        'val': val_ids.tolist(),
        'test': test_ids.tolist()
    }
    
    # Add patch-level information
    patch_splits = {'train': [], 'val': [], 'test': []}
    
    for meta in patch_metadata:
        image_id = meta['image_id']
        if image_id in train_ids:
            split = 'train'
        elif image_id in val_ids:
            split = 'val'
        else:
            split = 'test'
        
        for patch in meta['patch_coords']:
            patch_splits[split].append({
                'image_id': image_id,
                'patch_id': patch['patch_id'],
                'epidermis_ratio': patch['epidermis_ratio']
            })
    
    # Save split file
    split_data = {
        'image_splits': splits,
        'patch_splits': patch_splits,
        'statistics': {
            'train': len(train_ids),
            'val': len(val_ids),
            'test': len(test_ids),
            'train_patches': len(patch_splits['train']),
            'val_patches': len(patch_splits['val']),
            'test_patches': len(patch_splits['test'])
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"\nSplit statistics:")
    print(f"Train: {len(train_ids)} images, {len(patch_splits['train'])} patches")
    print(f"Val: {len(val_ids)} images, {len(patch_splits['val'])} patches")
    print(f"Test: {len(test_ids)} images, {len(patch_splits['test'])} patches")


def main():
    parser = argparse.ArgumentParser(
        description="Create patches from WSI and mask pairs"
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='dataset',
        help='Root directory containing datasets'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='patches',
        help='Directory to save patches'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=384,
        help='Patch size (default: 384)'
    )
    parser.add_argument(
        '--tissue_threshold',
        type=float,
        default=0.1,
        help='Minimum tissue ratio to keep patch (default: 0.1)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.1,
        help='Test set fraction (default: 0.1)'
    )
    parser.add_argument(
        '--val_size',
        type=float,
        default=0.1,
        help='Validation set fraction (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging('patching_logs.txt')
    
    # Initialize patcher
    patcher = WSIPatcher(
        patch_size=args.patch_size,
        overlap=0.0,  # Non-overlapping
        tissue_threshold=args.tissue_threshold
    )
    
    # Process datasets
    datasets = ['Histo-Seg', 'Queensland']
    all_metadata = []
    
    for dataset in datasets:
        wsi_dir = os.path.join(args.data_root, dataset, 'WSI')
        mask_dir = os.path.join(args.data_root, dataset, 'Binary_Mask')
        
        if not os.path.exists(wsi_dir) or not os.path.exists(mask_dir):
            logger.warning(f"Skipping {dataset}: directories not found")
            continue
        
        # Get paired files
        wsi_files = sorted([f for f in os.listdir(wsi_dir) 
                           if f.endswith(('.png', '.tif', '.jpg', '.svs'))])
        mask_files = sorted([f for f in os.listdir(mask_dir) 
                            if f.endswith(('.png', '.tif', '.jpg'))])
        
        # Match files by name
        paired_files = []
        for wsi_file in wsi_files:
            base_name = Path(wsi_file).stem
            # Find corresponding mask
            mask_file = None
            for mf in mask_files:
                if Path(mf).stem == base_name:
                    mask_file = mf
                    break
            
            if mask_file:
                paired_files.append((wsi_file, mask_file))
            else:
                logger.warning(f"No mask found for {wsi_file}")
        
        logger.info(f"Found {len(paired_files)} paired files in {dataset}")
        
        # Process each pair
        dataset_output_dir = os.path.join(args.output_dir, dataset)
        
        for wsi_file, mask_file in tqdm(paired_files, desc=f"Processing {dataset}"):
            wsi_path = os.path.join(wsi_dir, wsi_file)
            mask_path = os.path.join(mask_dir, mask_file)
            image_id = f"{dataset}_{Path(wsi_file).stem}"
            
            metadata = patcher.extract_patches(
                wsi_path=wsi_path,
                mask_path=mask_path,
                save_dir=dataset_output_dir,
                image_id=image_id
            )
            
            if metadata.get('status') != 'error':
                all_metadata.append(metadata)
    
    # Create train/val/test splits
    if all_metadata:
        split_path = os.path.join(args.output_dir, 'data_splits.json')
        create_split_file(
            all_metadata,
            split_path,
            test_size=args.test_size,
            val_size=args.val_size
        )
        logger.info(f"Created data splits: {split_path}")
    
    logger.info("Patching complete!")


if __name__ == "__main__":
    main()