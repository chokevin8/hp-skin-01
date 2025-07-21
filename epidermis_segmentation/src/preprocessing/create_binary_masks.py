"""
Create binary epidermis masks from multiclass segmentation masks.

This script processes multiclass masks from Histo-Seg and Queensland datasets,
extracting only the epidermis regions and creating binary masks.
"""

import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import argparse
import logging
from typing import Tuple, Dict, List


def setup_logging(log_file: str = 'preprocessing_logs.csv'):
    """Setup logging for preprocessing steps."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_binary_mask(
    mask_path: str,
    epidermis_color: Tuple[int, int, int],
    min_epidermis_ratio: float = 0.01
) -> Tuple[np.ndarray, float]:
    """
    Create binary mask from multiclass mask by extracting epidermis pixels.
    
    Args:
        mask_path: Path to multiclass mask image
        epidermis_color: RGB tuple for epidermis pixels
        min_epidermis_ratio: Minimum ratio of epidermis pixels required
        
    Returns:
        binary_mask: Binary mask (0 or 255)
        epidermis_ratio: Ratio of epidermis pixels
    """
    # Read mask
    mask = cv2.imread(mask_path)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    # Create binary mask for epidermis
    epidermis_mask = np.all(mask_rgb == epidermis_color, axis=2)
    binary_mask = epidermis_mask.astype(np.uint8) * 255
    
    # Calculate epidermis ratio
    total_pixels = mask.shape[0] * mask.shape[1]
    epidermis_pixels = np.sum(epidermis_mask)
    epidermis_ratio = epidermis_pixels / total_pixels
    
    return binary_mask, epidermis_ratio


def process_dataset(
    dataset_name: str,
    mask_dir: str,
    output_dir: str,
    epidermis_color: Tuple[int, int, int],
    min_epidermis_ratio: float = 0.01
) -> pd.DataFrame:
    """
    Process all masks in a dataset to create binary epidermis masks.
    
    Args:
        dataset_name: Name of dataset (Histo-Seg or Queensland)
        mask_dir: Directory containing multiclass masks
        output_dir: Directory to save binary masks
        epidermis_color: RGB tuple for epidermis pixels
        min_epidermis_ratio: Minimum ratio of epidermis pixels required
        
    Returns:
        DataFrame with processing results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing {dataset_name} dataset...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all mask files
    mask_files = []
    for ext in ['*.png', '*.tif', '*.jpg']:
        mask_files.extend(Path(mask_dir).glob(ext))
    
    logger.info(f"Found {len(mask_files)} mask files")
    
    # Process results
    results = []
    skipped_count = 0
    
    for mask_path in tqdm(mask_files, desc=f"Processing {dataset_name}"):
        try:
            # Create binary mask
            binary_mask, epidermis_ratio = create_binary_mask(
                str(mask_path),
                epidermis_color,
                min_epidermis_ratio
            )
            
            # Check if mask has enough epidermis
            if epidermis_ratio < min_epidermis_ratio:
                logger.warning(
                    f"Skipping {mask_path.name}: epidermis ratio "
                    f"{epidermis_ratio:.4f} < {min_epidermis_ratio}"
                )
                results.append({
                    'dataset': dataset_name,
                    'filename': mask_path.name,
                    'epidermis_ratio': epidermis_ratio,
                    'status': 'skipped',
                    'reason': 'insufficient_epidermis'
                })
                skipped_count += 1
                continue
            
            # Save binary mask
            output_path = os.path.join(output_dir, mask_path.name)
            cv2.imwrite(output_path, binary_mask)
            
            results.append({
                'dataset': dataset_name,
                'filename': mask_path.name,
                'epidermis_ratio': epidermis_ratio,
                'status': 'processed',
                'reason': None
            })
            
        except Exception as e:
            logger.error(f"Error processing {mask_path.name}: {str(e)}")
            results.append({
                'dataset': dataset_name,
                'filename': mask_path.name,
                'epidermis_ratio': None,
                'status': 'error',
                'reason': str(e)
            })
    
    logger.info(
        f"Completed {dataset_name}: "
        f"{len(results) - skipped_count} processed, {skipped_count} skipped"
    )
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Create binary epidermis masks from multiclass masks"
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='dataset',
        help='Root directory containing datasets'
    )
    parser.add_argument(
        '--min_epidermis_ratio',
        type=float,
        default=0.01,
        help='Minimum ratio of epidermis pixels required (default: 0.01)'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default='preprocessing_logs.csv',
        help='CSV file to save preprocessing logs'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Dataset configurations
    datasets = {
        'Histo-Seg': {
            'mask_dir': os.path.join(args.data_root, 'Histo-Seg', 'Mask'),
            'output_dir': os.path.join(args.data_root, 'Histo-Seg', 'Binary_Mask'),
            'epidermis_color': (112, 48, 160)
        },
        'Queensland': {
            'mask_dir': os.path.join(args.data_root, 'Queensland', 'Mask'),
            'output_dir': os.path.join(args.data_root, 'Queensland', 'Binary_Mask'),
            'epidermis_color': (73, 0, 106)
        }
    }
    
    # Process all datasets
    all_results = []
    
    for dataset_name, config in datasets.items():
        if os.path.exists(config['mask_dir']):
            results_df = process_dataset(
                dataset_name=dataset_name,
                mask_dir=config['mask_dir'],
                output_dir=config['output_dir'],
                epidermis_color=config['epidermis_color'],
                min_epidermis_ratio=args.min_epidermis_ratio
            )
            all_results.append(results_df)
        else:
            logger.warning(f"Dataset directory not found: {config['mask_dir']}")
    
    # Save combined results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(args.log_file, index=False)
        logger.info(f"Saved preprocessing logs to {args.log_file}")
        
        # Print summary
        print("\n=== Preprocessing Summary ===")
        print(combined_results.groupby(['dataset', 'status']).size())
        print(f"\nTotal images processed: {len(combined_results)}")
        print(f"Successfully processed: {len(combined_results[combined_results['status'] == 'processed'])}")
        print(f"Skipped (insufficient epidermis): {len(combined_results[combined_results['status'] == 'skipped'])}")
        print(f"Errors: {len(combined_results[combined_results['status'] == 'error'])}")


if __name__ == "__main__":
    main()