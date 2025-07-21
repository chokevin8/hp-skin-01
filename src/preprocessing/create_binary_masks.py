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
    epidermis_color: Tuple[int, int, int]
) -> Tuple[np.ndarray, float, bool]:
    """
    Create binary mask from multiclass mask by extracting epidermis pixels.
    
    Args:
        mask_path: Path to multiclass mask image
        epidermis_color: RGB tuple for epidermis pixels
        
    Returns:
        binary_mask: Binary mask (0 or 255)
        epidermis_ratio: Ratio of epidermis pixels (for logging)
        has_epidermis: True if any epidermis pixels found
    """
    # Read mask - handle different formats
    if mask_path.endswith('.png'):
        mask = cv2.imread(mask_path)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    elif mask_path.endswith(('.jpg', '.jpeg')):
        mask = cv2.imread(mask_path)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    else:
        # For other formats, use PIL
        mask_pil = Image.open(mask_path).convert('RGB')
        mask_rgb = np.array(mask_pil)
    
    # Create binary mask for epidermis
    epidermis_mask = np.all(mask_rgb == epidermis_color, axis=2)
    binary_mask = epidermis_mask.astype(np.uint8) * 255
    
    # Calculate epidermis ratio and check if any epidermis present
    total_pixels = mask_rgb.shape[0] * mask_rgb.shape[1]
    epidermis_pixels = np.sum(epidermis_mask)
    epidermis_ratio = epidermis_pixels / total_pixels
    has_epidermis = epidermis_pixels > 0
    
    return binary_mask, epidermis_ratio, has_epidermis


def process_dataset(
    dataset_name: str,
    mask_dir: str,
    output_dir: str,
    epidermis_color: Tuple[int, int, int]
) -> pd.DataFrame:
    """
    Process all masks in a dataset to create binary epidermis masks.
    
    Args:
        dataset_name: Name of dataset (Histo-Seg or Queensland)
        mask_dir: Directory containing multiclass masks
        output_dir: Directory to save binary masks
        epidermis_color: RGB tuple for epidermis pixels
        
    Returns:
        DataFrame with processing results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing {dataset_name} dataset...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all mask files - handle different extensions
    mask_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        mask_files.extend(Path(mask_dir).glob(ext))
    
    logger.info(f"Found {len(mask_files)} mask files")
    
    # Process results
    results = []
    no_epidermis_count = 0
    
    for mask_path in tqdm(mask_files, desc=f"Processing {dataset_name}"):
        try:
            # Create binary mask
            binary_mask, epidermis_ratio, has_epidermis = create_binary_mask(
                str(mask_path),
                epidermis_color
            )
            
            # Always save the mask, but note if no epidermis found
            output_filename = mask_path.stem + '.png'
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, binary_mask)
            
            if not has_epidermis:
                logger.info(f"Note: {mask_path.name} has no epidermis pixels")
                no_epidermis_count += 1
                status = 'no_epidermis'
            else:
                status = 'processed'
            
            results.append({
                'dataset': dataset_name,
                'filename': mask_path.name,
                'output_filename': output_filename,
                'epidermis_ratio': epidermis_ratio,
                'status': status,
                'reason': None
            })
            
        except Exception as e:
            logger.error(f"Error processing {mask_path.name}: {str(e)}")
            results.append({
                'dataset': dataset_name,
                'filename': mask_path.name,
                'output_filename': None,
                'epidermis_ratio': None,
                'status': 'error',
                'reason': str(e)
            })
    
    logger.info(
        f"Completed {dataset_name}: "
        f"{len(results) - no_epidermis_count} with epidermis, {no_epidermis_count} without epidermis"
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
# Removed min_epidermis_ratio parameter - no longer filtering by ratio
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
                epidermis_color=config['epidermis_color']
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
        print(f"With epidermis: {len(combined_results[combined_results['status'] == 'processed'])}")
        print(f"Without epidermis: {len(combined_results[combined_results['status'] == 'no_epidermis'])}")
        print(f"Errors: {len(combined_results[combined_results['status'] == 'error'])}")
        print(f"\nAll binary masks saved (including those without epidermis).")


if __name__ == "__main__":
    main()