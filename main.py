#!/usr/bin/env python3
"""
HP-Skin-01: Epidermis Segmentation Pipeline
Main entry point for epidermis analysis in skin histopathology
"""

import os
# Disable wandb to avoid protobuf compatibility issues - must be set before ANY imports that use wandb
os.environ['WANDB_MODE'] = 'disabled'

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from datetime import datetime
import torch

# Set security helper to DEV mode
import security_helper.config as SecurityConfig
SecurityConfig.set_mode(_op_mode='DEV', _run_mode='COMPOSE', _output_mode='FILE')

# Set default environment variables for DicomModuleHisto if not set
if 'DEVICE' not in os.environ:
    os.environ['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'GPU_IDX' not in os.environ:
    os.environ['GPU_IDX'] = '0'

# Configure logging for compliance (audit trail)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('epidermis_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import modules
from models.epidermis_classifier import EpidermisClassifier
from postprocessing_epidermis import EpidermisPostProcessor

# Import medical_image I/O utilities and DicomModuleHisto
try:
    import medical_image.utils as MedicalImageUtils
    from medical_image.generate_dcm import DicomModuleHisto
    MEDICAL_IMAGE_AVAILABLE = True
except ImportError:
    logger.warning("medical_image module not available - continuing with basic I/O")
    MEDICAL_IMAGE_AVAILABLE = False
    # Create dummy DicomModuleHisto for inheritance when medical_image not available
    class DicomModuleHisto:
        def __init__(self, web_json, logger=None):
            pass
        def init_models(self):
            pass
        def make_report(self):
            pass

# Import existing preprocessing modules
from src.preprocessing.tissue_segmentation import TissueSegmenter


class EpidermisAnalysisPipeline(DicomModuleHisto):
    """
    Main pipeline for epidermis segmentation analysis
    Inherits from DicomModuleHisto for medical image I/O compliance
    Handles security, processing, and output generation
    """
    
    def __init__(self, config_path: str = 'configuration/deepai_epidermis.json', output_dir: str = None):
        """
        Initialize pipeline
        
        Args:
            config_path: Path to configuration file
            output_dir: Override output directory from config (optional)
        """
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Override output directory if specified
        if output_dir:
            self.config['output']['output_dir'] = output_dir
        
        # Call parent constructor with config
        super().__init__(self.config, logger=logger)
        
        logger.info(f"Initializing Epidermis Analysis Pipeline")
        logger.info(f"Product: {self.config['product_name']}")
        logger.info(f"Model: {self.config['model_name']}")
        logger.info(f"Version: {self.config['version']}")
        
        # Initialize epidermis-specific components
        self.classifier = EpidermisClassifier(config_path)
        self.postprocessor = EpidermisPostProcessor(self.config)
        
        # Initialize tissue segmenter
        tissue_config = self.config['models']['tissue_segmentation']
        self.tissue_segmenter = TissueSegmenter(
            seg_level=tissue_config['seg_level'],
            sthresh=tissue_config['sthresh'],
            mthresh=tissue_config['mthresh'],
            close=tissue_config['close'],
            use_otsu=tissue_config['use_otsu'],
            filter_params=tissue_config['filter_params']
        )
        
        logger.info("Pipeline initialization complete")
        logger.info(f"Model performance - Dice Score: {self.config['models']['epidermis_classifier']['performance']['dice_score']}")
    
    def init_models(self):
        """
        Implementation of DicomModuleHisto abstract method
        Models are already initialized in __init__
        """
        # Models already initialized in __init__
        # This method is required by DicomModuleHisto
        pass
    
    def make_report(self):
        """
        Implementation of DicomModuleHisto abstract method
        Report generation is handled by postprocessor
        """
        # Report generation handled by postprocessor
        # This method is required by DicomModuleHisto
        pass
    
    def process_image(self, image_path: str, output_format: str = 'JSON') -> Dict:
        """
        Process single image
        
        Args:
            image_path: Path to image file
            output_format: Output format (JSON only for now)
            
        Returns:
            Processing results dictionary
        """
        logger.info(f"Processing image: {image_path}")
        
        try:
            # Load image
            if image_path.endswith('.dcm'):
                # Handle DICOM using medical_image utilities if available
                if MEDICAL_IMAGE_AVAILABLE:
                    from medical_image.utils import load_dicom_image
                    image = load_dicom_image(image_path)
                else:
                    raise ValueError("DICOM support requires medical_image module")
            else:
                # Handle standard image formats
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get slide ID
            slide_id = Path(image_path).stem
            
            # Run classification with morphological analysis
            logger.info("Running epidermis segmentation...")
            classification_result = self.classifier.classify_image(image)
            
            # Generate outputs
            logger.info(f"Generating {output_format} output...")
            result = self.postprocessor.process_results(
                image,
                classification_result,
                slide_id,
                output_format
            )
            
            # Add epidermis-specific metrics
            result['epidermis_score'] = classification_result['class_name']
            result['confidence'] = classification_result['confidence']
            result['morphological_metrics'] = classification_result.get('morphological_metrics', {})
            
            # Log result for audit trail
            self._log_result(slide_id, result)
            
            logger.info(f"Processing complete for {slide_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            logger.debug(traceback.format_exc())
            return {
                'status': 'error',
                'error_message': str(e),
                'slide_id': Path(image_path).stem
            }
    
    def process_wsi(self, wsi_path: str, tile_size: int = None, 
                   overlap: int = 0, output_format: str = 'JSON') -> Dict:
        """
        Process whole slide image
        
        Args:
            wsi_path: Path to WSI file
            tile_size: Size of tiles for processing (from config if None)
            overlap: Overlap between tiles
            output_format: Output format
            
        Returns:
            Processing results dictionary
        """
        logger.info(f"Processing WSI: {wsi_path}")
        
        # Use tile_size from config if not specified
        if tile_size is None:
            tile_size = self.config['processing']['tile_size']
        
        try:
            # Try to use medical_image for loading if available
            if MEDICAL_IMAGE_AVAILABLE and wsi_path.lower().endswith(('.svs', '.ndpi', '.mrxs')):
                try:
                    logger.info("Loading WSI using medical_image...")
                    wsi_obj = MedicalImageUtils.load_wsi_image(wsi_path)
                    
                    # Convert to numpy array
                    if hasattr(wsi_obj, 'read_region'):
                        # OpenSlide object
                        dims = wsi_obj.dimensions
                        wsi = np.array(wsi_obj.read_region((0, 0), 0, dims))
                        if wsi.shape[2] == 4:
                            wsi = wsi[:, :, :3]
                    else:
                        wsi = np.array(wsi_obj)
                    
                    logger.info("Successfully loaded WSI using medical_image")
                except Exception as e:
                    logger.warning(f"Could not load with medical_image: {e}, falling back")
                    # Fallback to standard loading
                    wsi = self._load_wsi_fallback(wsi_path)
            else:
                # Use fallback loading for standard formats
                wsi = self._load_wsi_fallback(wsi_path)
            
            slide_id = Path(wsi_path).stem
            h, w = wsi.shape[:2]
            logger.info(f"WSI dimensions: {w}x{h}")
            
            # Initialize full segmentation map
            full_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Process tiles
            n_patches_x = (w + tile_size - 1) // tile_size
            n_patches_y = (h + tile_size - 1) // tile_size
            total_patches = n_patches_x * n_patches_y
            
            logger.info(f"Processing {total_patches} patches")
            patches_processed = 0
            patches_with_epidermis = 0
            
            for py in range(n_patches_y):
                for px in range(n_patches_x):
                    # Extract patch
                    x_start = px * tile_size
                    y_start = py * tile_size
                    x_end = min(x_start + tile_size, w)
                    y_end = min(y_start + tile_size, h)
                    
                    patch = wsi[y_start:y_end, x_start:x_end]
                    
                    # Check for tissue using tissue segmenter logic
                    if self._has_tissue(patch):
                        # Pad if needed
                        patch_h, patch_w = patch.shape[:2]
                        if patch_h < tile_size or patch_w < tile_size:
                            padded = np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 255
                            padded[:patch_h, :patch_w] = patch
                            patch = padded
                        
                        # Classify patch
                        tile_result = self.classifier.classify_image(patch)
                        patch_mask = tile_result['class_map']
                        
                        # Check if patch has epidermis
                        if np.sum(patch_mask > 0) > 0:
                            patches_with_epidermis += 1
                        
                        # Copy to full mask
                        full_mask[y_start:y_end, x_start:x_end] = patch_mask[:patch_h, :patch_w]
                        
                        patches_processed += 1
                        if patches_processed % 100 == 0:
                            logger.info(f"Processed {patches_processed} patches...")
            
            logger.info(f"Processed {patches_processed} tissue patches")
            logger.info(f"Patches with epidermis: {patches_with_epidermis}")
            
            # Create aggregated result with morphological analysis
            mpp = self._get_mpp_for_magnification()
            morphological_metrics = self.classifier.calculate_morphological_metrics(full_mask, mpp)
            
            # Calculate overall statistics
            total_pixels = h * w
            epidermis_pixels = np.sum(full_mask > 0)
            epidermis_percentage = (epidermis_pixels / total_pixels) * 100
            
            aggregated_result = {
                'class_map': full_mask,
                'class': 1 if epidermis_percentage > 1.0 else 0,
                'class_name': 'epidermis' if epidermis_percentage > 1.0 else 'background',
                'confidence': epidermis_percentage / 100,
                'pixel_percentages': {
                    'epidermis': epidermis_percentage,
                    'background': 100 - epidermis_percentage
                },
                'morphological_metrics': morphological_metrics,
                'image_dimensions': {'height': h, 'width': w}
            }
            
            # Create visualization-sized image if needed
            max_viz_size = 4096
            scale = min(max_viz_size / h, max_viz_size / w, 1.0)
            
            if scale < 1.0:
                viz_h = int(h * scale)
                viz_w = int(w * scale)
                viz_image = cv2.resize(wsi, (viz_w, viz_h), interpolation=cv2.INTER_AREA)
                viz_mask = cv2.resize(full_mask, (viz_w, viz_h), interpolation=cv2.INTER_NEAREST)
                aggregated_result['class_map'] = viz_mask
            else:
                viz_image = wsi
            
            # Generate outputs using postprocessor
            result = self.postprocessor.process_results(
                viz_image,
                aggregated_result,
                slide_id,
                output_format
            )
            
            # Add WSI-specific information
            result['wsi_info'] = {
                'dimensions': {'width': w, 'height': h},
                'tile_size': tile_size,
                'overlap': overlap,
                'patches_processed': patches_processed,
                'patches_with_epidermis': patches_with_epidermis
            }
            
            logger.info(f"WSI processing complete for {slide_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing WSI {wsi_path}: {str(e)}")
            logger.debug(traceback.format_exc())
            return {
                'status': 'error',
                'error_message': str(e),
                'slide_id': Path(wsi_path).stem
            }
    
    def process_batch(self, input_dir: str, output_format: str = 'JSON',
                     file_pattern: str = '*.jpg') -> List[Dict]:
        """
        Process batch of images
        
        Args:
            input_dir: Directory containing images
            output_format: Output format
            file_pattern: File pattern to match
            
        Returns:
            List of processing results
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Find all matching files
        image_files = list(input_path.glob(file_pattern))
        
        # Also check for other common formats
        for pattern in ['*.png', '*.tif', '*.tiff']:
            if pattern != file_pattern:
                image_files.extend(input_path.glob(pattern))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        results = []
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {image_path.name}")
            
            # Check if it's a WSI based on extension
            if image_path.suffix.lower() in ['.svs', '.ndpi', '.mrxs', '.tif', '.tiff']:
                result = self.process_wsi(str(image_path), output_format=output_format)
            else:
                result = self.process_image(str(image_path), output_format)
            
            results.append(result)
        
        # Generate batch report
        if results:
            report_path = self.postprocessor.generate_batch_report(results)
            logger.info(f"Batch report saved to {report_path}")
        
        return results
    
    def _load_wsi_fallback(self, wsi_path: str) -> np.ndarray:
        """Fallback WSI loading for standard formats"""
        if wsi_path.lower().endswith(('.tif', '.tiff')):
            import tifffile
            wsi = tifffile.imread(wsi_path)
        else:
            # Try with PIL/opencv for jpg/png
            from PIL import Image
            wsi = np.array(Image.open(wsi_path))
        
        # Ensure RGB format
        if len(wsi.shape) == 2:
            wsi = cv2.cvtColor(wsi, cv2.COLOR_GRAY2RGB)
        elif len(wsi.shape) == 3 and wsi.shape[2] == 4:
            wsi = cv2.cvtColor(wsi, cv2.COLOR_RGBA2RGB)
        
        return wsi
    
    def _has_tissue(self, patch: np.ndarray, threshold: float = None) -> bool:
        """Check if patch contains tissue"""
        if threshold is None:
            threshold = self.config['processing'].get('background_threshold', 0.9)
        
        # Convert to grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # Check for white background
        white_pixels = np.sum(gray > 220)
        total_pixels = gray.size
        white_ratio = white_pixels / total_pixels
        
        return white_ratio < threshold
    
    def _get_mpp_for_magnification(self) -> float:
        """Get microns per pixel based on configuration"""
        morph_config = self.config['processing']['morphological_analysis']
        default_mag = morph_config.get('default_magnification', 20.0)
        
        mpp_map = {
            10.0: morph_config.get('mpp_10x', 1.0),
            20.0: morph_config.get('mpp_20x', 0.5),
            40.0: morph_config.get('mpp_40x', 0.25)
        }
        
        return mpp_map.get(default_mag, 0.5)
    
    def _log_result(self, slide_id: str, result: Dict):
        """
        Log result for audit trail
        
        Args:
            slide_id: Slide identifier
            result: Processing result
        """
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'slide_id': slide_id,
            'epidermis_score': result.get('epidermis_score'),
            'confidence': result.get('confidence'),
            'morphological_metrics': {
                'area_mm2': result.get('morphological_metrics', {}).get('total_area_mm2'),
                'perimeter_mm': result.get('morphological_metrics', {}).get('total_perimeter_mm'),
                'num_contours': result.get('morphological_metrics', {}).get('num_contours')
            },
            'model_version': self.config['version'],
            'user': 'system'
        }
        
        logger.info(f"AUDIT: {json.dumps(audit_entry)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='HP-Skin-01: Epidermis Segmentation Pipeline'
    )
    
    parser.add_argument(
        'input',
        help='Input image, WSI, or directory path'
    )
    
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--output-format',
        choices=['JSON'],
        default='JSON',
        help='Output format (default: JSON)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process batch of images in directory'
    )
    
    parser.add_argument(
        '--wsi',
        action='store_true',
        help='Process as whole slide image'
    )
    
    parser.add_argument(
        '--config',
        default='configuration/deepai_epidermis.json',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--tile-size',
        type=int,
        default=None,
        help='Tile size for WSI processing (default: from config)'
    )
    
    parser.add_argument(
        '--overlap',
        type=int,
        default=0,
        help='Tile overlap for WSI processing (default: 0)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = EpidermisAnalysisPipeline(
            config_path=args.config,
            output_dir=args.output_dir
        )
        
        # Process based on mode
        if args.batch:
            results = pipeline.process_batch(
                args.input,
                output_format=args.output_format
            )
            print(f"Processed {len(results)} images")
            
            # Print summary
            successful = sum(1 for r in results if r.get('status') == 'success')
            print(f"Successful: {successful}/{len(results)}")
            
        elif args.wsi:
            result = pipeline.process_wsi(
                args.input,
                tile_size=args.tile_size,
                overlap=args.overlap,
                output_format=args.output_format
            )
            
            if result['status'] == 'success':
                metrics = result.get('morphological_metrics', {})
                print(f"WSI processing complete")
                print(f"Epidermis area: {metrics.get('total_area_mm2', 0):.2f} mm²")
                print(f"Epidermis perimeter: {metrics.get('total_perimeter_mm', 0):.2f} mm")
                print(f"Number of contours: {metrics.get('num_contours', 0)}")
            else:
                print(f"Processing failed: {result.get('error_message')}")
            
        else:
            # Check if input is WSI based on extension
            input_path = Path(args.input)
            if input_path.suffix.lower() in ['.svs', '.ndpi', '.mrxs', '.tif', '.tiff']:
                result = pipeline.process_wsi(
                    args.input,
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                    output_format=args.output_format
                )
            else:
                result = pipeline.process_image(
                    args.input,
                    output_format=args.output_format
                )
            
            if result['status'] == 'success':
                print(f"Classification: {result.get('epidermis_score', 'N/A')}")
                print(f"Confidence: {result.get('confidence', 0):.2%}")
                
                metrics = result.get('morphological_metrics', {})
                if metrics:
                    print(f"Area: {metrics.get('total_area_mm2', 0):.2f} mm²")
                    print(f"Perimeter: {metrics.get('total_perimeter_mm', 0):.2f} mm")
                    print(f"Contours: {metrics.get('num_contours', 0)}")
            else:
                print(f"Processing failed: {result.get('error_message')}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())