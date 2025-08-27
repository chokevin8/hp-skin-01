#!/usr/bin/env python3
"""
Epidermis Classifier Module
Wraps the DeepLabv3+ model for epidermis segmentation
"""

import os
# Disable wandb to avoid protobuf compatibility issues
os.environ['WANDB_MODE'] = 'disabled'

import sys
import json
import logging
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Add src to path for importing existing modules
sys.path.append(str(Path(__file__).parent.parent))

# Import existing model architecture
from src.models.deeplabv3plus_smp import create_deeplabv3plus_models

logger = logging.getLogger(__name__)


class EpidermisClassifier:
    """
    Epidermis segmentation classifier
    Wraps the DeepLabv3+ model and provides classification interface
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the epidermis classifier
        
        Args:
            config_path: Path to configuration JSON file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model_config = self.config['models']['epidermis_classifier']
        self.tissue_config = self.config['models']['tissue_segmentation']
        self.morph_config = self.config['processing']['morphological_analysis']
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        self.patch_size = self.model_config['patch_size']
        self.batch_size = self.model_config['batch_size']
        
        logger.info("Epidermis classifier initialized successfully")
    
    def _load_model(self) -> nn.Module:
        """Load the DeepLabv3+ model from checkpoint"""
        try:
            # Create model architecture
            encoder = self.model_config['encoder']
            models = create_deeplabv3plus_models({'encoders': [encoder]})
            model_key = f"deeplabv3plus_{encoder}"  # Keep hyphen as-is
            model = models[model_key]
            
            # Load checkpoint
            checkpoint_path = Path(self.model_config['model_path'])
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            logger.info(f"Model loaded successfully from {checkpoint_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def classify_image(self, image: np.ndarray) -> Dict:
        """
        Classify a single image/patch
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            
        Returns:
            Dictionary containing classification results
        """
        try:
            # Ensure correct input shape
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            h, w = image.shape[:2]
            
            # Check if we need to process in patches or as whole
            if h <= self.patch_size and w <= self.patch_size:
                # Process as single patch
                mask = self._process_single_patch(image)
            else:
                # Process in patches
                mask = self._process_patches(image)
            
            # Calculate morphological metrics
            mpp = self._get_mpp_for_image(image)
            morphological_metrics = self.calculate_morphological_metrics(mask, mpp)
            
            # Calculate pixel percentages
            total_pixels = mask.size
            epidermis_pixels = np.sum(mask > 0)
            epidermis_percentage = (epidermis_pixels / total_pixels) * 100
            
            # Prepare result
            result = {
                'class_map': mask,
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
            
            return result
            
        except Exception as e:
            logger.error(f"Error in classify_image: {str(e)}")
            raise
    
    def _process_single_patch(self, patch: np.ndarray) -> np.ndarray:
        """Process a single patch through the model"""
        # Pad if needed
        h, w = patch.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            padded = np.ones((self.patch_size, self.patch_size, 3), dtype=np.uint8) * 255
            padded[:h, :w] = patch
            patch = padded
        
        # Convert to tensor
        patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1) / 255.0
        patch_tensor = patch_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(patch_tensor)
            pred = torch.sigmoid(output) > 0.5
            pred = pred[0, 0].cpu().numpy().astype(np.uint8) * 255
        
        # Crop back to original size
        if h < self.patch_size or w < self.patch_size:
            pred = pred[:h, :w]
        
        return pred
    
    def _process_patches(self, image: np.ndarray) -> np.ndarray:
        """Process image in patches"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Process in non-overlapping patches
        for y in range(0, h, self.patch_size):
            for x in range(0, w, self.patch_size):
                # Extract patch
                y_end = min(y + self.patch_size, h)
                x_end = min(x + self.patch_size, w)
                patch = image[y:y_end, x:x_end]
                
                # Check if patch has tissue
                if self._has_tissue(patch):
                    # Process patch
                    patch_mask = self._process_single_patch(patch)
                    
                    # Place in full mask
                    mask[y:y_end, x:x_end] = patch_mask[:y_end-y, :x_end-x]
        
        return mask
    
    def _has_tissue(self, patch: np.ndarray, threshold: float = 0.9) -> bool:
        """Check if patch contains tissue"""
        # Convert to grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # Check for white background
        white_pixels = np.sum(gray > 220)
        total_pixels = gray.size
        white_ratio = white_pixels / total_pixels
        
        return white_ratio < threshold
    
    def _get_mpp_for_image(self, image: np.ndarray) -> float:
        """Get microns per pixel based on configuration"""
        # Default to 20x magnification MPP
        default_mag = self.morph_config.get('default_magnification', 20.0)
        
        mpp_map = {
            10.0: self.morph_config.get('mpp_10x', 1.0),
            20.0: self.morph_config.get('mpp_20x', 0.5),
            40.0: self.morph_config.get('mpp_40x', 0.25)
        }
        
        return mpp_map.get(default_mag, 0.5)
    
    def calculate_morphological_metrics(
        self,
        mask: np.ndarray,
        mpp: float
    ) -> Dict:
        """
        Calculate morphological metrics for epidermis regions
        
        Args:
            mask: Binary mask with epidermis regions
            mpp: Microns per pixel
            
        Returns:
            Dictionary with morphological metrics
        """
        # Ensure binary mask
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Calculate total area
        total_area_pixels = binary_mask.sum()
        area_um2 = total_area_pixels * (mpp ** 2)
        area_mm2 = area_um2 / 1e6  # Convert µm² to mm²
        
        # Calculate perimeters and contour details
        total_perimeter_pixels = 0
        contour_details = []
        
        for i, contour in enumerate(contours):
            # Calculate contour metrics
            contour_area_pixels = cv2.contourArea(contour)
            contour_area_mm2 = (contour_area_pixels * (mpp ** 2)) / 1e6
            
            contour_perimeter_pixels = cv2.arcLength(contour, closed=True)
            contour_perimeter_mm = (contour_perimeter_pixels * mpp) / 1000
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Store contour details (only top 10 largest)
            if len(contour_details) < 10:
                contour_details.append({
                    'id': i + 1,
                    'area_mm2': float(contour_area_mm2),
                    'perimeter_mm': float(contour_perimeter_mm),
                    'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'num_points': len(contour)
                })
            
            total_perimeter_pixels += contour_perimeter_pixels
        
        # Convert total perimeter to mm
        perimeter_mm = (total_perimeter_pixels * mpp) / 1000
        
        # Sort contours by area
        contour_details.sort(key=lambda x: x['area_mm2'], reverse=True)
        
        return {
            'total_area_mm2': float(area_mm2),
            'total_perimeter_mm': float(perimeter_mm),
            'num_contours': len(contours),
            'contours': contour_details,
            'mpp': mpp,
            'pixel_metrics': {
                'total_pixels': int(total_area_pixels),
                'image_coverage_percent': float((total_area_pixels / mask.size) * 100)
            }
        }
    
    def process_wsi_patches(
        self,
        patches: List[np.ndarray],
        batch_process: bool = True
    ) -> List[Dict]:
        """
        Process multiple patches from a WSI
        
        Args:
            patches: List of patch arrays
            batch_process: Whether to process in batches
            
        Returns:
            List of classification results
        """
        results = []
        
        if batch_process:
            # Process in batches
            for i in range(0, len(patches), self.batch_size):
                batch = patches[i:i + self.batch_size]
                batch_results = self._process_batch(batch)
                results.extend(batch_results)
        else:
            # Process individually
            for patch in patches:
                result = self.classify_image(patch)
                results.append(result)
        
        return results
    
    def _process_batch(self, batch: List[np.ndarray]) -> List[Dict]:
        """Process a batch of patches"""
        # Convert batch to tensor
        batch_tensors = []
        original_sizes = []
        
        for patch in batch:
            h, w = patch.shape[:2]
            original_sizes.append((h, w))
            
            # Pad if needed
            if h < self.patch_size or w < self.patch_size:
                padded = np.ones((self.patch_size, self.patch_size, 3), dtype=np.uint8) * 255
                padded[:h, :w] = patch
                patch = padded
            
            tensor = torch.from_numpy(patch).float().permute(2, 0, 1) / 255.0
            batch_tensors.append(tensor)
        
        batch_tensor = torch.stack(batch_tensors).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            preds = torch.sigmoid(outputs) > 0.5
            preds = preds.cpu().numpy().astype(np.uint8) * 255
        
        # Create results
        results = []
        mpp = self._get_mpp_for_image(batch[0])
        
        for i, (pred, orig_size) in enumerate(zip(preds, original_sizes)):
            mask = pred[0]  # Remove channel dimension
            
            # Crop to original size
            h, w = orig_size
            if h < self.patch_size or w < self.patch_size:
                mask = mask[:h, :w]
            
            # Calculate metrics
            epidermis_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            epidermis_percentage = (epidermis_pixels / total_pixels) * 100
            
            results.append({
                'class_map': mask,
                'pixel_percentages': {
                    'epidermis': epidermis_percentage,
                    'background': 100 - epidermis_percentage
                }
            })
        
        return results