"""
Tissue segmentation utilities for WSI preprocessing.
Based on CLAM's tissue segmentation approach.
"""

import numpy as np
import cv2
from PIL import Image
from skimage import morphology, filters
from typing import Tuple, Dict, Optional


class TissueSegmenter:
    """
    Tissue segmentation for whole slide images.
    Identifies tissue regions and removes background.
    """
    
    def __init__(
        self,
        seg_level: int = -1,
        sthresh: int = 8,
        mthresh: int = 7,
        close: int = 4,
        use_otsu: bool = False,
        filter_params: Optional[Dict] = None
    ):
        """
        Initialize tissue segmenter.
        
        Args:
            seg_level: Level for segmentation (-1 for lowest resolution)
            sthresh: Saturation threshold
            mthresh: Mean threshold
            close: Morphological closing kernel size
            use_otsu: Whether to use Otsu thresholding
            filter_params: Parameters for filtering (area thresholds, etc.)
        """
        self.seg_level = seg_level
        self.sthresh = sthresh
        self.mthresh = mthresh
        self.close = close
        self.use_otsu = use_otsu
        
        # Default filter parameters
        self.filter_params = filter_params or {
            'a_t': 100,      # Area threshold
            'a_h': 16,       # Hole area threshold
            'max_n_holes': 8  # Maximum number of holes
        }
    
    def segment_tissue(
        self,
        img: np.ndarray,
        return_mask: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Segment tissue from background in WSI.
        
        Args:
            img: Input image (RGB)
            return_mask: Whether to return the tissue mask
            
        Returns:
            contours: List of tissue contours
            mask: Binary tissue mask (if return_mask=True)
        """
        # Convert to HSV for better color separation
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Create tissue mask based on saturation and brightness
        if self.use_otsu:
            # Otsu thresholding on saturation channel
            _, mask_s = cv2.threshold(
                img_hsv[:, :, 1], 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            # Otsu thresholding on value channel
            _, mask_v = cv2.threshold(
                img_hsv[:, :, 2], 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            # Manual thresholding
            mask_s = img_hsv[:, :, 1] > self.sthresh
            mask_v = img_hsv[:, :, 2] > self.mthresh
        
        # Combine masks
        mask = np.logical_and(mask_s, mask_v).astype(np.uint8) * 255
        
        # Morphological closing to fill gaps
        if self.close > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.close, self.close)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.filter_params['a_t']:
                filtered_contours.append(contour)
        
        # Create final mask from filtered contours
        if return_mask:
            final_mask = np.zeros_like(mask)
            cv2.drawContours(final_mask, filtered_contours, -1, 255, -1)
            
            # Remove small holes
            final_mask = self._remove_small_holes(
                final_mask,
                self.filter_params['a_h'],
                self.filter_params['max_n_holes']
            )
            
            return filtered_contours, final_mask
        
        return filtered_contours, None
    
    def _remove_small_holes(
        self,
        mask: np.ndarray,
        min_hole_area: int,
        max_n_holes: int
    ) -> np.ndarray:
        """Remove small holes from binary mask."""
        # Invert mask to find holes
        inverted = cv2.bitwise_not(mask)
        
        # Find hole contours
        contours, _ = cv2.findContours(
            inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Keep only significant holes
        holes_to_fill = []
        for i, contour in enumerate(contours):
            if i >= max_n_holes:
                holes_to_fill.append(contour)
            elif cv2.contourArea(contour) < min_hole_area:
                holes_to_fill.append(contour)
        
        # Fill small holes
        cv2.drawContours(mask, holes_to_fill, -1, 255, -1)
        
        return mask
    
    def is_tissue_patch(
        self,
        patch: np.ndarray,
        tissue_threshold: float = 0.1
    ) -> bool:
        """
        Check if a patch contains sufficient tissue.
        
        Args:
            patch: Image patch
            tissue_threshold: Minimum fraction of tissue pixels
            
        Returns:
            True if patch contains sufficient tissue
        """
        _, mask = self.segment_tissue(patch, return_mask=True)
        tissue_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        return tissue_ratio >= tissue_threshold