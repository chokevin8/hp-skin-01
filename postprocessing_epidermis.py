#!/usr/bin/env python3
"""
Postprocessing Module for Epidermis Segmentation
Handles visualization and report generation
"""

import os
import json
import logging
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class EpidermisPostProcessor:
    """
    Postprocessor for epidermis segmentation results
    Handles visualization, overlay creation, and report generation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize postprocessor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.viz_config = config['models']['visualization']
        self.output_config = config['output']
        
        # Get visualization parameters
        self.opacity = float(self.viz_config['opacity']['value'])
        self.overlay_color = (
            self.viz_config['overlay_color']['red'],
            self.viz_config['overlay_color']['green'],
            self.viz_config['overlay_color']['blue']
        )
        
        # Output directory
        self.output_dir = Path(self.output_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Postprocessor initialized with output dir: {self.output_dir}")
    
    def process_results(
        self,
        image: np.ndarray,
        classification_result: Dict,
        slide_id: str,
        output_format: str = 'JSON'
    ) -> Dict:
        """
        Process classification results and generate outputs
        
        Args:
            image: Original image
            classification_result: Classification results from classifier
            slide_id: Slide identifier
            output_format: Output format (JSON only for now)
            
        Returns:
            Processed results dictionary
        """
        try:
            # Create output subdirectory for this slide
            slide_output_dir = self.output_dir / slide_id
            slide_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get morphological metrics and contours
            morph_metrics = classification_result.get('morphological_metrics', {})
            
            # Extract contour coordinates
            class_map = classification_result.get('class_map')
            coordinates = self._extract_contour_coordinates(class_map)
            
            # Create the exact output format requested
            result = {
                'type': 'contour',
                'output_type': 'JSON',
                'slide_id': slide_id,
                'short_report': {
                    'total_area_mm2': morph_metrics.get('total_area_mm2', 0.0),
                    'total_perimeter_mm': morph_metrics.get('total_perimeter_mm', 0.0)
                },
                'coordinates': coordinates
            }
            
            # Save outputs based on configuration
            if self.output_config.get('save_mask', True) and class_map is not None:
                mask_path = self._save_mask(class_map, slide_output_dir, slide_id)
            
            if self.output_config.get('save_overlay', True) and class_map is not None:
                overlay = self._create_overlay(image, class_map)
                overlay_path = self._save_overlay(overlay, slide_output_dir, slide_id)
            
            # Generate JSON report (primary output format)
            if output_format in ['JSON', 'ALL']:
                report_path = self._generate_json_report(result, slide_output_dir, slide_id)
            
            # Save visualization with annotations if requested
            if self.viz_config.get('@write_version', {}).get('selected') == 'On':
                annotated = self._create_annotated_visualization(image, class_map, result)
                viz_path = self._save_visualization(annotated, slide_output_dir, slide_id)
            
            logger.info(f"Postprocessing complete for {slide_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in postprocessing for {slide_id}: {str(e)}")
            return {
                'slide_id': slide_id,
                'status': 'error',
                'error_message': str(e)
            }
    
    def _extract_contour_coordinates(self, mask: np.ndarray) -> List[List[List[int]]]:
        """
        Extract contour coordinates from binary mask
        
        Args:
            mask: Binary mask
            
        Returns:
            List of contours, each contour is a list of [x, y] coordinates
        """
        if mask is None:
            return []
        
        # Ensure binary mask
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Convert contours to list of coordinates
        coordinates = []
        for contour in contours:
            # Reshape contour from (N, 1, 2) to (N, 2)
            contour_points = contour.reshape(-1, 2)
            # Convert to list of [x, y] pairs
            contour_list = [[int(point[0]), int(point[1])] for point in contour_points]
            coordinates.append(contour_list)
        
        return coordinates
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Create overlay visualization
        
        Args:
            image: Original image
            mask: Binary mask
            
        Returns:
            Overlay image
        """
        # Ensure mask is binary
        mask_binary = (mask > 127).astype(bool)
        
        # Create colored overlay
        overlay = image.copy()
        
        # Apply overlay color where mask is positive
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_binary,
                (1 - self.opacity) * image[:, :, c] + self.opacity * self.overlay_color[c],
                image[:, :, c]
            )
        
        return overlay.astype(np.uint8)
    
    def _create_annotated_visualization(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        result: Dict
    ) -> np.ndarray:
        """
        Create annotated visualization with metrics
        
        Args:
            image: Original image
            mask: Binary mask
            result: Processing results
            
        Returns:
            Annotated image
        """
        # Create base overlay
        annotated = self._create_overlay(image, mask)
        
        # Convert to PIL for text annotation
        pil_image = Image.fromarray(annotated)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load font, fallback to default if not available
        try:
            font_size = min(image.shape[:2]) // 40
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Add text annotations
        text_y = 10
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0, 180)
        
        # Add title
        title = f"HP-Skin-01: Epidermis Segmentation"
        self._draw_text_with_background(draw, (10, text_y), title, font, text_color, bg_color)
        text_y += 30
        
        # Add metrics from short_report
        short_report = result.get('short_report', {})
        if short_report:
            area = short_report.get('total_area_mm2', 0)
            perimeter = short_report.get('total_perimeter_mm', 0)
            
            metrics_text = [
                f"Area: {area:.2f} mm²",
                f"Perimeter: {perimeter:.2f} mm"
            ]
            
            for text in metrics_text:
                self._draw_text_with_background(draw, (10, text_y), text, font, text_color, bg_color)
                text_y += 25
        
        # Add version info if configured
        if self.viz_config.get('@write_version', {}).get('selected') == 'On':
            version = self.config.get('version', '1.0.0')
            version_text = f"v{version}"
            bbox = draw.textbbox((0, 0), version_text, font=font)
            text_width = bbox[2] - bbox[0]
            x_pos = image.shape[1] - text_width - 10
            self._draw_text_with_background(draw, (x_pos, 10), version_text, font, text_color, bg_color)
        
        return np.array(pil_image)
    
    def _draw_text_with_background(
        self,
        draw: ImageDraw.Draw,
        position: Tuple[int, int],
        text: str,
        font,
        text_color: Tuple,
        bg_color: Tuple
    ):
        """Draw text with background for better visibility"""
        # Get text bounding box
        bbox = draw.textbbox(position, text, font=font)
        
        # Draw background rectangle
        padding = 5
        draw.rectangle(
            [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
            fill=bg_color
        )
        
        # Draw text
        draw.text(position, text, fill=text_color, font=font)
    
    def _save_mask(self, mask: np.ndarray, output_dir: Path, slide_id: str) -> Path:
        """Save binary mask"""
        mask_path = output_dir / f"{slide_id}_epidermis_mask.png"
        cv2.imwrite(str(mask_path), mask)
        logger.info(f"Saved mask to {mask_path}")
        return mask_path
    
    def _save_overlay(self, overlay: np.ndarray, output_dir: Path, slide_id: str) -> Path:
        """Save overlay image"""
        overlay_path = output_dir / f"{slide_id}_overlay.png"
        # Convert RGB to BGR for cv2
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(overlay_path), overlay_bgr)
        logger.info(f"Saved overlay to {overlay_path}")
        return overlay_path
    
    def _save_visualization(self, image: np.ndarray, output_dir: Path, slide_id: str) -> Path:
        """Save annotated visualization"""
        viz_path = output_dir / f"{slide_id}_visualization.png"
        # Convert RGB to BGR for cv2
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(viz_path), image_bgr)
        logger.info(f"Saved visualization to {viz_path}")
        return viz_path
    
    def _generate_json_report(self, result: Dict, output_dir: Path, slide_id: str) -> Path:
        """Generate JSON report with exact format requested"""
        report_path = output_dir / f"{slide_id}_result.json"
        
        # Save the exact format requested
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved JSON report to {report_path}")
        return report_path
    
    def generate_batch_report(self, results: List[Dict]) -> Path:
        """
        Generate report for batch processing
        
        Args:
            results: List of processing results
            
        Returns:
            Path to batch report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"batch_report_{timestamp}.json"
        
        # Aggregate statistics
        total_slides = len(results)
        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = total_slides - successful
        
        # Calculate average metrics for successful results
        total_area = 0
        total_perimeter = 0
        total_contours = 0
        count = 0
        
        for result in results:
            if result.get('status') == 'success' and 'morphological_metrics' in result:
                metrics = result['morphological_metrics']
                total_area += metrics.get('total_area_mm2', 0)
                total_perimeter += metrics.get('total_perimeter_mm', 0)
                total_contours += metrics.get('num_contours', 0)
                count += 1
        
        avg_area = total_area / count if count > 0 else 0
        avg_perimeter = total_perimeter / count if count > 0 else 0
        avg_contours = total_contours / count if count > 0 else 0
        
        # Create batch report
        batch_report = {
            'metadata': {
                'timestamp': timestamp,
                'software_version': self.config.get('version', '1.0.0'),
                'product_name': self.config.get('product_name', 'HP-Skin-01')
            },
            'summary': {
                'total_slides': total_slides,
                'successful': successful,
                'failed': failed,
                'average_metrics': {
                    'area_mm2': avg_area,
                    'perimeter_mm': avg_perimeter,
                    'num_contours': avg_contours
                }
            },
            'individual_results': results
        }
        
        # Save batch report
        with open(report_path, 'w') as f:
            json.dump(batch_report, f, indent=2)
        
        logger.info(f"Saved batch report to {report_path}")
        return report_path