#!/usr/bin/env python
"""
Fabric Detection Module for SAM2 Initialization.

This module provides intelligent fabric detection using depth and color cues
to find a reliable prompt point for SAM2 segmentation.

Strategy:
1. Use depth to extract foreground objects
2. Identify and mask robot parts (low saturation, depth edges)
3. Identify and mask table (white, flat surface)
4. Fabric = foreground - robot - table
5. Select centroid of largest fabric blob
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectionResult:
    """Result of fabric detection."""
    point: Optional[Tuple[int, int]]  # (x, y) point on fabric, or None if failed
    confidence: float  # 0.0-1.0, higher is better
    fabric_mask: np.ndarray  # (H, W) bool mask of detected fabric
    debug_info: dict  # Additional debug information


@dataclass
class FabricDetectorConfig:
    """Configuration for fabric detection."""
    # Depth filtering
    depth_min: float = 0.2  # meters - filter very close objects
    depth_max: float = 1.0  # meters - filter background/table
    
    # Robot detection (assumes robot is low saturation)
    robot_saturation_max: float = 0.3  # HSV saturation (0-1)
    robot_value_range: Tuple[float, float] = (0.0, 1.0)  # HSV value range
    
    # Table detection (white surface)
    table_saturation_max: float = 0.2  # HSV saturation
    table_value_min: float = 0.7  # HSV value (bright)
    table_depth_std_max: float = 0.02  # meters - flat surface
    
    # Fabric validation
    min_fabric_area: int = 1000  # pixels
    min_confidence: float = 0.5
    
    # Morphology operations
    morph_kernel_size: int = 5
    
    # Static mask
    static_mask_path: Optional[str] = None  # Path to .png mask for fixed objects
    
    # Debug
    enable_debug: bool = False


class FabricDetector:
    """Intelligent fabric detection for SAM2 initialization.
    
    Uses depth and color information to distinguish fabric from:
    - Robot arm/gripper (low saturation, depth edges)
    - Table surface (white, flat depth)
    - Background (far depth)
    
    Works with any fabric color by using elimination strategy.
    """
    
    def __init__(self, config: FabricDetectorConfig, depth_scale: float = 0.00025):
        """Initialize fabric detector.
        
        Args:
            config: Detection configuration
            depth_scale: Depth units to meters conversion
        """
        self.config = config
        self.depth_scale = depth_scale
        
        # Load static mask if provided
        self.static_mask: Optional[np.ndarray] = None
        self.static_mask_original_shape: Optional[Tuple[int, int]] = None
        if config.static_mask_path:
            if os.path.exists(config.static_mask_path):
                mask_img = cv2.imread(config.static_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    self.static_mask = mask_img > 127  # Binary mask
                    self.static_mask_original_shape = self.static_mask.shape
                    masked_pixels = np.sum(self.static_mask)
                    total_pixels = self.static_mask.size
                    print(f"✓ Loaded static mask from {config.static_mask_path}")
                    print(f"  Mask shape: {self.static_mask.shape}")
                    print(f"  Masked pixels: {masked_pixels} / {total_pixels} ({masked_pixels/total_pixels*100:.1f}%)")
                else:
                    print(f"✗ Failed to load static mask from {config.static_mask_path} (cv2.imread returned None)")
            else:
                print(f"✗ Static mask path does not exist: {config.static_mask_path}")
        
        # Cached resized mask (will be populated on first use)
        self._cached_mask: Optional[np.ndarray] = None
        self._cached_mask_shape: Optional[Tuple[int, int]] = None
        
        # Morphology kernel
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.morph_kernel_size, config.morph_kernel_size)
        )
    
    def detect(self, rgb: np.ndarray, depth: np.ndarray) -> DetectionResult:
        """Detect fabric and return a prompt point.
        
        Args:
            rgb: (H, W, 3) uint8 RGB image
            depth: (H, W) uint16 depth in camera units
            
        Returns:
            DetectionResult with point, confidence, and masks
        """
        H, W = depth.shape
        
        # Convert depth to meters
        depth_m = depth.astype(np.float32) * self.depth_scale
        
        # Step 1: Depth-based foreground extraction
        foreground_mask = self._extract_foreground(depth_m)
        
        # Step 2: Robot segmentation
        robot_mask = self._detect_robot(rgb, depth_m)
        
        # Step 3: Table segmentation
        table_mask = self._detect_table(rgb, depth_m)
        
        # Step 4: Apply static mask if available
        if self.static_mask is not None:
            # Resize static mask to match image size if needed (cached)
            if self._cached_mask is None or self._cached_mask_shape != (H, W):
                if self.static_mask_original_shape != (H, W):
                    self._cached_mask = cv2.resize(
                        self.static_mask.astype(np.uint8),
                        (W, H),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    self._cached_mask_shape = (H, W)
                    print(f"  Static mask resized from {self.static_mask_original_shape} to {(H, W)} (cached)")
                else:
                    self._cached_mask = self.static_mask
                    self._cached_mask_shape = (H, W)
                    print(f"  Static mask shape matches processing resolution (no resize needed)")
            
            static_mask_resized = self._cached_mask
            
            if self.config.enable_debug:
                print(f"  Applying static mask: {np.sum(static_mask_resized)} pixels masked")
        else:
            static_mask_resized = np.zeros((H, W), dtype=bool)
        
        # Step 5: Fabric = foreground - robot - table - static
        fabric_mask_before_static = (
            foreground_mask &
            ~robot_mask &
            ~table_mask
        )
        fabric_mask = fabric_mask_before_static & ~static_mask_resized
        
        if self.config.enable_debug and self.static_mask is not None:
            pixels_removed = np.sum(fabric_mask_before_static) - np.sum(fabric_mask)
            print(f"  Static mask removed {pixels_removed} fabric pixels")
        
        # Step 6: Clean up mask with morphology
        fabric_mask = self._clean_mask(fabric_mask)
        
        # Step 7: Find point and estimate confidence
        point, confidence = self._select_point(fabric_mask)
        
        # Prepare debug info
        debug_info = {
            'foreground_mask': foreground_mask,
            'robot_mask': robot_mask,
            'table_mask': table_mask,
            'static_mask': static_mask_resized,
            'fabric_area': np.sum(fabric_mask),
            'num_components': self._count_components(fabric_mask),
        }
        
        return DetectionResult(
            point=point,
            confidence=confidence,
            fabric_mask=fabric_mask,
            debug_info=debug_info
        )
    
    def _extract_foreground(self, depth_m: np.ndarray) -> np.ndarray:
        """Extract foreground using depth thresholds.
        
        Args:
            depth_m: (H, W) depth in meters
            
        Returns:
            (H, W) bool mask of foreground
        """
        # Valid depth range
        valid_depth = depth_m > 0
        
        # Within interest zone
        in_range = (depth_m >= self.config.depth_min) & (depth_m <= self.config.depth_max)
        
        foreground = valid_depth & in_range
        return foreground
    
    def _detect_robot(self, rgb: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """Detect robot parts using low saturation and depth edges.
        
        Robot parts are typically:
        - Black/white/gray (low saturation)
        - Have sharp depth discontinuities (edges)
        
        Args:
            rgb: (H, W, 3) uint8 RGB image
            depth_m: (H, W) depth in meters
            
        Returns:
            (H, W) bool mask of robot parts
        """
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # Low saturation = grayscale = robot
        low_saturation = s < self.config.robot_saturation_max
        
        # Value in valid range
        value_ok = (v >= self.config.robot_value_range[0]) & (v <= self.config.robot_value_range[1])
        
        # Depth gradient (sharp edges = robot/objects)
        depth_for_grad = depth_m.copy()
        depth_for_grad[depth_m == 0] = np.nan  # Ignore invalid depth
        
        # Compute gradient magnitude
        grad_y = np.abs(np.gradient(depth_for_grad, axis=0))
        grad_x = np.abs(np.gradient(depth_for_grad, axis=1))
        grad_mag = np.sqrt(grad_y**2 + grad_x**2)
        grad_mag[np.isnan(grad_mag)] = 0
        
        # High gradient = edges (likely robot or object boundaries)
        high_gradient = grad_mag > 0.05  # 5cm depth change
        
        # Dilate high gradient regions to capture robot body
        high_gradient_dilated = cv2.dilate(
            high_gradient.astype(np.uint8),
            self.morph_kernel,
            iterations=2
        ).astype(bool)
        
        # Combine: low saturation OR high gradient edges
        robot_mask = (low_saturation & value_ok) | high_gradient_dilated
        
        return robot_mask
    
    def _detect_table(self, rgb: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """Detect white table surface using color and depth flatness.
        
        Args:
            rgb: (H, W, 3) uint8 RGB image
            depth_m: (H, W) depth in meters
            
        Returns:
            (H, W) bool mask of table
        """
        H, W = depth_m.shape
        
        # Convert to HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        
        # White = low saturation + high value
        is_white = (s < self.config.table_saturation_max) & (v > self.config.table_value_min)
        
        # Compute local depth standard deviation (flatness indicator)
        # Use a window to compute local std
        kernel_size = 15
        depth_valid = depth_m.copy()
        depth_valid[depth_m == 0] = np.nan
        
        # Local std using convolution
        depth_mean = cv2.blur(depth_valid, (kernel_size, kernel_size))
        depth_sq_mean = cv2.blur(depth_valid**2, (kernel_size, kernel_size))
        depth_std = np.sqrt(np.abs(depth_sq_mean - depth_mean**2))
        depth_std[np.isnan(depth_std)] = 1.0  # Invalid regions are not flat
        
        # Flat surface = low std
        is_flat = depth_std < self.config.table_depth_std_max
        
        # Table is typically at bottom of image (lower rows)
        # Create a weight that favors bottom region
        y_coords = np.arange(H)[:, None]  # (H, 1)
        bottom_weight = (y_coords / H) > 0.5  # Bottom half
        
        # Combine: white AND flat AND preferably at bottom
        table_mask = is_white & is_flat
        
        # Keep only large connected component at bottom
        if np.any(table_mask & bottom_weight):
            # Find connected components
            num_labels, labels = cv2.connectedComponents(table_mask.astype(np.uint8))
            
            # Find largest component that overlaps with bottom region
            max_area = 0
            best_label = 0
            for label_id in range(1, num_labels):
                mask_i = labels == label_id
                area_i = np.sum(mask_i)
                bottom_overlap = np.sum(mask_i & bottom_weight)
                
                # Prefer components with good bottom overlap and large area
                score = area_i * (1 + bottom_overlap / max(area_i, 1))
                if score > max_area:
                    max_area = score
                    best_label = label_id
            
            if best_label > 0:
                table_mask = labels == best_label
        
        return table_mask
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean mask using morphological operations.
        
        Args:
            mask: (H, W) bool mask
            
        Returns:
            (H, W) cleaned bool mask
        """
        mask_uint8 = mask.astype(np.uint8)
        
        # Opening (erosion + dilation) to remove small noise
        opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Closing (dilation + erosion) to fill small holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.morph_kernel)
        
        return closed.astype(bool)
    
    def _count_components(self, mask: np.ndarray) -> int:
        """Count number of connected components in mask.
        
        Args:
            mask: (H, W) bool mask
            
        Returns:
            Number of components
        """
        num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8))
        return num_labels - 1  # Exclude background
    
    def _select_point(self, fabric_mask: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
        """Select a reliable point on the fabric and estimate confidence.
        
        Args:
            fabric_mask: (H, W) bool mask of fabric
            
        Returns:
            Tuple of (point, confidence)
            - point: (x, y) or None if detection failed
            - confidence: 0.0-1.0
        """
        # Find connected components
        num_labels, labels = cv2.connectedComponents(fabric_mask.astype(np.uint8))
        
        if num_labels <= 1:  # Only background
            return None, 0.0
        
        # Find largest component
        areas = []
        for label_id in range(1, num_labels):
            area = np.sum(labels == label_id)
            areas.append((area, label_id))
        
        areas.sort(reverse=True)
        largest_area, largest_label = areas[0]
        
        # Check if area is sufficient
        if largest_area < self.config.min_fabric_area:
            confidence = largest_area / self.config.min_fabric_area
            return None, confidence
        
        # Get largest component mask
        largest_mask = labels == largest_label
        
        # Compute centroid
        y_coords, x_coords = np.where(largest_mask)
        centroid_x = int(np.mean(x_coords))
        centroid_y = int(np.mean(y_coords))
        
        # Estimate confidence based on:
        # 1. Area (larger is better)
        # 2. Compactness (area / perimeter ratio)
        # 3. Number of components (fewer is better)
        
        # Area score (saturates at 5x minimum)
        area_score = min(1.0, largest_area / (5 * self.config.min_fabric_area))
        
        # Compactness score
        contours, _ = cv2.findContours(
            largest_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 0:
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter > 0:
                compactness = (4 * np.pi * largest_area) / (perimeter ** 2)
                compactness_score = min(1.0, compactness)  # Circle = 1.0
            else:
                compactness_score = 0.5
        else:
            compactness_score = 0.5
        
        # Component score (penalize fragmentation)
        component_score = 1.0 / (1.0 + (num_labels - 2))  # -2 because we have background + 1 good component
        
        # Overall confidence
        confidence = (
            0.5 * area_score +
            0.3 * compactness_score +
            0.2 * component_score
        )
        
        return (centroid_x, centroid_y), confidence
    
    def visualize_detection(
        self,
        rgb: np.ndarray,
        result: DetectionResult,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """Create visualization of detection process.
        
        Args:
            rgb: (H, W, 3) uint8 RGB image
            result: Detection result
            save_path: Optional path to save visualization
            
        Returns:
            (H*2, W*2, 3) uint8 visualization image
        """
        H, W = rgb.shape[:2]
        
        # Create 2x2 grid
        vis = np.zeros((H * 2, W * 2, 3), dtype=np.uint8)
        
        # Top-left: Original RGB
        vis[:H, :W] = rgb
        
        # Top-right: Foreground (green), Robot (red), Table (blue)
        overlay = rgb.copy()
        debug = result.debug_info
        
        overlay[debug['foreground_mask']] = overlay[debug['foreground_mask']] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
        overlay[debug['robot_mask']] = overlay[debug['robot_mask']] * 0.5 + np.array([255, 0, 0], dtype=np.uint8) * 0.5
        overlay[debug['table_mask']] = overlay[debug['table_mask']] * 0.5 + np.array([0, 0, 255], dtype=np.uint8) * 0.5
        
        vis[:H, W:] = overlay
        
        # Bottom-left: Final fabric mask
        fabric_vis = rgb.copy()
        fabric_vis[result.fabric_mask] = fabric_vis[result.fabric_mask] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
        
        if result.point is not None:
            cv2.circle(fabric_vis, result.point, 10, (255, 0, 255), -1)
            cv2.circle(fabric_vis, result.point, 12, (255, 255, 255), 2)
        
        vis[H:, :W] = fabric_vis
        
        # Bottom-right: Stats
        stats_img = np.zeros((H, W, 3), dtype=np.uint8)
        
        text_lines = [
            f"Confidence: {result.confidence:.2f}",
            f"Fabric area: {debug['fabric_area']} px",
            f"Components: {debug['num_components']}",
            f"Point: {result.point}" if result.point else "Point: None",
        ]
        
        y_offset = 30
        for line in text_lines:
            cv2.putText(
                stats_img,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            y_offset += 30
        
        vis[H:, W:] = stats_img
        
        # Add labels
        cv2.putText(vis, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, "Masks (G:FG R:Robot B:Table)", (W+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, "Detected Fabric", (10, H+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, "Statistics", (W+10, H+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        return vis
