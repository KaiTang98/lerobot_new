#!/usr/bin/env python
"""
Create a static mask from a captured image with support for multiple polygons.

This script:
1. Captures one RGB-D frame using the camera settings from config.yaml
2. Displ                # Save mask
                cv2.imwrite(str(output_path), self.mask)
                print(f"\n✓ Mask saved to: {output_path.absolute()}")
                print(f"  Number of polygons: {len(all_polygons)}")
                print(f"  Masked pixels: {np.sum(self.mask > 0)} / {self.mask.size} "
                      f"({np.sum(self.mask > 0) / self.mask.size * 100:.1f}%)")
                print(f"\nTo use this mask, add to config.yaml:")
                print(f"  fabric_detection:")
                print(f"    static_mask_path: {output_path}")GB image
3. Allows you to draw multiple masks by clicking points to create polygons
4. Each polygon can be finished and a new one started
5. Saves the combined mask as a PNG file that can be used with fabric_detection.static_mask_path

Usage:
    python create_static_mask.py --config config.yaml --output static_mask.png
    
Controls:
    - Left click: Add point to current polygon
    - Right click: Remove last point from current polygon
    - 'c': Clear current polygon
    - 'n': Finish current polygon and start a new one
    - 'u': Undo (remove last completed polygon)
    - 's': Save mask and exit
    - 'q': Quit without saving
    - 'f': Fill polygons (toggle preview)
"""

import argparse
import yaml
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple

from camera_manager import CameraManager


class MaskCreator:
    """Interactive mask creation tool with support for multiple polygons."""
    
    def __init__(self, rgb: np.ndarray, output_path: str):
        """Initialize mask creator.
        
        Args:
            rgb: (H, W, 3) RGB image
            output_path: Path to save the mask
        """
        self.rgb = rgb
        self.output_path = output_path
        self.current_points: List[Tuple[int, int]] = []  # Current polygon being drawn
        self.completed_polygons: List[List[Tuple[int, int]]] = []  # List of completed polygons
        self.mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        self.show_filled = False
        
        # Create window
        cv2.namedWindow("Create Mask (Multiple Polygons)", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Create Mask (Multiple Polygons)", self._mouse_callback)
        
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point to current polygon
            self.current_points.append((x, y))
            print(f"Polygon {len(self.completed_polygons) + 1}, Point {len(self.current_points)}: ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last point from current polygon
            if self.current_points:
                removed = self.current_points.pop()
                print(f"Removed point: {removed}")
    
    def _draw_display(self) -> np.ndarray:
        """Draw the current state on display image."""
        display = self.rgb.copy()
        
        # Draw completed polygons
        for poly_idx, polygon in enumerate(self.completed_polygons):
            # Draw points
            for i, pt in enumerate(polygon):
                cv2.circle(display, pt, 4, (0, 200, 0), -1)  # Darker green for completed
            
            # Draw lines
            if len(polygon) > 1:
                for i in range(len(polygon)):
                    pt1 = polygon[i]
                    pt2 = polygon[(i+1) % len(polygon)]
                    cv2.line(display, pt1, pt2, (0, 200, 0), 2)
            
            # Draw filled polygon if requested
            if self.show_filled and len(polygon) >= 3:
                overlay = display.copy()
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], (255, 0, 0))
                display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
            
            # Label polygon
            if polygon:
                centroid = np.mean(polygon, axis=0).astype(int)
                cv2.putText(display, f"P{poly_idx+1}", tuple(centroid),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        
        # Draw current polygon being drawn (brighter green)
        for i, pt in enumerate(self.current_points):
            cv2.circle(display, pt, 5, (0, 255, 0), -1)
            cv2.putText(display, str(i+1), (pt[0]+10, pt[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw lines between points in current polygon
        if len(self.current_points) > 1:
            for i in range(len(self.current_points)):
                pt1 = self.current_points[i]
                pt2 = self.current_points[(i+1) % len(self.current_points)]
                cv2.line(display, pt1, pt2, (0, 255, 0), 2)
        
        # Draw filled current polygon if requested and enough points
        if self.show_filled and len(self.current_points) >= 3:
            overlay = display.copy()
            pts = np.array(self.current_points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (255, 0, 0))
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
        
        # Draw instructions
        instructions = [
            "Left click: Add point | Right click: Remove last",
            "C: Clear current polygon | N: Finish & start new",
            "U: Undo last polygon | F: Toggle fill preview",
            "S: Save mask | Q: Quit without saving",
            f"Current polygon: {len(self.current_points)} points",
            f"Completed polygons: {len(self.completed_polygons)}"
        ]
        y_offset = 30
        for instruction in instructions:
            cv2.putText(display, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
        
        return display
    
    def run(self) -> bool:
        """Run the interactive mask creation.
        
        Returns:
            True if mask was saved, False if cancelled
        """
        print("\n" + "="*60)
        print("MASK CREATION TOOL - MULTIPLE POLYGONS SUPPORT")
        print("="*60)
        print("\nInstructions:")
        print("  1. Click points on the image to define a polygon")
        print("  2. Press 'n' to finish current polygon and start a new one")
        print("  3. Create as many polygons as needed (multiple objects)")
        print("  4. All polygons will be combined into one mask")
        print("  5. Press 's' to save when done")
        print("  6. Press 'q' to quit without saving")
        print("\nControls:")
        print("  Left click:  Add point to current polygon")
        print("  Right click: Remove last point from current polygon")
        print("  'c': Clear current polygon (not yet finished)")
        print("  'n': Finish current polygon and start a new one")
        print("  'u': Undo - remove last completed polygon")
        print("  'f': Toggle fill preview")
        print("  's': Save all polygons as combined mask and exit")
        print("  'q': Quit without saving")
        print("="*60 + "\n")
        
        while True:
            display = self._draw_display()
            cv2.imshow("Create Mask (Multiple Polygons)", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nCancelled - mask not saved")
                cv2.destroyAllWindows()
                return False
                
            elif key == ord('s'):
                # Need at least one polygon with 3+ points
                all_polygons = self.completed_polygons.copy()
                if len(self.current_points) >= 3:
                    all_polygons.append(self.current_points)
                
                if not all_polygons:
                    print("\nError: No polygons to save. Draw at least one polygon with 3+ points.")
                    continue
                
                if any(len(poly) < 3 for poly in all_polygons):
                    print("\nError: All polygons must have at least 3 points")
                    continue
                
                # Create mask by combining all polygons
                self.mask = np.zeros(self.rgb.shape[:2], dtype=np.uint8)
                for polygon in all_polygons:
                    pts = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(self.mask, [pts], 255)
                
                # Create parent directory if it doesn't exist
                output_path = Path(self.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save mask
                cv2.imwrite(str(output_path), self.mask)
                print(f"\n✓ Mask saved to: {self.output_path}")
                print(f"  Number of polygons: {len(all_polygons)}")
                print(f"  Masked pixels: {np.sum(self.mask > 0)} / {self.mask.size} "
                      f"({np.sum(self.mask > 0) / self.mask.size * 100:.1f}%)")
                print(f"\nTo use this mask, add to config.yaml:")
                print(f"  fabric_detection:")
                print(f"    static_mask_path: {self.output_path}")
                
                cv2.destroyAllWindows()
                return True
                
            elif key == ord('c'):
                self.current_points = []
                print(f"\nCleared current polygon (kept {len(self.completed_polygons)} completed polygons)")
                
            elif key == ord('n'):
                # Finish current polygon and start new one
                if len(self.current_points) >= 3:
                    self.completed_polygons.append(self.current_points.copy())
                    print(f"\n✓ Finished polygon {len(self.completed_polygons)} "
                          f"with {len(self.current_points)} points")
                    self.current_points = []
                    print(f"Started polygon {len(self.completed_polygons) + 1}")
                elif len(self.current_points) > 0:
                    print(f"\nError: Need at least 3 points to finish polygon (have {len(self.current_points)})")
                else:
                    print("\nNo current polygon to finish")
                    
            elif key == ord('u'):
                # Undo last completed polygon
                if self.completed_polygons:
                    removed = self.completed_polygons.pop()
                    print(f"\n↶ Undid polygon {len(self.completed_polygons) + 1} "
                          f"(had {len(removed)} points)")
                else:
                    print("\nNo completed polygons to undo")
                
            elif key == ord('f'):
                self.show_filled = not self.show_filled
                status = "ON" if self.show_filled else "OFF"
                print(f"\nFill preview: {status}")


def main():
    parser = argparse.ArgumentParser(description="Create static mask from camera image")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='static_mask.png',
                       help='Output mask file path')
    parser.add_argument('--serial', type=str, default=None,
                       help='Camera serial number (overrides config)')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract camera settings
    camera_cfg = config.get('camera', {})
    camera_serial = args.serial or camera_cfg.get('serial', 'f1181599')
    camera_color_width = camera_cfg.get('color_width', camera_cfg.get('width', 640))
    camera_color_height = camera_cfg.get('color_height', camera_cfg.get('height', 480))
    camera_depth_width = camera_cfg.get('depth_width', camera_cfg.get('width', 640))
    camera_depth_height = camera_cfg.get('depth_height', camera_cfg.get('height', 480))
    camera_fps = camera_cfg.get('fps', 30)
    
    print("\n" + "="*60)
    print("STATIC MASK CREATION")
    print("="*60)
    print(f"\nCamera settings:")
    print(f"  Serial: {camera_serial}")
    print(f"  Color resolution: {camera_color_width}x{camera_color_height}")
    print(f"  Depth resolution: {camera_depth_width}x{camera_depth_height}")
    print(f"  FPS: {camera_fps}")
    print(f"\nOutput: {args.output}")
    print("="*60 + "\n")
    
    # Initialize camera
    print("Initializing camera...")
    try:
        camera = CameraManager(
            serial=camera_serial,
            color_width=camera_color_width,
            color_height=camera_color_height,
            depth_width=camera_depth_width,
            depth_height=camera_depth_height,
            fps=camera_fps,
            use_depth=True,
        )
        camera.start()
        print("✓ Camera initialized\n")
    except Exception as e:
        print(f"✗ Failed to initialize camera: {e}")
        return 1
    
    # Capture one frame
    print("Capturing frame...")
    try:
        rgb, depth = camera.get_frame(timeout=2.0)
        print(f"✓ Frame captured: {rgb.shape}\n")
    except Exception as e:
        print(f"✗ Failed to capture frame: {e}")
        camera.stop()
        return 1
    
    # Stop camera (we only need one frame)
    camera.stop()
    
    # Create mask interactively
    creator = MaskCreator(rgb, args.output)
    success = creator.run()
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        return 0
    else:
        return 1


if __name__ == '__main__':
    exit(main())
