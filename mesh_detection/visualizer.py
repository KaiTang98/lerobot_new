#!/usr/bin/env python
"""
Visualization module for standalone mesh detection.

Provides 2D (OpenCV) and 3D (Open3D) visualization of results.
"""

import time
from typing import Optional

import cv2
import numpy as np

# Open3D import is optional (for 3D visualization)
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. 3D visualization disabled.")


class MeshVisualizer:
    """Real-time visualization of mesh detection results.
    
    Features:
    - 2D: RGB with mask overlay, FPS counter
    - 3D: Mesh and pointcloud visualization (if Open3D available)
    """
    
    def __init__(
        self,
        window_name: str = "Mesh Detection",
        enable_3d: bool = True,
        mesh_color: tuple = (0.8, 0.2, 0.2),
        pcl_color: tuple = (0.2, 0.8, 0.2),
        show_original_rgb: bool = False,
        show_original_depth: bool = False,
        show_filtered_depth: bool = False,
        depth_scale: float = 0.00025,
    ):
        """Initialize visualizer.
        
        Args:
            window_name: Base name for windows
            enable_3d: Enable Open3D 3D visualization
            mesh_color: RGB color for mesh (0-1 range)
            pcl_color: RGB color for pointcloud (0-1 range)
            show_original_rgb: Show separate window with original RGB
            show_original_depth: Show separate window with original depth
            show_filtered_depth: Show separate window with filtered depth (after depth_min/max)
            depth_scale: Depth units to meters conversion factor (e.g., 0.00025 for L515)
        """
        self.window_name = window_name
        self.enable_3d = enable_3d and OPEN3D_AVAILABLE
        self.mesh_color = mesh_color
        self.pcl_color = pcl_color
        self.show_original_rgb = show_original_rgb
        self.show_original_depth = show_original_depth
        self.show_filtered_depth = show_filtered_depth
        self.depth_scale = depth_scale
        
        # 2D windows
        self._2d_window_created = False
        self._rgb_window_created = False
        self._depth_window_created = False
        self._filtered_depth_window_created = False
        
        # 3D visualization (Open3D)
        self._vis: Optional['o3d.visualization.Visualizer'] = None
        self._pcl_geom: Optional['o3d.geometry.PointCloud'] = None
        self._mesh_geom: Optional['o3d.geometry.TriangleMesh'] = None
        self._axes_geom: Optional['o3d.geometry.TriangleMesh'] = None
        
        # FPS tracking
        self._fps_history = []
        self._last_time = time.time()
        
        # Status
        self._should_close = False
        self._paused = False
    
    def update_2d(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray] = None,
        fps: Optional[float] = None,
        status_text: Optional[str] = None,
    ) -> None:
        """Update 2D OpenCV window with RGB + mask overlay.
        
        Args:
            rgb: (H, W, 3) uint8 RGB image
            mask: (H, W) bool mask (optional)
            fps: FPS value to display (optional)
            status_text: Additional status text (optional)
        """
        # Create window on first call
        if not self._2d_window_created:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self._2d_window_created = True
            except cv2.error as e:
                print(f"Warning: Could not create OpenCV window: {e}")
                print("2D visualization disabled")
                self._2d_window_created = None  # Mark as failed
                return
        elif self._2d_window_created is None:
            # Previously failed to create window
            return
        
        # Convert RGB to BGR for OpenCV
        display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Overlay mask if provided
        if mask is not None:
            overlay = display.copy()
            overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
            display = overlay
        
        # Add FPS text
        if fps is not None:
            cv2.putText(
                display,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
        
        # Add status text
        if status_text:
            cv2.putText(
                display,
                status_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
        
        # Add controls help
        help_text = "Q:Quit SPACE:Pause R:Reset S:Save"
        cv2.putText(
            display,
            help_text,
            (10, display.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        
        # Show
        try:
            cv2.imshow(self.window_name, display)
        except cv2.error:
            pass  # Window closed
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            self._should_close = True
        elif key == ord(' '):  # SPACE
            self._paused = not self._paused
            print(f"{'Paused' if self._paused else 'Resumed'}")
        elif key == ord('r'):  # R
            print("Reset requested")
            # This can be handled by main loop
        elif key == ord('s'):  # S
            print("Save requested")
            # This can be handled by main loop
    
    def get_2d_frame(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray] = None,
        status_text: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Create visualization frame without displaying it (for video recording).
        
        Args:
            rgb: (H, W, 3) uint8 RGB image
            mask: (H, W) bool mask (optional)
            status_text: Additional status text (optional)
            
        Returns:
            (H, W, 3) uint8 RGB visualization frame (not BGR!)
        """
        # Create display frame
        display = rgb.copy()
        
        # Overlay mask if provided
        if mask is not None:
            overlay = display.copy()
            overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
            display = overlay
        
        # Add status text
        if status_text:
            cv2.putText(
                display,
                status_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
        
        # Add controls help
        help_text = "Q:Quit SPACE:Pause R:Reset S:Save"
        cv2.putText(
            display,
            help_text,
            (10, display.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        
        return display
    
    def update_original_rgb(self, rgb: np.ndarray) -> None:
        """Show original (full-resolution) RGB image in a separate window.
        
        Args:
            rgb: (H, W, 3) uint8 RGB image
        """
        if not self.show_original_rgb:
            return
            
        # Create window on first call
        if not self._rgb_window_created:
            try:
                cv2.namedWindow("Original RGB", cv2.WINDOW_NORMAL)
                self._rgb_window_created = True
            except cv2.error as e:
                print(f"Warning: Could not create Original RGB window: {e}")
                self._rgb_window_created = None
                return
        elif self._rgb_window_created is None:
            return
        
        # Convert RGB to BGR for OpenCV
        display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Add resolution info
        cv2.putText(
            display,
            f"Original RGB: {rgb.shape[1]}x{rgb.shape[0]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        
        try:
            cv2.imshow("Original RGB", display)
        except cv2.error:
            pass
    
    def update_original_depth(self, depth: np.ndarray) -> None:
        """Show original (full-resolution) depth image in a separate window.
        
        Args:
            depth: (H, W) uint16 or float32 depth (raw units or meters)
        """
        if not self.show_original_depth:
            return
            
        # Create window on first call
        if not self._depth_window_created:
            try:
                cv2.namedWindow("Original Depth", cv2.WINDOW_NORMAL)
                self._depth_window_created = True
            except cv2.error as e:
                print(f"Warning: Could not create Original Depth window: {e}")
                self._depth_window_created = None
                return
        elif self._depth_window_created is None:
            return
        
        # Convert raw depth to meters if needed
        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) * self.depth_scale
        else:
            depth_m = depth
        
        # Normalize depth for visualization (clip to reasonable range)
        depth_vis = np.clip(depth_m, 0, 2.0)  # 0-2 meters
        depth_vis = (depth_vis / 2.0 * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        # Add info
        valid_depth = depth_m[depth_m > 0]
        if len(valid_depth) > 0:
            depth_min, depth_max, depth_mean = valid_depth.min(), valid_depth.max(), valid_depth.mean()
            info_text = f"Depth: {depth_m.shape[1]}x{depth_m.shape[0]} | Min:{depth_min:.3f} Max:{depth_max:.3f} Mean:{depth_mean:.3f}m"
        else:
            info_text = f"Depth: {depth_m.shape[1]}x{depth_m.shape[0]} | No valid depth"
        
        cv2.putText(
            depth_vis,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        
        try:
            cv2.imshow("Original Depth", depth_vis)
        except cv2.error:
            pass
    
    def update_filtered_depth(self, depth: np.ndarray, depth_min: float, depth_max: float) -> None:
        """Show filtered depth after applying depth_min/max range filtering.
        
        Args:
            depth: (H, W) uint16 or float32 depth (raw units or meters)
            depth_min: Minimum depth threshold in meters
            depth_max: Maximum depth threshold in meters
        """
        if not self.show_filtered_depth:
            return
            
        # Create window on first call
        if not self._filtered_depth_window_created:
            try:
                cv2.namedWindow("Filtered Depth", cv2.WINDOW_NORMAL)
                self._filtered_depth_window_created = True
            except cv2.error as e:
                print(f"Warning: Could not create Filtered Depth window: {e}")
                self._filtered_depth_window_created = None
                return
        elif self._filtered_depth_window_created is None:
            return
        
        # Convert raw depth to meters if needed
        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) * self.depth_scale
        else:
            depth_m = depth
        
        # Apply depth filtering - set out-of-range pixels to 0
        depth_filtered = depth_m.copy()
        depth_filtered[(depth_m < depth_min) | (depth_m > depth_max)] = 0
        
        # Normalize depth for visualization
        # Scale valid depth range [depth_min, depth_max] to colormap range [0.1, 0.9]
        # This gives better contrast for the filtered range
        depth_vis = np.zeros_like(depth_filtered)
        valid_mask = depth_filtered > 0
        if valid_mask.any():
            # Map [depth_min, depth_max] -> [0.1, 0.9] of the 2m range
            # So depth_min shows as blue-green, depth_max shows as yellow-red
            depth_vis[valid_mask] = 0.2 + (depth_filtered[valid_mask] - depth_min) / (depth_max - depth_min) * 1.6
        
        depth_vis = np.clip(depth_vis, 0, 2.0)  # 0-2 meters
        depth_vis = (depth_vis / 2.0 * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        # Get statistics for all valid depth (to show what's available)
        all_valid_depth = depth_m[depth_m > 0]
        
        # Get statistics for filtered depth
        valid_depth = depth_filtered[depth_filtered > 0]
        if len(valid_depth) > 0:
            d_min, d_max, d_mean = valid_depth.min(), valid_depth.max(), valid_depth.mean()
            valid_pixels = len(valid_depth)
            total_pixels = depth.shape[0] * depth.shape[1]
            percent_valid = (valid_pixels / total_pixels) * 100
            info_text = f"Filtered Depth [{depth_min:.2f}-{depth_max:.2f}m]: {depth.shape[1]}x{depth.shape[0]}"
            info_text2 = f"Valid: {valid_pixels}/{total_pixels} ({percent_valid:.1f}%) | Min:{d_min:.3f} Max:{d_max:.3f} Mean:{d_mean:.3f}m"
        else:
            info_text = f"Filtered Depth [{depth_min:.2f}-{depth_max:.2f}m]: {depth.shape[1]}x{depth.shape[0]}"
            if len(all_valid_depth) > 0:
                scene_min, scene_max, scene_mean = all_valid_depth.min(), all_valid_depth.max(), all_valid_depth.mean()
                info_text2 = f"NO VALID DEPTH IN RANGE! Scene depth: {scene_min:.3f}-{scene_max:.3f}m (mean:{scene_mean:.3f}m)"
            else:
                info_text2 = f"NO VALID DEPTH! Camera may not be getting depth data"
        
        cv2.putText(
            depth_vis,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            depth_vis,
            info_text2,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255) if len(valid_depth) == 0 else (255, 255, 255),  # Red if no valid depth
            2,
        )
        
        try:
            cv2.imshow("Filtered Depth", depth_vis)
        except cv2.error:
            pass
    
    def update_3d(
        self,
        pointcloud: Optional[np.ndarray] = None,
        mesh_vertices: Optional[np.ndarray] = None,
        mesh_faces: Optional[np.ndarray] = None,
    ) -> None:
        """Update 3D Open3D window with pointcloud and mesh.
        
        Args:
            pointcloud: (N, 3) float32 pointcloud
            mesh_vertices: (M, 3) float32 mesh vertices
            mesh_faces: (F, 3) int32 mesh faces (optional)
        """
        if not self.enable_3d:
            return
        
        # Initialize visualizer on first call
        if self._vis is None:
            self._vis = o3d.visualization.Visualizer()
            self._vis.create_window(window_name=f"{self.window_name} - 3D", width=800, height=600)
            
            # Add coordinate axes
            self._axes_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self._vis.add_geometry(self._axes_geom)
            
            # Setup view
            opt = self._vis.get_render_option()
            opt.background_color = np.array([0.1, 0.1, 0.1])
            opt.point_size = 2.0
            opt.show_coordinate_frame = True
        
        # Update pointcloud
        if pointcloud is not None and len(pointcloud) > 0:
            if self._pcl_geom is None:
                self._pcl_geom = o3d.geometry.PointCloud()
                self._pcl_geom.points = o3d.utility.Vector3dVector(pointcloud)
                self._pcl_geom.paint_uniform_color(self.pcl_color)
                self._vis.add_geometry(self._pcl_geom)
            else:
                self._pcl_geom.points = o3d.utility.Vector3dVector(pointcloud)
                self._vis.update_geometry(self._pcl_geom)
        
        # Update mesh
        if mesh_vertices is not None and len(mesh_vertices) > 0:
            if self._mesh_geom is None:
                self._mesh_geom = o3d.geometry.TriangleMesh()
                self._mesh_geom.vertices = o3d.utility.Vector3dVector(mesh_vertices)
                if mesh_faces is not None:
                    self._mesh_geom.triangles = o3d.utility.Vector3iVector(mesh_faces)
                self._mesh_geom.paint_uniform_color(self.mesh_color)
                self._mesh_geom.compute_vertex_normals()
                self._vis.add_geometry(self._mesh_geom)
            else:
                self._mesh_geom.vertices = o3d.utility.Vector3dVector(mesh_vertices)
                if mesh_faces is not None:
                    self._mesh_geom.triangles = o3d.utility.Vector3iVector(mesh_faces)
                self._mesh_geom.compute_vertex_normals()
                self._vis.update_geometry(self._mesh_geom)
        
        # Poll events and render
        self._vis.poll_events()
        self._vis.update_renderer()
        
        # Check if window was closed
        if not self._vis.poll_events():
            self._should_close = True
    
    def compute_fps(self) -> float:
        """Compute current FPS based on frame times.
        
        Returns:
            FPS value
        """
        current_time = time.time()
        dt = current_time - self._last_time
        self._last_time = current_time
        
        if dt > 0:
            fps = 1.0 / dt
            self._fps_history.append(fps)
            
            # Keep last 30 samples
            if len(self._fps_history) > 30:
                self._fps_history.pop(0)
            
            # Return smoothed FPS
            return np.mean(self._fps_history)
        
        return 0.0
    
    def should_close(self) -> bool:
        """Check if user requested to close.
        
        Returns:
            True if should close
        """
        return self._should_close
    
    def is_paused(self) -> bool:
        """Check if visualization is paused.
        
        Returns:
            True if paused
        """
        return self._paused
    
    def close(self) -> None:
        """Close all windows and cleanup."""
        print("Closing visualizer...")
        
        # Close OpenCV window
        if self._2d_window_created:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        # Close Open3D window
        if self._vis is not None:
            self._vis.destroy_window()
        
        print("Visualizer closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    """Test visualizer with dummy data."""
    print("Testing visualizer (press Q to quit)...")
    
    with MeshVisualizer(enable_3d=True) as viz:
        frame_count = 0
        
        while not viz.should_close() and frame_count < 100:
            # Create dummy data
            rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            mask = np.random.rand(480, 640) > 0.7
            pcl = np.random.randn(1024, 3).astype(np.float32) * 0.1
            mesh = np.random.randn(442, 3).astype(np.float32) * 0.15
            
            # Update visualizations
            fps = viz.compute_fps()
            viz.update_2d(rgb, mask, fps, f"Frame {frame_count}")
            viz.update_3d(pcl, mesh)
            
            frame_count += 1
            time.sleep(0.033)  # ~30 FPS
        
        print(f"\nRendered {frame_count} frames")
