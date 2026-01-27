#!/usr/bin/env python
"""
Camera Manager for standalone mesh detection.

This module handles RealSense camera initialization and asynchronous frame capture.
Reuses logic from lerobot.cameras.realsense but in a standalone context.
"""

import queue
import threading
import time
from typing import Optional, Tuple

import numpy as np
import pyrealsense2 as rs


class CameraManager:
    """Manages RealSense L515 camera for RGB-D capture.
    
    Features:
    - Async frame capture in background thread
    - Thread-safe frame access via queue
    - Camera intrinsics and depth scale access
    - Graceful shutdown
    """
    
    def __init__(
        self,
        serial: str,
        color_width: int = 640,
        color_height: int = 480,
        depth_width: int = 640,
        depth_height: int = 480,
        fps: int = 30,
        use_depth: bool = True,
    ):
        """Initialize camera configuration (does not start capture yet).
        
        Args:
            serial: Camera serial number (e.g., "f1181599")
            color_width: Color frame width in pixels
            color_height: Color frame height in pixels
            depth_width: Depth frame width in pixels (L515: 320x240, 640x480, or 1024x768)
            depth_height: Depth frame height in pixels
            fps: Frames per second
            use_depth: If True, capture depth frames
        """
        self.serial = serial
        self.color_width = color_width
        self.color_height = color_height
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.fps = fps
        self.use_depth = use_depth
        
        # RealSense objects
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.profile: Optional[rs.pipeline_profile] = None
        
        # Camera properties
        self.camera_intrinsics: Optional[np.ndarray] = None
        self.depth_scale: Optional[float] = None
        
        # Async capture thread
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False
        
    def start(self) -> None:
        """Initialize camera and start async frame capture."""
        if self._is_running:
            print("Camera already running")
            return
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable device by serial number
        self.config.enable_device(self.serial)
        
        # Enable color stream
        self.config.enable_stream(
            rs.stream.color,
            self.color_width,
            self.color_height,
            rs.format.rgb8,
            self.fps
        )
        
        # Enable depth stream if requested
        if self.use_depth:
            self.config.enable_stream(
                rs.stream.depth,
                self.depth_width,
                self.depth_height,
                rs.format.z16,
                self.fps
            )
        
        # Start pipeline
        print(f"Starting RealSense camera {self.serial}...")
        self.profile = self.pipeline.start(self.config)
        
        # Get camera intrinsics
        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.camera_intrinsics = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Get depth scale if available
        if self.use_depth:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
        else:
            self.depth_scale = None
        
        print(f"Camera intrinsics:\n{self.camera_intrinsics}")
        if self.depth_scale:
            print(f"Depth scale: {self.depth_scale:.6f} m")
        
        # Warm up camera (discard first few frames)
        print("Warming up camera...")
        for _ in range(10):
            self.pipeline.wait_for_frames(timeout_ms=2000)
        
        # Start capture thread
        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self._is_running = True
        
        print(f"Camera {self.serial} started successfully")
    
    def _capture_loop(self) -> None:
        """Background thread that continuously captures frames."""
        align = rs.align(rs.stream.color) if self.use_depth else None
        
        while not self._stop_event.is_set():
            try:
                # Wait for frames
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                
                # Align depth to color if needed
                if self.use_depth and align:
                    frames = align.process(frames)
                
                # Get color frame
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                # Convert to numpy
                color_image = np.asanyarray(color_frame.get_data())
                
                # Get depth frame if available
                if self.use_depth:
                    depth_frame = frames.get_depth_frame()
                    if not depth_frame:
                        continue
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    # Resize depth to match color resolution if they differ
                    if depth_image.shape[:2] != color_image.shape[:2]:
                        import cv2
                        depth_image = cv2.resize(
                            depth_image, 
                            (color_image.shape[1], color_image.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                else:
                    depth_image = None
                
                # Put in queue (non-blocking, discard if full)
                try:
                    self._frame_queue.put_nowait((color_image, depth_image))
                except queue.Full:
                    # Discard oldest frame
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    # Try again
                    try:
                        self._frame_queue.put_nowait((color_image, depth_image))
                    except queue.Full:
                        pass
                
            except RuntimeError as e:
                if not self._stop_event.is_set():
                    print(f"Frame capture error: {e}")
                break
    
    def get_frame(self, timeout: float = 1.0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get the latest RGB-D frame.
        
        Args:
            timeout: Maximum time to wait for a frame in seconds
            
        Returns:
            Tuple of (rgb, depth) where:
            - rgb: (H, W, 3) uint8 array
            - depth: (H, W) uint16 array or None if use_depth=False
            
        Raises:
            TimeoutError: If no frame available within timeout
            RuntimeError: If camera not started
        """
        if not self._is_running:
            raise RuntimeError("Camera not started. Call start() first.")
        
        try:
            color, depth = self._frame_queue.get(timeout=timeout)
            return color, depth
        except queue.Empty:
            raise TimeoutError(f"No frame received within {timeout}s")
    
    def get_intrinsics(self) -> np.ndarray:
        """Get camera intrinsics matrix.
        
        Returns:
            (3, 3) float32 array: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            
        Raises:
            RuntimeError: If camera not started
        """
        if self.camera_intrinsics is None:
            raise RuntimeError("Camera not started. Call start() first.")
        return self.camera_intrinsics.copy()
    
    def get_depth_scale(self) -> float:
        """Get depth scale (depth units to meters).
        
        Returns:
            Depth scale factor (e.g., 0.00025 for L515)
            
        Raises:
            RuntimeError: If camera not started or depth not enabled
        """
        if not self.use_depth:
            raise RuntimeError("Depth not enabled")
        if self.depth_scale is None:
            raise RuntimeError("Camera not started. Call start() first.")
        return self.depth_scale
    
    def stop(self) -> None:
        """Stop camera capture and cleanup."""
        if not self._is_running:
            return
        
        print(f"Stopping camera {self.serial}...")
        
        # Signal thread to stop
        self._stop_event.set()
        
        # Wait for thread to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        
        # Stop pipeline
        if self.pipeline:
            self.pipeline.stop()
        
        self._is_running = False
        print("Camera stopped")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


if __name__ == "__main__":
    """Test camera manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test camera manager")
    parser.add_argument("--serial", type=str, default="f1181599", help="Camera serial")
    parser.add_argument("--duration", type=float, default=5.0, help="Test duration (s)")
    args = parser.parse_args()
    
    print(f"Testing camera {args.serial} for {args.duration}s...")
    
    with CameraManager(
        args.serial, 
        color_width=1920, 
        color_height=1080,
        depth_width=1024,
        depth_height=768,
        fps=30, 
        use_depth=True
    ) as camera:
        print(f"Intrinsics:\n{camera.get_intrinsics()}")
        print(f"Depth scale: {camera.get_depth_scale()}")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < args.duration:
            try:
                rgb, depth = camera.get_frame(timeout=1.0)
                frame_count += 1
                print(f"Frame {frame_count}: RGB {rgb.shape}, Depth {depth.shape if depth is not None else 'None'}")
            except TimeoutError:
                print("Frame timeout")
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        print(f"\nCaptured {frame_count} frames in {elapsed:.2f}s ({fps:.1f} FPS)")
