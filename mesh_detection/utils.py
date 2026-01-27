#!/usr/bin/env python
"""
Utility functions for mesh detection system.
"""

import time
from collections import deque
from typing import Optional

import numpy as np


class FPSCounter:
    """Rolling FPS counter with smoothing."""
    
    def __init__(self, window_size: int = 30):
        """Initialize FPS counter.
        
        Args:
            window_size: Number of samples to average over
        """
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.last_time = time.time()
    
    def tick(self) -> float:
        """Record a frame and return current FPS.
        
        Returns:
            Smoothed FPS value
        """
        current_time = time.time()
        self.timestamps.append(current_time)
        self.last_time = current_time
        
        if len(self.timestamps) < 2:
            return 0.0
        
        # Compute FPS from time deltas
        total_time = self.timestamps[-1] - self.timestamps[0]
        frame_count = len(self.timestamps) - 1
        
        if total_time > 0:
            return frame_count / total_time
        return 0.0


class Timer:
    """Simple context manager timer."""
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        """Initialize timer.
        
        Args:
            name: Name of operation being timed
            verbose: If True, print timing info
        """
        self.name = name
        self.verbose = verbose
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and optionally print."""
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"{self.name}: {self.elapsed*1000:.2f}ms")


def save_mesh_obj(vertices: np.ndarray, faces: Optional[np.ndarray], filepath: str) -> None:
    """Save mesh to OBJ file.
    
    Args:
        vertices: (N, 3) vertex coordinates
        faces: (F, 3) face indices (1-indexed) or None
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces if available
        if faces is not None:
            for face in faces:
                # OBJ uses 1-indexed faces
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"Saved mesh to {filepath}")


def save_pointcloud_ply(points: np.ndarray, colors: Optional[np.ndarray], filepath: str) -> None:
    """Save pointcloud to PLY file.
    
    Args:
        points: (N, 3) point coordinates
        colors: (N, 3) RGB colors (0-255) or None
        filepath: Output file path
    """
    n_points = len(points)
    
    with open(filepath, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Data
        for i in range(n_points):
            p = points[i]
            if colors is not None:
                c = colors[i]
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            else:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    
    print(f"Saved pointcloud to {filepath}")


if __name__ == "__main__":
    """Test utilities."""
    # Test FPS counter
    print("Testing FPS counter...")
    fps_counter = FPSCounter()
    for _ in range(10):
        time.sleep(0.033)  # ~30 FPS
        fps = fps_counter.tick()
        print(f"FPS: {fps:.1f}")
    
    # Test timer
    print("\nTesting timer...")
    with Timer("Sleep test"):
        time.sleep(0.1)
    
    # Test mesh save
    print("\nTesting mesh save...")
    vertices = np.random.randn(10, 3).astype(np.float32)
    faces = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32)
    save_mesh_obj(vertices, faces, "/tmp/test_mesh.obj")
    
    # Test pointcloud save
    print("\nTesting pointcloud save...")
    points = np.random.randn(100, 3).astype(np.float32)
    colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
    save_pointcloud_ply(points, colors, "/tmp/test_pcl.ply")
    
    print("\n✓ Utils test complete!")
