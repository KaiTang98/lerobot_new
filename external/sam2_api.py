"""API wrapper for SAM2 integration with LeRobot processors.

This module provides two modes:
1. Camera Predictor Mode: For real-time tracking (like demo.py)
2. Image Predictor Mode: For single-frame segmentation

The camera predictor is stateful and tracks objects across frames,
which is ideal for robot teleoperation where you want consistent
fabric segmentation.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, List
import warnings

import numpy as np
from numpy.typing import NDArray
import torch

try:
    from sam2.build_sam import build_sam2_camera_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    build_sam2_camera_predictor = None
    warnings.warn(
        "SAM2 is not available. Make sure it's installed: "
        "cd external/sam2 && pip install -e ."
    )


class SAM2CameraRunner:
    """Stateful SAM2 runner for continuous camera tracking.
    
    This wrapper manages SAM2's camera predictor state, allowing
    consistent object tracking across frames. It's designed for
    use in robot teleoperation where you want to segment the same
    fabric/cloth object continuously.
    
    Usage:
        runner = SAM2CameraRunner(
            checkpoint="checkpoints/sam2.1_hiera_tiny.pt",
            model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
            initial_prompt_point=[320, 240],  # Click on fabric
        )
        
        # In control loop:
        mask = runner(rgb_frame)  # Returns (H, W) bool mask
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        model_cfg: str,
        initial_prompt_point: Optional[List[int]] = None,
        initial_prompt_box: Optional[List[int]] = None,
        device: str = "cuda",
    ):
        """Initialize SAM2 camera predictor.
        
        Args:
            checkpoint_path: Path to SAM2 checkpoint (.pt file)
            model_cfg: Model config path relative to sam2 package
                      (e.g., "configs/sam2.1/sam2.1_hiera_t.yaml")
            initial_prompt_point: [x, y] point on object for initial frame
            initial_prompt_box: [x1, y1, x2, y2] bounding box for initial frame
            device: "cuda" or "cpu"
        
        Note: You must provide either initial_prompt_point OR initial_prompt_box.
        """
        if not SAM2_AVAILABLE:
            raise RuntimeError("SAM2 is not available. Install it first.")
        
        if initial_prompt_point is None and initial_prompt_box is None:
            raise ValueError("Must provide either initial_prompt_point or initial_prompt_box")
        
        # Store device for later use in forward pass
        self.device = device
        
        # Enable TF32 for better performance (but not bfloat16 globally to avoid issues with other models)
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Build predictor
        self.predictor = build_sam2_camera_predictor(model_cfg, checkpoint_path, device=device)
        
        # Store initial prompt
        self.initial_prompt_point = initial_prompt_point
        self.initial_prompt_box = initial_prompt_box
        
        # State tracking
        self.initialized = False
        self.frame_count = 0
        self.obj_id = 1  # Track one object (the fabric)
    
    def reset(self):
        """Reset the predictor state (e.g., when starting a new episode)."""
        self.initialized = False
        self.frame_count = 0
        self.predictor.reset_state()
    
    def __call__(self, rgb_frame: NDArray[np.uint8]) -> NDArray[np.bool_]:
        """Run SAM2 segmentation on RGB frame.
        
        Args:
            rgb_frame: RGB image (H, W, 3) uint8
        
        Returns:
            Binary mask (H, W) bool
        """
        # Use bfloat16 only for SAM2 inference
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=(self.device == "cuda")):
            if not self.initialized:
                # First frame: initialize with prompt
                self.predictor.load_first_frame(rgb_frame)
                
                if self.initial_prompt_point is not None:
                    # Use point prompt
                    points = np.array([self.initial_prompt_point], dtype=np.float32)
                    labels = np.array([1], dtype=np.int32)  # 1 = foreground
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                        frame_idx=0,
                        obj_id=self.obj_id,
                        points=points,
                        labels=labels,
                    )
                else:
                    # Use box prompt
                    bbox = np.array(self.initial_prompt_box, dtype=np.float32).reshape(2, 2)
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                        frame_idx=0,
                        obj_id=self.obj_id,
                        bbox=bbox,
                    )
                
                self.initialized = True
            else:
                # Subsequent frames: track the object
                out_obj_ids, out_mask_logits = self.predictor.track(rgb_frame)
        
        self.frame_count += 1
        
        # Extract mask for our object (index 0 since we're tracking one object)
        if len(out_mask_logits) > 0:
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
            return mask.astype(bool)
        else:
            # No detection: return empty mask
            H, W = rgb_frame.shape[:2]
            return np.zeros((H, W), dtype=bool)


def create_sam2_camera_runner(
    checkpoint_path: str = "checkpoints/sam2.1_hiera_tiny.pt",
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
    initial_prompt_point: Optional[List[int]] = None,
    initial_prompt_box: Optional[List[int]] = None,
    device: str = "cuda",
) -> Callable[[NDArray[np.uint8]], NDArray[np.bool_]]:
    """Create a SAM2 camera runner callable for use in FabricPointCloudProcessorStep.
    
    This is the recommended way to use SAM2 with LeRobot processors.
    
    Args:
        checkpoint_path: Path to SAM2 checkpoint
        model_cfg: Model config file (relative to sam2 package)
        initial_prompt_point: [x, y] point to click on fabric (foreground)
        initial_prompt_box: [x1, y1, x2, y2] box around fabric
        device: "cuda" or "cpu"
    
    Returns:
        Callable that takes RGB (H, W, 3) uint8 and returns mask (H, W) bool
    
    Example:
        # Create runner with point prompt
        sam_runner = create_sam2_camera_runner(
            checkpoint_path="external/sam2/checkpoints/sam2.1_hiera_tiny.pt",
            model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
            initial_prompt_point=[320, 240],  # Center of fabric
        )
        
        # Use in processor
        processor = FabricPointCloudProcessorStep(
            ...,
            sam_runner=sam_runner,
        )
        
        # In control loop:
        mask = sam_runner(rgb_frame)  # Tracks fabric consistently
    
    Model Options (speed vs accuracy):
        - sam2.1_hiera_tiny.pt + configs/sam2.1/sam2.1_hiera_t.yaml (fastest, ~10ms)
        - sam2.1_hiera_small.pt + configs/sam2.1/sam2.1_hiera_s.yaml (fast, ~15ms)
        - sam2.1_hiera_base_plus.pt + configs/sam2.1/sam2.1_hiera_b+.yaml (balanced, ~25ms)
        - sam2.1_hiera_large.pt + configs/sam2.1/sam2.1_hiera_l.yaml (best, ~40ms)
    """
    runner = SAM2CameraRunner(
        checkpoint_path=checkpoint_path,
        model_cfg=model_cfg,
        initial_prompt_point=initial_prompt_point,
        initial_prompt_box=initial_prompt_box,
        device=device,
    )
    
    return runner


# Convenience function for quick testing
def test_sam2_runner(
    checkpoint_path: str = "checkpoints/sam2.1_hiera_tiny.pt",
    initial_prompt_point: List[int] = [320, 240],
    num_frames: int = 10,
) -> None:
    """Test SAM2 runner with synthetic data.
    
    Args:
        checkpoint_path: Path to SAM2 checkpoint
        initial_prompt_point: [x, y] initial point prompt
        num_frames: Number of frames to test
    """
    print("Creating SAM2 camera runner...")
    runner = create_sam2_camera_runner(
        checkpoint_path=checkpoint_path,
        model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
        initial_prompt_point=initial_prompt_point,
    )
    
    print(f"Testing with {num_frames} synthetic frames...")
    for i in range(num_frames):
        # Create synthetic RGB frame
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run SAM2
        mask = runner(rgb)
        
        print(f"Frame {i}: mask shape={mask.shape}, dtype={mask.dtype}, "
              f"pixels={mask.sum()}, initialized={runner.initialized}")
    
    print("✓ Test complete!")


if __name__ == "__main__":
    # Run test if executed directly
    test_sam2_runner()
