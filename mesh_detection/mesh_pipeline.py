#!/usr/bin/env python
"""
Mesh Processing Pipeline for standalone mesh detection.

This module chains SAM2 → PointCloud Generation → MeshGAT inference.
Reuses logic from lerobot processors but in a standalone context.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class MeshPipeline:
    """Process RGB-D frames through SAM2 → PointCloud → MeshGAT pipeline.
    
    Pipeline:
    1. SAM2 segments fabric from RGB
    2. Project masked depth to 3D pointcloud
    3. Center and resample pointcloud
    4. MeshGAT predicts mesh vertices
    """
    
    def __init__(
        self,
        sam2_checkpoint: str,
        sam2_model_cfg: str,
        meshgat_checkpoint: str,
        meshgat_config: str,
        camera_intrinsics: np.ndarray,
        depth_scale: float,
        initial_point: Optional[List[int]] = None,
        initial_box: Optional[List[int]] = None,
        meshgat_template: Optional[str] = None,
        target_num_points: int = 1024,
        resample_method: str = "random",
        device: str = "cuda",
        # Auto-detection mode
        auto_detect: bool = False,
        fabric_detector: Optional['FabricDetector'] = None,
        tracking_config: Optional[dict] = None,
    ):
        """Initialize the mesh processing pipeline.
        
        Args:
            sam2_checkpoint: Path to SAM2 checkpoint
            sam2_model_cfg: SAM2 model config (e.g., "configs/sam2.1/sam2.1_hiera_t.yaml")
            meshgat_checkpoint: Path to MeshGAT checkpoint
            meshgat_config: Path to MeshGAT config YAML
            camera_intrinsics: (3, 3) camera intrinsics matrix
            depth_scale: Depth units to meters conversion
            initial_point: [x, y] initial point on fabric for SAM2 (used if auto_detect=False)
            initial_box: [x1, y1, x2, y2] initial box around fabric for SAM2 (used if auto_detect=False)
            meshgat_template: Optional template mesh path
            target_num_points: Number of points in resampled pointcloud
            resample_method: "random" or "fps" (farthest point sampling)
            device: "cuda" or "cpu"
            auto_detect: If True, use fabric detector for SAM2 initialization
            fabric_detector: FabricDetector instance (required if auto_detect=True)
            tracking_config: Tracking configuration dict (for auto-detection mode)
        """
        self.device = device
        self.target_num_points = target_num_points
        self.resample_method = resample_method
        
        # Auto-detection mode
        self.auto_detect = auto_detect
        self.fabric_detector = fabric_detector
        self.tracking_config = tracking_config or {}
        
        # Store SAM2 parameters for re-initialization in auto-detect mode
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_model_cfg = sam2_model_cfg
        self.device = device
        
        # Tracking state (for auto-detection mode)
        if self.auto_detect:
            from dataclasses import dataclass, field
            from typing import List as ListType
            
            @dataclass
            class TrackingState:
                initialized: bool = False
                lost: bool = False
                frame_count: int = 0
                last_mask_area: int = 0
                lost_frame_count: int = 0
                last_point: Optional[tuple] = None
                point_history: ListType = field(default_factory=list)
                
                # Quality metrics
                mask_iou_history: ListType[float] = field(default_factory=list)
                mask_area_history: ListType[int] = field(default_factory=list)
                last_mask: Optional[np.ndarray] = None
            
            self.tracking_state = TrackingState()
            print(f"Auto-detection mode ENABLED")
        else:
            self.tracking_state = None
            print(f"Manual mode: using fixed initial point/box")
        
        # Camera parameters
        self.fx = float(camera_intrinsics[0, 0])
        self.fy = float(camera_intrinsics[1, 1])
        self.cx = float(camera_intrinsics[0, 2])
        self.cy = float(camera_intrinsics[1, 2])
        self.depth_scale = depth_scale
        
        # Add external paths
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        external_path = os.path.join(workspace_root, "external")
        sam2_path = os.path.join(workspace_root, "external", "sam2")
        meshgat_path = os.path.join(workspace_root, "external", "mesh_gat")
        
        for path in [external_path, sam2_path, meshgat_path]:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)
        
        # Initialize SAM2
        print("Initializing SAM2...")
        import sam2_api
        
        # In auto-detection mode, SAM2 starts with a dummy center point
        # It will be reinitialized with the detected fabric point on first frame
        # In manual mode, SAM2 uses provided point/box
        if self.auto_detect:
            # Use image center as placeholder (will be replaced on first frame)
            dummy_point = [320, 240]
            self.sam_runner = sam2_api.create_sam2_camera_runner(
                checkpoint_path=sam2_checkpoint,
                model_cfg=sam2_model_cfg,
                initial_prompt_point=dummy_point,
                initial_prompt_box=None,
                device=device,
            )
            print("SAM2 initialized (auto-detection mode - will reinitialize on first frame)")
        else:
            self.sam_runner = sam2_api.create_sam2_camera_runner(
                checkpoint_path=sam2_checkpoint,
                model_cfg=sam2_model_cfg,
                initial_prompt_point=initial_point,
                initial_prompt_box=initial_box,
                device=device,
            )
            print("SAM2 initialized (manual mode with fixed prompt)")
        
        # Initialize MeshGAT
        print("Initializing MeshGAT...")
        from api import load_meshgat_model
        self.meshgat_model, self.meshgat_cfg = load_meshgat_model(
            checkpoint_path=meshgat_checkpoint,
            config_path=meshgat_config,
            template_path=meshgat_template,
            device=device,
        )
        self.meshgat_model.eval()
        print("MeshGAT initialized")
        
        # Cache for last mask (for visualization)
        self._last_mask: Optional[np.ndarray] = None
    
    def process_frame(self, rgb: np.ndarray, depth: np.ndarray) -> Dict[str, np.ndarray]:
        """Process a single RGB-D frame through the full pipeline.
        
        Args:
            rgb: (H, W, 3) uint8 RGB image
            depth: (H, W) uint16 depth image
            
        Returns:
            Dictionary with keys:
            - 'mask': (H, W) bool segmentation mask
            - 'pointcloud': (N, 3) float32 centered pointcloud
            - 'mesh_vertices': (M, 3) float32 predicted mesh vertices
            - 'tracking_info': dict with tracking state (if auto_detect=True)
        """
        # Step 1: SAM2 Segmentation (with optional auto-detection/re-init)
        if self.auto_detect:
            mask, tracking_info = self._segment_fabric_auto(rgb, depth)
        else:
            mask = self._segment_fabric_manual(rgb)
            tracking_info = {}
        
        # Step 2: Generate PointCloud from masked depth
        pointcloud = self._depth_to_pointcloud(depth, mask)
        
        # Step 3: MeshGAT Inference
        mesh_vertices = self._predict_mesh(pointcloud)
        
        result = {
            'mask': mask,
            'pointcloud': pointcloud,
            'mesh_vertices': mesh_vertices,
        }
        
        if self.auto_detect:
            result['tracking_info'] = tracking_info
        
        return result
    
    def _segment_fabric_manual(self, rgb: np.ndarray) -> np.ndarray:
        """Run SAM2 segmentation with manual (fixed) prompt.
        
        Args:
            rgb: (H, W, 3) uint8
            
        Returns:
            mask: (H, W) bool numpy array
        """
        # SAM2 expects RGB uint8 and returns (H, W) bool mask directly
        mask = self.sam_runner(rgb)
        
        # Convert to numpy if it's a torch tensor
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        
        # Ensure it's a 2D boolean array
        if mask.ndim > 2:
            mask = mask.squeeze()
        
        mask = mask.astype(bool)
        self._last_mask = mask
        return mask
    
    def _segment_fabric_auto(self, rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Run SAM2 segmentation with auto-detection and tracking.
        
        Handles:
        - Initial fabric detection
        - SAM2 tracking
        - Loss detection
        - Auto re-initialization
        
        Args:
            rgb: (H, W, 3) uint8
            depth: (H, W) uint16
            
        Returns:
            Tuple of (mask, tracking_info)
            - mask: (H, W) bool numpy array
            - tracking_info: dict with tracking state
        """
        state = self.tracking_state
        state.frame_count += 1
        
        # Check if we need to initialize or re-initialize
        need_init = not state.initialized or state.lost
        
        if need_init:
            # Run fabric detector
            detection_result = self.fabric_detector.detect(rgb, depth)
            
            min_confidence = self.tracking_config.get('min_confidence', 0.3)
            if detection_result.point is not None and detection_result.confidence >= min_confidence:
                # Good detection - initialize SAM2
                print(f"[Frame {state.frame_count}] {'Initializing' if not state.initialized else 'Re-initializing'} SAM2 with point {detection_result.point}, confidence={detection_result.confidence:.2f}")
                
                # Re-create SAM2 runner with new point
                import sam2_api
                self.sam_runner = sam2_api.create_sam2_camera_runner(
                    checkpoint_path=self.sam2_checkpoint,
                    model_cfg=self.sam2_model_cfg,
                    initial_prompt_point=detection_result.point,
                    initial_prompt_box=None,
                    device=self.device,
                )
                
                state.initialized = True
                state.lost = False
                state.lost_frame_count = 0
                state.last_point = detection_result.point
                state.point_history = [detection_result.point]
            else:
                # Detection failed - return empty mask
                mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=bool)
                tracking_info = {
                    'initialized': state.initialized,
                    'lost': True,
                    'confidence': detection_result.confidence if detection_result else 0.0,
                    'message': 'Waiting for good detection...'
                }
                return mask, tracking_info
        
        # Run SAM2 tracking
        mask = self.sam_runner(rgb)
        
        # Convert to numpy
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        if mask.ndim > 2:
            mask = mask.squeeze()
        mask = mask.astype(bool)
        
        # Check mask quality
        mask_quality = self._check_mask_quality(mask, state)
        
        # Update tracking state
        state.last_mask = mask
        self._last_mask = mask
        
        # Prepare tracking info
        tracking_info = {
            'initialized': state.initialized,
            'lost': state.lost,
            'frame_count': state.frame_count,
            'mask_area': mask_quality['area'],
            'iou': mask_quality.get('iou', 1.0),
            'components': mask_quality['components'],
            'quality_score': mask_quality['quality_score'],
        }
        
        return mask, tracking_info
    
    def _check_mask_quality(self, mask: np.ndarray, state) -> dict:
        """Check SAM2 mask quality for loss detection.
        
        Args:
            mask: (H, W) bool mask
            state: TrackingState
            
        Returns:
            dict with quality metrics
        """
        import cv2
        
        area = np.sum(mask)
        
        # Count connected components
        num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8))
        components = num_labels - 1  # Exclude background
        
        # Compute IoU with previous mask
        iou = 1.0
        if state.last_mask is not None:
            intersection = np.sum(mask & state.last_mask)
            union = np.sum(mask | state.last_mask)
            iou = intersection / union if union > 0 else 0.0
        
        # Update history
        state.mask_area_history.append(area)
        state.mask_iou_history.append(iou)
        if len(state.mask_area_history) > 30:
            state.mask_area_history.pop(0)
            state.mask_iou_history.pop(0)
        
        # Check for loss conditions
        loss_detected = False
        
        # Initial area baseline (use first 10 frames average)
        if len(state.mask_area_history) >= 10 and state.last_mask_area == 0:
            state.last_mask_area = int(np.mean(state.mask_area_history[:10]))
        
        if state.last_mask_area > 0:
            area_ratio = area / state.last_mask_area
            min_ratio = self.tracking_config.get('min_mask_area_ratio', 0.3)
            
            if area_ratio < min_ratio:
                loss_detected = True
        
        # IoU check
        min_iou = self.tracking_config.get('min_mask_iou', 0.5)
        if iou < min_iou:
            loss_detected = True
        
        # Fragmentation check
        max_components = self.tracking_config.get('max_components', 50)
        if components > max_components:
            loss_detected = True
        
        # Update loss state
        if loss_detected:
            state.lost_frame_count += 1
            patience = self.tracking_config.get('loss_patience_frames', 5)
            if state.lost_frame_count >= patience:
                state.lost = True
                print(f"[Frame {state.frame_count}] Tracking lost (area_ratio={area/max(state.last_mask_area,1):.2f}, iou={iou:.2f}, components={components})")
        else:
            state.lost_frame_count = 0
        
        # Quality score (0-1)
        quality_score = min(1.0, (
            0.4 * min(1.0, area / max(state.last_mask_area, 1000)) +
            0.4 * iou +
            0.2 * (1.0 / (1.0 + components / 10.0))
        ))
        
        return {
            'area': int(area),
            'iou': float(iou),
            'components': int(components),
            'quality_score': float(quality_score),
            'loss_detected': loss_detected,
        }
    
    def _depth_to_pointcloud(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Project masked depth to 3D pointcloud.
        
        Args:
            depth: (H, W) uint16 depth in camera units
            mask: (H, W) bool segmentation mask
            
        Returns:
            pointcloud: (N, 3) float32 centered, resampled pointcloud
        """
        # Ensure depth is 2D numpy array
        if torch.is_tensor(depth):
            depth = depth.cpu().numpy()
        if depth.ndim > 2:
            depth = depth.squeeze()
        
        # Ensure mask is 2D boolean numpy array
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        if mask.ndim > 2:
            mask = mask.squeeze()
        mask = mask.astype(bool)
        
        H, W = depth.shape
        
        # Convert depth to meters
        depth_m = depth.astype(np.float32) * self.depth_scale
        
        # Get pixel coordinates of masked pixels
        v, u = np.where(mask)
        
        if len(u) == 0:
            # No valid points, return empty
            return np.zeros((0, 3), dtype=np.float32)
        
        # Get depth values at masked pixels (using integer indices)
        z = depth_m[v, u]
        
        # Filter out zero depth (returns boolean array)
        valid_mask = z > 0
        
        # Apply filter to get valid points
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = z[valid_mask]
        
        if len(z_valid) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        
        # Project to 3D camera coordinates
        x = (u_valid - self.cx) * z_valid / self.fx
        y = (v_valid - self.cy) * z_valid / self.fy
        
        # Stack to (N, 3)
        points = np.stack([x, y, z_valid], axis=1)
        
        # Center pointcloud (mean = 0)
        centroid = points.mean(axis=0)
        points_centered = points - centroid
        
        # Resample to target number of points
        points_resampled = self._resample_pointcloud(points_centered, self.target_num_points)
        
        return points_resampled.astype(np.float32)
    
    def _resample_pointcloud(self, points: np.ndarray, target_n: int) -> np.ndarray:
        """Resample pointcloud to fixed number of points.
        
        Args:
            points: (N, 3) pointcloud
            target_n: Target number of points
            
        Returns:
            resampled: (target_n, 3) pointcloud
        """
        n_points = len(points)
        
        if n_points == 0:
            return np.zeros((target_n, 3), dtype=np.float32)
        
        if n_points < target_n:
            # Upsample by repeating random points
            indices = np.random.choice(n_points, target_n, replace=True)
            return points[indices]
        elif n_points > target_n:
            # Downsample
            if self.resample_method == "random":
                indices = np.random.choice(n_points, target_n, replace=False)
                return points[indices]
            elif self.resample_method == "fps":
                # Farthest point sampling
                return self._farthest_point_sampling(points, target_n)
            else:
                raise ValueError(f"Unknown resample_method: {self.resample_method}")
        else:
            return points
    
    def _farthest_point_sampling(self, points: np.ndarray, n_samples: int) -> np.ndarray:
        """Farthest Point Sampling (FPS) for pointcloud downsampling.
        
        Args:
            points: (N, 3) pointcloud
            n_samples: Number of samples to select
            
        Returns:
            sampled: (n_samples, 3) pointcloud
        """
        n_points = len(points)
        sampled_indices = np.zeros(n_samples, dtype=np.int32)
        distances = np.full(n_points, np.inf)
        
        # Start with a random point
        current = np.random.randint(n_points)
        
        for i in range(n_samples):
            sampled_indices[i] = current
            current_point = points[current]
            
            # Update distances
            dists = np.linalg.norm(points - current_point, axis=1)
            distances = np.minimum(distances, dists)
            
            # Select farthest point
            current = np.argmax(distances)
        
        return points[sampled_indices]
    
    def _predict_mesh(self, pointcloud: np.ndarray) -> np.ndarray:
        """Run MeshGAT to predict mesh vertices from pointcloud.
        
        Args:
            pointcloud: (N, 3) float32 pointcloud
            
        Returns:
            mesh_vertices: (M, 3) float32 predicted mesh vertices
        """
        if len(pointcloud) == 0:
            # Return empty mesh
            return np.zeros((0, 3), dtype=np.float32)
        
        # Convert to torch tensor
        pcl_tensor = torch.from_numpy(pointcloud).float().to(self.device)
        
        # MeshGAT expects (1, N, 3) batch
        if pcl_tensor.dim() == 2:
            pcl_tensor = pcl_tensor.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            # Check MeshGAT config input_type
            if self.meshgat_cfg.input_type == "pointcloud":
                # MeshGAT expects dict input for pointcloud mode
                output = self.meshgat_model({"points": pcl_tensor})
            elif self.meshgat_cfg.input_type == "depth":
                # Depth mode expects tensor directly
                output = self.meshgat_model(pcl_tensor)
            else:
                raise ValueError(f"Unknown input_type: {self.meshgat_cfg.input_type}")
        
        # Extract vertices (output format depends on MeshGAT)
        # Assuming output is (1, M, 3) or (M, 3)
        if isinstance(output, dict):
            vertices = output['vertices']
        else:
            vertices = output
        
        # Convert to numpy
        if vertices.dim() == 3:
            vertices = vertices.squeeze(0)  # (M, 3)
        
        mesh_vertices = vertices.cpu().numpy().astype(np.float32)
        
        return mesh_vertices
    
    def reset_sam2_state(self) -> None:
        """Reset SAM2 tracking state (useful if tracking drifts)."""
        if hasattr(self.sam_runner, 'reset'):
            self.sam_runner.reset()
        print("SAM2 state reset")


if __name__ == "__main__":
    """Test mesh pipeline with dummy data."""
    print("Testing mesh pipeline...")
    
    # Dummy camera intrinsics
    intrinsics = np.array([
        [600.974, 0, 331.946],
        [0, 600.819, 248.233],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Create pipeline
    pipeline = MeshPipeline(
        sam2_checkpoint="external/sam2/checkpoints/sam2.1_hiera_tiny.pt",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
        meshgat_checkpoint="/home/ktang/ws/data/mesh_gat/collar_pcl/checkpoint/finalbestmodel_1999_0.00645.pt",
        meshgat_config="/home/ktang/ws/data/mesh_gat/collar_pcl/checkpoint/config.yaml",
        camera_intrinsics=intrinsics,
        depth_scale=0.00025,
        initial_point=[320, 240],
        target_num_points=1024,
        device="cuda",
    )
    
    # Create dummy RGB-D frame
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.random.randint(1000, 5000, (480, 640), dtype=np.uint16)
    
    print("\nProcessing dummy frame...")
    result = pipeline.process_frame(rgb, depth)
    
    print(f"Mask shape: {result['mask'].shape}")
    print(f"Pointcloud shape: {result['pointcloud'].shape}")
    print(f"Mesh vertices shape: {result['mesh_vertices'].shape}")
    print("\n✓ Pipeline test complete!")
