# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FabricPointCloudProcessorStep: projects masked RGB-D to a centered 3D pointcloud.

This processor takes RGB and aligned depth images, runs a segmentation mask (e.g. from SAM2),
and produces a centered, resampled pointcloud for downstream fabric manipulation tasks.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("fabric_pointcloud_processor")
@dataclass
class FabricPointCloudProcessorStep(ObservationProcessorStep):
    """Projects masked RGB-D to a centered 3D pointcloud.

    Given aligned RGB + depth images and a segmentation mask, this processor:
    1. Runs a segmentation function (sam_runner) on RGB to get a binary mask.
    2. Projects masked depth pixels to 3D camera coordinates using intrinsics.
    3. Centers the pointcloud (mean = 0).
    4. Resamples to a fixed number of points via random sampling or FPS.
    5. Writes the result to observation[output_key] as (N, 3) float32.

    Args:
        rgb_key: Key in observation dict containing RGB image (H, W, 3) uint8.
        depth_key: Key in observation dict containing aligned depth (H, W) uint16/float.
        fx: Focal length x (pixels).
        fy: Focal length y (pixels).
        cx: Principal point x (pixels).
        cy: Principal point y (pixels).
        depth_scale: Depth units to meters (e.g. 0.001 for mm, 0.00025 for L515).
        target_num_points: Number of points to resample to.
        sam_runner: Callable that takes RGB (H, W, 3) and returns binary mask (H, W) bool.
        output_key: Key to write the resulting pointcloud (default: "pcl").
        resample_method: "random" or "fps" (farthest point sampling). Default: "random".
    """

    rgb_key: str = "rgb"
    depth_key: str = "depth"
    fx: float = 600.974
    fy: float = 600.819
    cx: float = 331.946
    cy: float = 248.233
    depth_scale: float = 0.000250
    target_num_points: int = 1024
    sam_runner: Callable[[NDArray[np.uint8]], NDArray[np.bool_]] | None = None
    output_key: str = "pcl"
    resample_method: str = "random"

    # Internal state (not serialized)
    _intrinsics_matrix: NDArray[np.float32] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        # Pre-compute intrinsics matrix for efficiency
        self._intrinsics_matrix = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float32
        )

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Adds the pointcloud output feature to the observation features.

        This step adds a new observation key (output_key) with shape (target_num_points, 3).
        """
        # Add the pointcloud as a new observation feature
        if PipelineFeatureType.OBSERVATION in features:
            obs_features = features[PipelineFeatureType.OBSERVATION].copy()
            # Define the pointcloud feature with PCL type
            obs_features[self.output_key] = PolicyFeature(
                type=FeatureType.PCL,
                shape=(self.target_num_points, 3),
            )
            features = features.copy()
            features[PipelineFeatureType.OBSERVATION] = obs_features
        return features

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Process RGB-D to generate fabric pointcloud."""
        # 1) Extract RGB and depth
        rgb = observation[self.rgb_key]  # (H, W, 3) uint8
        depth = observation[self.depth_key]  # (H, W) uint16 or float

        if rgb is None or depth is None:
            raise ValueError(f"{self.__class__.__name__}: RGB or depth is None in observation.")

        # Ensure depth is 2D
        if depth.ndim != 2:
            raise ValueError(
                f"{self.__class__.__name__}: depth must be 2D (H, W), got shape {depth.shape}."
            )

        H, W = depth.shape

        # 2) Run segmentation to get fabric mask
        if self.sam_runner is None:
            raise ValueError(
                f"{self.__class__.__name__}: sam_runner is required but not provided. "
                "Pass a callable that returns a binary mask."
            )

        mask = self.sam_runner(rgb)  # (H, W) bool

        if mask.shape != (H, W):
            raise ValueError(
                f"{self.__class__.__name__}: sam_runner returned mask with shape {mask.shape}, "
                f"expected ({H}, {W})."
            )

        # 3) Project masked depth to 3D
        pcl = self._depth_to_pointcloud(depth, mask)

        if pcl.shape[0] == 0:
            raise ValueError(
                f"{self.__class__.__name__}: no valid points after masking. Check sam_runner output."
            )

        # 4) Center pointcloud
        pcl = self._center_pointcloud(pcl)

        # 5) Resample to target_num_points
        pcl = self._resample_pointcloud(pcl, self.target_num_points, method=self.resample_method)

        # 6) Write to observation
        observation[self.output_key] = pcl.astype(np.float32)

        return observation

    def _depth_to_pointcloud(self, depth: NDArray, mask: NDArray[np.bool_]) -> NDArray[np.float32]:
        """Project masked depth pixels to 3D camera coordinates.

        Args:
            depth: (H, W) depth map in device units.
            mask: (H, W) bool mask indicating valid pixels.

        Returns:
            (N, 3) array of XYZ points in meters (camera frame).
        """
        H, W = depth.shape

        # Get pixel coordinates of masked region
        v_coords, u_coords = np.nonzero(mask)  # row (y), col (x)

        if len(v_coords) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # Get corresponding depth values (in device units)
        z_device = depth[v_coords, u_coords]

        # Filter out zero/invalid depth
        valid = z_device > 0
        u_coords = u_coords[valid]
        v_coords = v_coords[valid]
        z_device = z_device[valid]

        if len(z_device) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # Convert depth to meters
        z_meters = z_device.astype(np.float32) * self.depth_scale

        # Backproject to 3D using pinhole model:
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = Z
        x = (u_coords.astype(np.float32) - self.cx) * z_meters / self.fx
        y = (v_coords.astype(np.float32) - self.cy) * z_meters / self.fy

        # Stack to (N, 3)
        pcl = np.stack([x, y, z_meters], axis=1)

        return pcl

    def _center_pointcloud(self, pcl: NDArray[np.float32]) -> NDArray[np.float32]:
        """Center pointcloud so mean = (0, 0, 0).

        Args:
            pcl: (N, 3) pointcloud.

        Returns:
            (N, 3) centered pointcloud.
        """
        if pcl.shape[0] == 0:
            return pcl

        centroid = pcl.mean(axis=0, keepdims=True)
        return pcl - centroid

    def _resample_pointcloud(
        self, pcl: NDArray[np.float32], num_points: int, method: str = "random"
    ) -> NDArray[np.float32]:
        """Resample pointcloud to a fixed number of points.

        Args:
            pcl: (N, 3) pointcloud.
            num_points: Target number of points.
            method: "random" or "fps" (farthest point sampling).

        Returns:
            (num_points, 3) resampled pointcloud.
        """
        N = pcl.shape[0]

        if N == 0:
            return np.zeros((num_points, 3), dtype=np.float32)

        if N == num_points:
            return pcl

        if N < num_points:
            # Upsample by repeating random points
            indices = np.random.choice(N, size=num_points, replace=True)
            return pcl[indices]

        # Downsample
        if method == "random":
            indices = np.random.choice(N, size=num_points, replace=False)
            return pcl[indices]
        elif method == "fps":
            # Simple farthest point sampling
            return self._farthest_point_sample(pcl, num_points)
        else:
            raise ValueError(f"Unknown resample_method: {method}. Use 'random' or 'fps'.")

    def _farthest_point_sample(self, pcl: NDArray[np.float32], num_points: int) -> NDArray[np.float32]:
        """Farthest point sampling (FPS) for downsampling.

        Args:
            pcl: (N, 3) pointcloud.
            num_points: Number of points to sample.

        Returns:
            (num_points, 3) sampled pointcloud.
        """
        N = pcl.shape[0]
        sampled_indices = np.zeros(num_points, dtype=np.int32)

        # Start with a random point
        current_idx = np.random.randint(0, N)
        sampled_indices[0] = current_idx

        # Track minimum distances from sampled set
        distances = np.full(N, np.inf, dtype=np.float32)

        for i in range(1, num_points):
            # Update distances to closest sampled point
            current_point = pcl[current_idx : current_idx + 1]  # (1, 3)
            dists_to_current = np.linalg.norm(pcl - current_point, axis=1)
            distances = np.minimum(distances, dists_to_current)

            # Select farthest point
            current_idx = np.argmax(distances)
            sampled_indices[i] = current_idx

        return pcl[sampled_indices]

    def get_config(self) -> dict[str, Any]:
        """Return serializable config (excludes sam_runner and internal state)."""
        return {
            "rgb_key": self.rgb_key,
            "depth_key": self.depth_key,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "depth_scale": self.depth_scale,
            "target_num_points": self.target_num_points,
            "output_key": self.output_key,
            "resample_method": self.resample_method,
            # sam_runner is not serializable; must be re-provided at load time
        }
