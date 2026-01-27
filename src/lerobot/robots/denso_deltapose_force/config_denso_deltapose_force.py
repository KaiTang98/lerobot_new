#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("denso_deltapose_force")
@dataclass(kw_only=True)
class DensoDeltaPoseForceConfig(RobotConfig):
    """Config for Denso delta-pose and force client proxied via a Windows PC.

    The Linux client connects over TCP to a Windows host that performs the
    low-level robot control and state reporting. This config provides the
    network endpoint and optional cameras attached to the Linux client.
    """

    # Remote Windows server address
    server_ip: str = "192.168.2.105"
    server_port: int = 12345

    # Optional: target loop rate for local timing/teleop assumptions
    fps: int | None = 100

    # Optional local cameras to record alongside robot state
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # ============ MeshGAT Integration (Optional) ============
    # Enable SAM2 + FabricPointCloud + MeshGAT pipeline for fabric mesh prediction
    enable_meshgat: bool = False

    # Required if enable_meshgat=True
    meshgat_checkpoint_path: str | None = None
    meshgat_config_path: str | None = None
    sam2_checkpoint_path: str | None = None

    # SAM2 initial prompt (provide either point OR box)
    sam2_initial_point: list[int] | None = None  # [x, y] e.g., [320, 240]
    sam2_initial_box: list[int] | None = None  # [x1, y1, x2, y2]

    # Optional MeshGAT parameters
    meshgat_template_path: str | None = None
    meshgat_device: str = "cuda"
    meshgat_target_num_points: int = 1024
    meshgat_expected_num_vertices: int | None = None  # Expected mesh size for dataset consistency
    meshgat_input_key: str = "pcl"
    meshgat_output_key: str = "mesh_vertices"

    # Camera to use for RGB-D input (must be a RealSense with depth enabled)
    meshgat_camera_key: str | None = None  # e.g., "camera_l515"

    def __post_init__(self):
        super().__post_init__()

        # Validate MeshGAT configuration if enabled
        if self.enable_meshgat:
            if not self.meshgat_checkpoint_path:
                raise ValueError("enable_meshgat=True requires meshgat_checkpoint_path")
            if not self.meshgat_config_path:
                raise ValueError("enable_meshgat=True requires meshgat_config_path")
            if not self.sam2_checkpoint_path:
                raise ValueError("enable_meshgat=True requires sam2_checkpoint_path")
            if not self.sam2_initial_point and not self.sam2_initial_box:
                raise ValueError(
                    "enable_meshgat=True requires either sam2_initial_point or sam2_initial_box"
                )
            if not self.meshgat_camera_key:
                raise ValueError(
                    "enable_meshgat=True requires meshgat_camera_key to specify which camera provides depth"
                )
            if self.meshgat_camera_key not in self.cameras:
                raise ValueError(
                    f"meshgat_camera_key='{self.meshgat_camera_key}' not found in cameras: {list(self.cameras.keys())}"
                )

            # Validate camera has depth capability
            camera_cfg = self.cameras[self.meshgat_camera_key]
            if camera_cfg.type != "intelrealsense":
                raise ValueError(f"MeshGAT requires RealSense camera with depth, got {camera_cfg.type}")
            if not getattr(camera_cfg, "use_depth", False):
                raise ValueError(
                    f"MeshGAT camera '{self.meshgat_camera_key}' must have use_depth=True"
                )
