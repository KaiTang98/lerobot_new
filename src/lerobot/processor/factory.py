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

import numpy as np

from lerobot.robots import denso_windows, denso_deltapose
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators import quest_haptics
from lerobot.teleoperators.config import TeleoperatorConfig
from .converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from .core import RobotAction, RobotObservation
from .pipeline import IdentityProcessorStep, RobotProcessorPipeline
from .quest_haptics_denso_windows_robot_action_step import QuestHapticsDensoWindowsRobotActionStep
from .quest_haptics_denso_deltapose_robot_action_step import QuestHapticsDensoDeltaPoseRobotActionStep
from.quest_haptics_denso_deltapose_force_robot_action_step import QuestHapticsDensoDeltaPoseForceRobotActionStep
from .denso_deltapose_teleop_fusion_step import DensoDeltaPoseTeleopFusionStep
from .denso_deltapose_strip_remote_action_step import DensoDeltaPoseStripRemoteActionStep
from .mesh_gat_processor import MeshGATObservationProcessorStep


# ---------------------------------------------
# ------ robot observation processor ----------
# ---------------------------------------------
def make_default_robot_observation_processor() -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor


def make_questhaptics_densowindows_robot_observation_processor() -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor

def make_questhaptics_densodeltapose_robot_observation_processor() -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    # Strip _last_remote_action so processed obs is clean (state + cameras only).
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[DensoDeltaPoseStripRemoteActionStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor

def make_questhaptics_densodeltapose_force_robot_observation_processor() -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    # Strip _last_remote_action so processed obs is clean (state + cameras only).
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[DensoDeltaPoseStripRemoteActionStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor


def make_denso_meshgat_robot_observation_processor(
    checkpoint_path: str,
    config_path: str,
    template_path: str | None = None,
    device: str = "cuda",
    input_key: str = "pcl",
    output_key: str = "mesh_vertices",
    # NEW: SAM2 and FabricPointCloud parameters
    camera_intrinsics: np.ndarray | None = None,
    depth_scale: float | None = None,
    sam2_checkpoint: str | None = None,
    sam2_initial_point: list[int] | None = None,
    sam2_initial_box: list[int] | None = None,
    target_num_points: int = 1024,
    enable_pointcloud: bool = False,
) -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    """Robot observation processor for Denso robots with MeshGAT inference.

    This helper composes the Denso obs cleanup step with optional
    FabricPointCloudProcessorStep (if enable_pointcloud=True) and
    MeshGATObservationProcessorStep.

    Args:
        checkpoint_path: Path to MeshGAT checkpoint
        config_path: Path to MeshGAT config
        template_path: Optional MeshGAT template path
        device: Device for inference ("cuda" or "cpu")
        input_key: Input key for MeshGAT ("pcl" or "depth")
        output_key: Output key for mesh vertices
        camera_intrinsics: (3, 3) intrinsics matrix [fx 0 cx; 0 fy cy; 0 0 1]
        depth_scale: Depth unit to meters (e.g., 0.00025 for L515)
        sam2_checkpoint: Path to SAM2 checkpoint (if enable_pointcloud=True)
        sam2_initial_point: [x, y] initial point on fabric (if enable_pointcloud=True)
        sam2_initial_box: [x1, y1, x2, y2] box around fabric (if enable_pointcloud=True)
        target_num_points: Number of points in output pointcloud
        enable_pointcloud: If True, add FabricPointCloudProcessorStep before MeshGAT

    Returns:
        RobotProcessorPipeline for observation processing
    
    Examples:
        # Mode 1: MeshGAT with pre-computed pointcloud (input_key="pcl")
        processor = make_denso_meshgat_robot_observation_processor(
            checkpoint_path="path/to/meshgat.pt",
            config_path="path/to/config.yaml",
            input_key="pcl",  # Expects obs["pcl"] already exists
        )
        
        # Mode 2: Full pipeline with SAM2 + FabricPointCloud + MeshGAT
        processor = make_denso_meshgat_robot_observation_processor(
            checkpoint_path="path/to/meshgat.pt",
            config_path="path/to/config.yaml",
            enable_pointcloud=True,
            camera_intrinsics=camera.config.camera_intrinsics,
            depth_scale=camera.config.depth_scale,
            sam2_checkpoint="external/sam2/checkpoints/sam2.1_hiera_tiny.pt",
            sam2_initial_point=[320, 240],  # Point on fabric
            target_num_points=1024,
        )
    """
    import numpy as np
    
    steps = [DensoDeltaPoseStripRemoteActionStep()]
    
    # Optionally add FabricPointCloudProcessorStep
    if enable_pointcloud:
        if camera_intrinsics is None or depth_scale is None:
            raise ValueError(
                "enable_pointcloud=True requires camera_intrinsics and depth_scale"
            )
        if sam2_checkpoint is None:
            raise ValueError("enable_pointcloud=True requires sam2_checkpoint")
        if sam2_initial_point is None and sam2_initial_box is None:
            raise ValueError(
                "enable_pointcloud=True requires either sam2_initial_point or sam2_initial_box"
            )
        
        # Import here to avoid circular imports
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../external/sam2"))
        
        from external.sam2_api import create_sam2_camera_runner
        from .fabric_pointcloud_processor import FabricPointCloudProcessorStep
        
        # Create SAM2 runner
        sam_runner = create_sam2_camera_runner(
            checkpoint_path=sam2_checkpoint,
            model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
            initial_prompt_point=sam2_initial_point,
            initial_prompt_box=sam2_initial_box,
            device=device,
        )
        
        # Extract intrinsics
        if isinstance(camera_intrinsics, np.ndarray):
            fx = float(camera_intrinsics[0, 0])
            fy = float(camera_intrinsics[1, 1])
            cx = float(camera_intrinsics[0, 2])
            cy = float(camera_intrinsics[1, 2])
        else:
            # Assume list of lists
            fx = float(camera_intrinsics[0][0])
            fy = float(camera_intrinsics[1][1])
            cx = float(camera_intrinsics[0][2])
            cy = float(camera_intrinsics[1][2])
        
        # Add FabricPointCloud step
        steps.append(
            FabricPointCloudProcessorStep(
                rgb_key="rgb",
                depth_key="depth",
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                depth_scale=depth_scale,
                target_num_points=target_num_points,
                sam_runner=sam_runner,
                output_key="pcl",
                resample_method="random",
            )
        )
    
    # Add MeshGAT step
    steps.append(
        MeshGATObservationProcessorStep(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            template_path=template_path,
            device=device,
            input_key=input_key,
            output_key=output_key,
        )
    )

    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=steps,
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor


# ---------------------------------------------
# ------- teleoperation action processor ------
# ---------------------------------------------
def make_default_teleop_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return teleop_action_processor


def make_questhaptics_densowindows_teleop_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return teleop_action_processor

def make_questhaptics_densodeltapose_teleop_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    # Teleop pipeline: first compute deltapose_* from Quest inputs, then fuse obs-derived intime fields.
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[QuestHapticsDensoDeltaPoseRobotActionStep(scale_mm=1000.0, deadzone=1.0),
               DensoDeltaPoseTeleopFusionStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return teleop_action_processor

def make_questhaptics_densodeltapose_force_teleop_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    # Teleop pipeline: first compute deltapose_* from Quest inputs, then fuse obs-derived intime fields.
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[QuestHapticsDensoDeltaPoseForceRobotActionStep(scale_mm=1000.0, deadzone=1.0),
               DensoDeltaPoseTeleopFusionStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return teleop_action_processor


# ---------------------------------------------
# ------- robot action processor --------------
# ---------------------------------------------
def make_default_robot_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return robot_action_processor

def make_questhaptics_densowindows_robot_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    robot_action_processor  = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[QuestHapticsDensoWindowsRobotActionStep(clamp=1.0, deadzone=1.0)],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return robot_action_processor

def make_questhaptics_densodeltapose_robot_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    # Robot action processor can be identity; conversion is done in teleop pipeline.
    robot_action_processor  = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return robot_action_processor

def make_questhaptics_densodeltapose_force_robot_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    # Robot action processor can be identity; conversion is done in teleop pipeline.
    robot_action_processor  = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return robot_action_processor

# ---------------------------------------------
# ------ teleop-robot processor factory -------
# ---------------------------------------------
def make_teleop_robot_processors(robotConfig: RobotConfig,
                                 teleopConfig: TeleoperatorConfig | None):

    # Determine teleop processors
    if teleopConfig is not None and teleopConfig.type == "bi_quest_haptics" and robotConfig.type == "denso_windows":
        return make_questhaptics_densowindows_processor()
    if teleopConfig is not None and teleopConfig.type == "bi_quest_haptics" and robotConfig.type == "denso_deltapose":
        return make_questhaptics_densodeltapose_processor()
    if teleopConfig is not None and teleopConfig.type == "bi_quest_haptics" and robotConfig.type == "denso_deltapose_force":
        return make_questhaptics_densodeltapose_force_processor()
    else:
        return make_default_processors()

def make_default_processors():
    teleop_action_processor = make_default_teleop_action_processor()
    robot_action_processor = make_default_robot_action_processor()
    robot_observation_processor = make_default_robot_observation_processor()
    return (teleop_action_processor, robot_action_processor, robot_observation_processor)


def make_questhaptics_densowindows_processor():

    robot_observation_processor = make_questhaptics_densowindows_robot_observation_processor()
    teleop_action_processor = make_questhaptics_densowindows_teleop_action_processor()
    robot_action_processor = make_questhaptics_densowindows_robot_action_processor()
    return (teleop_action_processor, robot_action_processor, robot_observation_processor)


def make_questhaptics_densodeltapose_processor():
    # For the delta-pose robot, convert Quest absolute to deltapose_* via dedicated step.
    robot_observation_processor = make_questhaptics_densodeltapose_robot_observation_processor()
    teleop_action_processor = make_questhaptics_densodeltapose_teleop_action_processor()
    robot_action_processor = make_questhaptics_densodeltapose_robot_action_processor()
    return (teleop_action_processor, robot_action_processor, robot_observation_processor)

def make_questhaptics_densodeltapose_force_processor():
    # For the delta-pose robot, convert Quest absolute to deltapose_* via dedicated step.
    robot_observation_processor = make_questhaptics_densodeltapose_force_robot_observation_processor()
    teleop_action_processor = make_questhaptics_densodeltapose_force_teleop_action_processor()
    robot_action_processor = make_questhaptics_densodeltapose_force_robot_action_processor()
    return (teleop_action_processor, robot_action_processor, robot_observation_processor)

