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
