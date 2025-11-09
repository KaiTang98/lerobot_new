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


@RobotConfig.register_subclass("robotiq_gripper")
@dataclass(kw_only=True)
class RobotiqGripperConfig(RobotConfig):
    """Configuration for a standalone Robotiq gripper robot.

    The underlying driver is provided by the optional dependency `pyrobotiqgripper`.
    Add any connection params here in the future if needed.
    """

    # Optional connection parameter: serial port or device path for the gripper
    port: str | None = None

    # cameras (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

