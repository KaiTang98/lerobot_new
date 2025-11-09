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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass(kw_only=True)
class SpacemouseConfig(TeleoperatorConfig):
    """Configuration for SpaceMouse teleoperator.

    Attributes:
        device_path: Optional OS device path (e.g. /dev/hidrawX). If None, uses pyspacemouse default.
        fps: Target read frequency for the background reader thread. Defaults to 100 Hz.
        use_gripper: If true, map buttons to gripper CLOSE/STAY/OPEN actions.
        translation_scale: Scale factor applied to x/y/z axes.
        invert_x/y/z: Axis inversion flags to match user preference/robot frame.
        deadzone: Values with absolute magnitude below this threshold are set to 0.
    """

    device_path: str | None = None
    fps: int = 100

    translation_scale: float = 1.0
    rotation_scale: float = 1.0

    invert_vx: bool = False
    invert_vy: bool = False
    invert_vz: bool = False
    invert_ox: bool = True
    invert_oy: bool = False
    invert_oz: bool = True

    deadzone: float = 0.0
