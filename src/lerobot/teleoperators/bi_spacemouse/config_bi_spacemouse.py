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


@TeleoperatorConfig.register_subclass("bi_spacemouse")
@dataclass(kw_only=True)
class BiSpacemouseConfig(TeleoperatorConfig):
    """Configuration for bimanual SpaceMouse teleoperation.

    Two SpaceMouse devices can be used (left and right). If one device_path is None, pyspacemouse will
    try to pick a default. Axis scale/inversions and deadzones are applied independently to both.

    This mirrors the single-device SpacemouseConfig naming: vx/vy/vz for translation, ox/oy/oz for rotation,
    b1/b2 for buttons, with separate translation and rotation scales and per-axis inversion flags.
    """

    left_device_path: str | None = None
    right_device_path: str | None = None
    fps: int = 100

    # Scales
    translation_scale_left: float = 1.0
    translation_scale_right: float = 1.0
    rotation_scale_left: float = 1.0
    rotation_scale_right: float = 1.0

    # Per-axis inversion (translation)
    invert_left_vx: bool = False
    invert_left_vy: bool = False
    invert_left_vz: bool = False
    invert_right_vx: bool = False
    invert_right_vy: bool = False
    invert_right_vz: bool = False

    # Per-axis inversion (rotation) — defaults align with single spacemouse (invert pitch/yaw)
    invert_left_ox: bool = True
    invert_left_oy: bool = False
    invert_left_oz: bool = True
    invert_right_ox: bool = True
    invert_right_oy: bool = False
    invert_right_oz: bool = True

    # Deadzones (applied to translation components)
    deadzone_left: float = 0.0
    deadzone_right: float = 0.0
