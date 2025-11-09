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


@TeleoperatorConfig.register_subclass("bi_quest")
@dataclass(kw_only=True)
class BiQuestConfig(TeleoperatorConfig):
    """Configuration for bimanual Meta Quest teleoperation.

    Two controllers (left/right) are used. Data comes from the bundled OculusReader (ADB logcat stream).
    """

    # ADB connection settings
    ip_address: str | None = None  # None -> USB; otherwise wireless debugging IP
    port: int = 5555

    # Loop settings
    fps: int = 100
    use_gripper: bool = True

    # Per-hand scaling/inversions/deadzone for translation deltas
    translation_scale_left: float = 1.0
    translation_scale_right: float = 1.0
    invert_left_x: bool = False
    invert_left_y: bool = False
    invert_left_z: bool = False
    invert_right_x: bool = False
    invert_right_y: bool = False
    invert_right_z: bool = False
    deadzone_left: float = 0.0
    deadzone_right: float = 0.0
