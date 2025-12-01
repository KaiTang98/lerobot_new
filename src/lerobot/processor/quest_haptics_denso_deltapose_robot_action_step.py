#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any

import numpy as np

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep


class QuestHapticsDensoDeltaPoseRobotActionStep(ProcessorStep):
    """Convert Quest-Haptics absolute poses into delta-pose actions for DensoDeltaPose.

    Behavior:
    - When left-X (l_teleop_active == 1) or right-A (r_teleop_active == 1) is active, compute
      quest delta: (current - initial) in Quest frame, rotate into robot frame, scale, apply deadzone.
    - Emit deltapose_l_* and deltapose_r_* fields (6 DoF; rotation terms currently 0.0 unless provided).
    - Emit start/end flags per arm so robot-side can latch as needed (start=1 while active, end=0; when inactive start=0, end=1).
    """

    def __init__(self, *, scale_mm: float = 1000.0, deadzone: float = 0.0):
        super().__init__()
        self._scale = float(scale_mm)
        self._deadzone = float(deadzone)
        # rotation from Quest to robot: same as Windows step
        self._R_quest2robot = np.array([[-1, 0, 0],
                                        [0, 0, 1],
                                        [0, 1, 0]], dtype=np.float32)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        tr = transition.copy()

        action = tr.get(TransitionKey.ACTION) or {}
        if not isinstance(action, dict):
            return tr

        # Helper fetchers from action
        def f(k: str, d: float = 0.0) -> float:
            try:
                return float(action.get(k, d))
            except Exception:
                return d

        # Active flags (latched upstream in teleop, here treated as on/off per tick)
        active_left = (f("l_teleop_active") == 1.0)
        active_right = (f("r_teleop_active") == 1.0)

        # Left absolute and initial
        l_pos = np.array([f("l_x"), f("l_y"), f("l_z")], dtype=np.float32)
        l_pos_i = np.array([f("l_i_x"), f("l_i_y"), f("l_i_z")], dtype=np.float32)
        l_pos = self._R_quest2robot @ l_pos
        l_pos_i = self._R_quest2robot @ l_pos_i
        if active_left:
            l_dpos = (l_pos - l_pos_i) * self._scale
            if self._deadzone > 0.0:
                l_dpos = np.where(np.abs(l_dpos) < self._deadzone, 0.0, l_dpos)
        else:
            l_dpos = np.zeros(3, dtype=np.float32)

        # Right absolute and initial
        r_pos = np.array([f("r_x"), f("r_y"), f("r_z")], dtype=np.float32)
        r_pos_i = np.array([f("r_i_x"), f("r_i_y"), f("r_i_z")], dtype=np.float32)
        r_pos = self._R_quest2robot @ r_pos
        r_pos_i = self._R_quest2robot @ r_pos_i
        if active_right:
            r_dpos = (r_pos - r_pos_i) * self._scale
            if self._deadzone > 0.0:
                r_dpos = np.where(np.abs(r_dpos) < self._deadzone, 0.0, r_dpos)
        else:
            r_dpos = np.zeros(3, dtype=np.float32)

        # Rotation deltas (rx, ry, rz): if available in teleop, add similarly; else 0.0
        l_rx = float(f("l_rx", 0.0))
        l_ry = float(f("l_ry", 0.0))
        l_rz = float(f("l_rz", 0.0))
        r_rx = float(f("r_rx", 0.0))
        r_ry = float(f("r_ry", 0.0))
        r_rz = float(f("r_rz", 0.0))

        # Start/end flags
        start_A = 1 if active_left else 0
        end_A = 0 if active_left else 1
        start_B = 1 if active_right else 0
        end_B = 0 if active_right else 1

        # Only emit the fields needed downstream; omit original teleop inputs.
        new_action: dict[str, Any] = {
            # Left arm delta pose
            "deltapose_l_x": float(l_dpos[0]),
            "deltapose_l_y": float(l_dpos[1]),
            "deltapose_l_z": float(l_dpos[2]),
            "deltapose_l_rx": l_rx,
            "deltapose_l_ry": l_ry,
            "deltapose_l_rz": l_rz,
            # Right arm delta pose
            "deltapose_r_x": float(r_dpos[0]),
            "deltapose_r_y": float(r_dpos[1]),
            "deltapose_r_z": float(r_dpos[2]),
            "deltapose_r_rx": r_rx,
            "deltapose_r_ry": r_ry,
            "deltapose_r_rz": r_rz,
            # Start/End flags
            "start_A": int(start_A),
            "end_A": int(end_A),
            "start_B": int(start_B),
            "end_B": int(end_B),
        }

        tr[TransitionKey.ACTION] = new_action
        return tr

    def transform_features(self, features):
        # No schema change advertised here; downstream may accept extra keys in action dict.
        return features
