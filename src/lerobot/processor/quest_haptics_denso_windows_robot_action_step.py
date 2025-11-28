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


class QuestHapticsDensoWindowsRobotActionStep(ProcessorStep):
    """Fuse Quest-Haptics teleop action with Denso observation to produce velocity-like commands.

    Behavior:
    - Start teleop for left when l_X pressed; for right when r_A pressed (latched).
    - Stop teleop for left when l_Y pressed; for right when r_B pressed (clears latch).
    - While active, compute Quest delta: current_pose - initial_pose (provided by teleop as l_* vs l_i_*),
      then form desired Denso pose: denso_init + quest_delta, and output error = desired - denso_current
      scaled and clipped to [-clamp, clamp] as velocity-like robot action.
    - Map trigger buttons to grippers: up=open(2), down=close(0), otherwise stay(1).
    - Emit is_teleop_active_A/B flags so the robot can capture/clear its own initial pose.
    """

    def __init__(self, *, clamp: float | None = 1.0, pos_gain: float = 1.0, deadzone: float = 0.0):
        super().__init__()
        self._clamp = float(clamp) if clamp is not None else None
        self._pos_gain = float(pos_gain)
        self._deadzone = float(deadzone)
        # Latched teleop state per arm
        self._active_left: bool = False
        self._active_right: bool = False
        # rotation about z axis of -90 degrees
        self._R_quest2robot = np.array([[-1, 0, 0],
                                        [0, 0, 1],
                                        [0, 1, 0]])

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        self._current_transition = transition.copy()
        tr = self._current_transition

        action = tr.get(TransitionKey.ACTION) or {}
        if not isinstance(action, dict):
            # Nothing to do if action is not a dict
            return tr

        obs = tr.get(TransitionKey.OBSERVATION) or {}
        if not isinstance(obs, dict):
            obs = {}

        # Helper fetchers
        def f(k: str, d: float = 0.0) -> float:
            try:
                return float(action.get(k, d))
            except Exception:
                return d

        def fo(k: str, d: float = 0.0) -> float:
            try:
                return float(obs.get(k, d))
            except Exception:
                return d

        # Latch start/stop events
        if f("l_teleop_active") == 1.0:
            self._active_left = True
        else:
            self._active_left = False
        if f("r_teleop_active") == 1.0:
            self._active_right = True
        else:
            self._active_right = False

        # Compute quest deltas (absolute - initial reference), if available
        # Left pose
        l_pos = np.array([f("l_x"), f("l_y"), f("l_z")], dtype=np.float32)
        l_pos_i = np.array([f("l_i_x"), f("l_i_y"), f("l_i_z")], dtype=np.float32)
        l_pos = self._R_quest2robot @ l_pos
        l_pos_i = self._R_quest2robot @ l_pos_i
        if self._active_left:
            l_dpos = (l_pos - l_pos_i) * 1000.0
            if self._deadzone > 0.0:
                l_dpos = np.where(np.abs(l_dpos) < self._deadzone, 0.0, l_dpos)
        else:
            l_dpos = np.zeros(3, dtype=np.float32)

        A_cur = np.array([fo("curPos_x_A"), fo("curPos_y_A"), fo("curPos_z_A")], dtype=np.float32)
        A_ini = np.array([fo("initPos_x_A"), fo("initPos_y_A"), fo("initPos_z_A")], dtype=np.float32)
        if self._active_left:
            A_err = np.zeros(3, dtype=np.float32)
            A_des = A_ini + l_dpos
            A_err = A_des - A_cur
        else:
            A_err = np.zeros(3, dtype=np.float32)
            
        # Right pose
        r_pos = np.array([f("r_x"), f("r_y"), f("r_z")], dtype=np.float32)
        r_pos_i = np.array([f("r_i_x"), f("r_i_y"), f("r_i_z")], dtype=np.float32)
        r_pos = self._R_quest2robot @ r_pos
        r_pos_i = self._R_quest2robot @ r_pos_i
        if self._active_right:
            r_dpos = (r_pos - r_pos_i) * 1000.0
            if self._deadzone > 0.0:
                r_dpos = np.where(np.abs(r_dpos) < self._deadzone, 0.0, r_dpos)
        else:
            r_dpos = np.zeros(3, dtype=np.float32)

        B_cur = np.array([fo("curPos_x_B"), fo("curPos_y_B"), fo("curPos_z_B")], dtype=np.float32)
        B_ini = np.array([fo("initPos_x_B"), fo("initPos_y_B"), fo("initPos_z_B")], dtype=np.float32)
        if self._active_right:     
            B_err = np.zeros(3, dtype=np.float32)
            B_des = B_ini + r_dpos
            B_err = B_des - B_cur
            
        else:
            B_err = np.zeros(3, dtype=np.float32)


        print("A_ini:", A_ini)
        print("B_ini:", B_ini)
        print("A_cur:", A_cur)
        print("B_cur:", B_cur)
        print("A_err:", A_err)
        print("B_err:", B_err)

        # # Apply deadzone to small errors
        # if self._deadzone > 0.0:
        #     A_err = np.where(np.abs(A_err) < self._deadzone, 0.0, A_err)
        #     B_err = np.where(np.abs(B_err) < self._deadzone, 0.0, B_err)

        # Velocity-like command with gain and clamp
        A_cmd = self._pos_gain * A_err
        B_cmd = self._pos_gain * B_err
        if self._clamp is not None and self._clamp > 0:
            A_cmd = np.clip(A_cmd, -self._clamp, self._clamp)
            B_cmd = np.clip(B_cmd, -self._clamp, self._clamp)

        print(f"l_dpos:", l_dpos)
        print(f"r_dpos:", r_dpos)
        print(f"A_cmd:", A_cmd)
        print(f"B_cmd:", B_cmd)

        # # Gripper mapping: triggers up/down per hand
        # l_open = bool(f("l_trigger_up_press"))
        # l_close = bool(f("l_trigger_down_press"))
        # r_open = bool(f("r_trigger_up_press"))
        # r_close = bool(f("r_trigger_down_press"))

        # if l_close and not l_open:
        #     left_gripper = 0
        # elif l_open and not l_close:
        #     left_gripper = 2
        # else:
        #     left_gripper = 1

        # if r_close and not r_open:
        #     right_gripper = 0
        # elif r_open and not r_close:
        #     right_gripper = 2
        # else:
        #     right_gripper = 1

        # If desired, incorporate robot state here:
        # obs = tr.get(TransitionKey.OBSERVATION) or {}
        # e.g., scale deltas based on measured force or proximity

        # Write back (Denso expects l_*, r_* deltas and flags)
        new_action: dict[str, Any] = {**action}
        new_action.update(
            {
                # Left arm
                "l_x": float(A_cmd[0]),
                "l_y": float(A_cmd[1]),
                "l_z": float(A_cmd[2]),
                "l_rx": 0.0,
                "l_ry": 0.0,
                "l_rz": 0.0,
                # Right arm
                "r_x": float(B_cmd[0]),
                "r_y": float(B_cmd[1]),
                "r_z": float(B_cmd[2]),
                "r_rx": 0.0,
                "r_ry": 0.0,
                "r_rz": 0.0,
                # Grippers
                "l_gripper": int(0),
                "r_gripper": int(0),
                # Teleop activation flags for robot-side init capture/clear
                "is_teleop_active_A": bool(self._active_left),
                "is_teleop_active_B": bool(self._active_right),
            }
        )
        tr[TransitionKey.ACTION] = new_action
        return tr

    def transform_features(self, features):
        # No schema change advertised here; downstream may accept extra keys in action dict.
        return features
