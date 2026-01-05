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


class QuestHapticsDensoDeltaPoseForceRobotActionStep(ProcessorStep):
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

        # Rotation quaternions (Left)
        l_quat = np.array([f("l_qx"), f("l_qy"), f("l_qz"), f("l_qw")], dtype=np.float32)
        l_quat_i = np.array([f("l_i_qx"), f("l_i_qy"), f("l_i_qz"), f("l_i_qw")], dtype=np.float32)
        # Convert to rotation matrices and compute delta rotation in robot frame
        Rl = quat_to_matrix(l_quat)
        Rl_i = quat_to_matrix(l_quat_i)
        # Apply Quest->Robot axis transform to orientations as well
        Rl = self._R_quest2robot @ Rl @ self._R_quest2robot.T
        Rl_i = self._R_quest2robot @ Rl_i @ self._R_quest2robot.T
        dRl = Rl @ Rl_i.T
        l_euler = rotation_matrix_to_euler_xyz(dRl)

        if active_left:
            l_dpos = (l_pos - l_pos_i) * self._scale
            if self._deadzone > 0.0:
                l_dpos = np.where(np.abs(l_dpos) < self._deadzone, 0.0, l_dpos)
        else:
            l_dpos = np.zeros(3, dtype=np.float32)
            l_euler = np.zeros(3, dtype=np.float32)

        # Right absolute and initial
        r_pos = np.array([f("r_x"), f("r_y"), f("r_z")], dtype=np.float32)
        r_pos_i = np.array([f("r_i_x"), f("r_i_y"), f("r_i_z")], dtype=np.float32)
        r_pos = self._R_quest2robot @ r_pos
        r_pos_i = self._R_quest2robot @ r_pos_i

        # Rotation quaternions (Right)
        r_quat = np.array([f("r_qx"), f("r_qy"), f("r_qz"), f("r_qw")], dtype=np.float32)
        r_quat_i = np.array([f("r_i_qx"), f("r_i_qy"), f("r_i_qz"), f("r_i_qw")], dtype=np.float32)
        Rr = quat_to_matrix(r_quat)
        Rr_i = quat_to_matrix(r_quat_i)
        Rr = self._R_quest2robot @ Rr @ self._R_quest2robot.T
        Rr_i = self._R_quest2robot @ Rr_i @ self._R_quest2robot.T
        dRr = Rr @ Rr_i.T
        r_euler = rotation_matrix_to_euler_xyz(dRr)
        
        if active_right:
            r_dpos = (r_pos - r_pos_i) * self._scale
            if self._deadzone > 0.0:
                r_dpos = np.where(np.abs(r_dpos) < self._deadzone, 0.0, r_dpos)
        else:
            r_dpos = np.zeros(3, dtype=np.float32)
            r_euler = np.zeros(3, dtype=np.float32)

        # Rotation deltas from quaternion delta, expressed as XYZ Euler (radians)
        l_rx, l_ry, l_rz = float(l_euler[0]), float(l_euler[1]), float(l_euler[2])
        r_rx, r_ry, r_rz = float(r_euler[0]), float(r_euler[1]), float(r_euler[2])

        # Start/end flags
        start_A = 1 if active_left else 0
        end_A = 0 if active_left else 1
        start_B = 1 if active_right else 0
        end_B = 0 if active_right else 1

        # button states
        button_A_1 = int(f("l_trigger_up_press"))
        button_A_2 = int(f("l_trigger_down_press"))
        button_B_1 = int(f("r_trigger_up_press"))
        button_B_2 = int(f("r_trigger_down_press"))

        print("button_A_1:", button_A_1)
        print("button_A_2:", button_A_2)
        print("button_B_1:", button_B_1)
        print("button_B_2:", button_B_2)

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
            # Button states
            "button_A_1": button_A_1,
            "button_A_2": button_A_2,
            "button_B_1": button_B_1,
            "button_B_2": button_B_2,
        }

        tr[TransitionKey.ACTION] = new_action
        return tr

    def transform_features(self, features):
        # No schema change advertised here; downstream may accept extra keys in action dict.
        return features


# --- Math helpers ---
def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix.
    Assumes quaternion is normalized; if not, normalizes it.
    """
    qx, qy, qz, qw = q.astype(np.float64)
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm == 0:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),       2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),   1 - 2*(xx + yy)],
    ], dtype=np.float64)
    return R


def rotation_matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to XYZ Euler angles (radians).
    Uses the convention R = Rx(x) @ Ry(y) @ Rz(z).
    Handles gimbal lock when cos(y) ~ 0.
    """
    r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]

    # y = asin(r02)
    y = np.arcsin(np.clip(r02, -1.0, 1.0))
    cy = np.cos(y)
    if abs(cy) > 1e-6:
        x = np.arctan2(-r12, r22)
        z = np.arctan2(-r01, r00)
    else:
        # Gimbal lock: set z = 0 and solve for x
        x = np.arctan2(r21, r11)
        z = 0.0
    return np.array([x, y, z], dtype=np.float64)
