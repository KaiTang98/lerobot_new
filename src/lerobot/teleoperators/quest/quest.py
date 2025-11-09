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

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_quest import BiQuestConfig


@dataclass
class _Pose:
    t: np.ndarray  # translation, shape (3,)
    ts: float      # timestamp


class BiQuestTeleop(Teleoperator):
    """Bimanual Meta Quest teleoperator.

    Produces left/right end-effector deltas (x,y,z) and optional gripper actions.
    Gripper mapping (default): right A=open, B=close; left X=open, Y=close; else stay.
    """

    config_class = BiQuestConfig
    name = "bi_quest"

    def __init__(self, config: BiQuestConfig):
        super().__init__(config)
        self.config = config

        self._reader = None  # OculusReader instance (lazy import)
        self._connected = False

        # Last seen translations to produce per-frame deltas
        self._last_left: _Pose | None = None
        self._last_right: _Pose | None = None

    # --- Teleoperator API ---
    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (8,),
                "names": {
                    "left_delta_x": 0,
                    "left_delta_y": 1,
                    "left_delta_z": 2,
                    "left_gripper": 3,
                    "right_delta_x": 4,
                    "right_delta_y": 5,
                    "right_delta_z": 6,
                    "right_gripper": 7,
                },
            }
        else:
            return {
                "dtype": "float32",
                "shape": (6,),
                "names": {
                    "left_delta_x": 0,
                    "left_delta_y": 1,
                    "left_delta_z": 2,
                    "right_delta_x": 3,
                    "right_delta_y": 4,
                    "right_delta_z": 5,
                },
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError("BiQuest is already connected.")

        try:
            # Lazy import; keep path absolute to avoid confusion
            from lerobot.teleoperators.quest.oculus_reader.reader import OculusReader
        except Exception as e:
            raise ImportError(
                "OculusReader dependency is required for the BiQuest teleoperator.\n"
                "Ensure pure-python-adb is installed and adb is available on PATH."
            ) from e

        self._reader = OculusReader(ip_address=self.config.ip_address, port=self.config.port)

        # Wait briefly for initial data from both controllers
        t0 = time.perf_counter()
        ok_left = ok_right = False
        while time.perf_counter() - t0 < 2.0:
            transforms, _ = self._reader.get_transformations_and_buttons()
            ok_left = ok_left or ('l' in transforms and isinstance(transforms['l'], np.ndarray))
            ok_right = ok_right or ('r' in transforms and isinstance(transforms['r'], np.ndarray))
            if ok_left and ok_right:
                break
            time.sleep(0.01)

        if not (ok_left and ok_right):
            self.disconnect()
            raise RuntimeError("Failed to receive initial transforms from both Quest controllers.")

        # Initialize last pose baseline
        self._refresh_last_poses()
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        # Treat current poses as baseline
        self._refresh_last_poses()

    def configure(self) -> None:
        return None

    def _refresh_last_poses(self) -> None:
        assert self._reader is not None
        transforms, _ = self._reader.get_transformations_and_buttons()
        now = time.perf_counter()
        if 'l' in transforms and isinstance(transforms['l'], np.ndarray):
            tl = np.asarray(transforms['l'], dtype=np.float32)[:3, 3]
            self._last_left = _Pose(t=tl.copy(), ts=now)
        if 'r' in transforms and isinstance(transforms['r'], np.ndarray):
            tr = np.asarray(transforms['r'], dtype=np.float32)[:3, 3]
            self._last_right = _Pose(t=tr.copy(), ts=now)

    def _proc_delta(self, d: np.ndarray, scale: float, invx: bool, invy: bool, invz: bool, dz: float) -> np.ndarray:
        x, y, z = map(float, d[:3])
        def dzf(a: float) -> float:
            return 0.0 if abs(a) < dz else a
        x, y, z = dzf(x), dzf(y), dzf(z)
        if invx:
            x = -x
        if invy:
            y = -y
        if invz:
            z = -z
        x *= scale
        y *= scale
        z *= scale
        return np.array([x, y, z], dtype=np.float32)

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError("BiQuest is not connected. Call connect() first.")

        assert self._reader is not None
        transforms, buttons = self._reader.get_transformations_and_buttons()
        now = time.perf_counter()

        out: dict[str, Any] = {}

        # Left side
        if 'l' in transforms and isinstance(transforms['l'], np.ndarray):
            tl = np.asarray(transforms['l'], dtype=np.float32)[:3, 3]
            if self._last_left is None:
                self._last_left = _Pose(t=tl.copy(), ts=now)
            dl = tl - self._last_left.t
            self._last_left = _Pose(t=tl.copy(), ts=now)
            l_vec = self._proc_delta(
                dl,
                self.config.translation_scale_left,
                self.config.invert_left_x,
                self.config.invert_left_y,
                self.config.invert_left_z,
                self.config.deadzone_left,
            )
            out["left_delta_x"], out["left_delta_y"], out["left_delta_z"] = map(float, l_vec)
        else:
            out["left_delta_x"] = out["left_delta_y"] = out["left_delta_z"] = 0.0

        # Right side
        if 'r' in transforms and isinstance(transforms['r'], np.ndarray):
            tr = np.asarray(transforms['r'], dtype=np.float32)[:3, 3]
            if self._last_right is None:
                self._last_right = _Pose(t=tr.copy(), ts=now)
            dr = tr - self._last_right.t
            self._last_right = _Pose(t=tr.copy(), ts=now)
            r_vec = self._proc_delta(
                dr,
                self.config.translation_scale_right,
                self.config.invert_right_x,
                self.config.invert_right_y,
                self.config.invert_right_z,
                self.config.deadzone_right,
            )
            out["right_delta_x"], out["right_delta_y"], out["right_delta_z"] = map(float, r_vec)
        else:
            out["right_delta_x"] = out["right_delta_y"] = out["right_delta_z"] = 0.0

        if self.config.use_gripper:
            # Button mapping: Left X=open(2), Y=close(0); Right A=open(2), B=close(0); else stay(1)
            # Buttons dict keys from parser: 'A','B' for right; 'X','Y' for left
            l_open = bool(buttons.get('X', False))
            l_close = bool(buttons.get('Y', False))
            if l_close and not l_open:
                out["left_gripper"] = 0
            elif l_open and not l_close:
                out["left_gripper"] = 2
            else:
                out["left_gripper"] = 1

            r_open = bool(buttons.get('A', False))
            r_close = bool(buttons.get('B', False))
            if r_close and not r_open:
                out["right_gripper"] = 0
            elif r_open and not r_close:
                out["right_gripper"] = 2
            else:
                out["right_gripper"] = 1

        return out

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    def disconnect(self) -> None:
        if self._reader is not None:
            try:
                self._reader.stop()
            except Exception:
                pass
            self._reader = None
        self._connected = False

    def get_teleop_events(self) -> dict[str, Any]:
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Intervention if any delta exceeds respective deadzone
        action = self.get_action()
        is_int = (
            abs(action.get("left_delta_x", 0.0)) >= self.config.deadzone_left
            or abs(action.get("left_delta_y", 0.0)) >= self.config.deadzone_left
            or abs(action.get("left_delta_z", 0.0)) >= self.config.deadzone_left
            or abs(action.get("right_delta_x", 0.0)) >= self.config.deadzone_right
            or abs(action.get("right_delta_y", 0.0)) >= self.config.deadzone_right
            or abs(action.get("right_delta_z", 0.0)) >= self.config.deadzone_right
        )

        # Map terminate/success to buttons as a sensible default (customize later if needed)
        try:
            assert self._reader is not None
            _, buttons = self._reader.get_transformations_and_buttons()
        except Exception:
            buttons = {}

        terminate = bool(buttons.get('RJ', False) or buttons.get('LJ', False))  # joystick press ends episode
        success = bool(buttons.get('RThU', False) or buttons.get('LThU', False))  # thumb-up marks success

        return {
            TeleopEvents.IS_INTERVENTION: bool(is_int),
            TeleopEvents.TERMINATE_EPISODE: bool(terminate),
            TeleopEvents.SUCCESS: bool(success),
            TeleopEvents.RERECORD_EPISODE: False,
        }
