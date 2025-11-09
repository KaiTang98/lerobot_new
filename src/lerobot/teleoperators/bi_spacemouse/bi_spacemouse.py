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


'''
This module implements a bimanual SpaceMouse teleoperator for controlling robotic end-effectors.
It uses the `pyspacemouse` library to interface with two SpaceMouse devices (left and right).
The teleoperator reads input from both devices, processes the translation vectors, and generates
actions for controlling the end-effectors, including optional gripper actions.

To run a sanity-check of the Bi-Spacemouse input, use the following command:

python -m lerobot.teleoperators.bi_spacemouse.utils_bi_spacemouse \
  --left_device_path /dev/hidraw5 \
  --right_device_path /dev/hidraw6 \
  --fps 100 \
  --duration 10

'''

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_bi_spacemouse import BiSpacemouseConfig


@dataclass
class _Reading:
    # 8-length vector per device: [vx, vy, vz, ox, oy, oz, b1, b2]
    action: np.ndarray
    timestamp: float


class BiSpacemouseTeleop(Teleoperator):
    """Bimanual SpaceMouse teleoperator.

    Produces left_*/right_* end-effector deltas and optional gripper actions (0 close, 1 stay, 2 open).
    """

    config_class = BiSpacemouseConfig
    name = "bi_spacemouse"

    def __init__(self, config: BiSpacemouseConfig):
        super().__init__(config)
        self.config = config

        # Left device state
        self._left_thread: threading.Thread | None = None
        self._left_stop: threading.Event | None = None
        self._left_new: threading.Event | None = None
        self._left_buf: deque[_Reading] = deque(maxlen=1)
        self._left_lock = threading.Lock()

        # Right device state
        self._right_thread: threading.Thread | None = None
        self._right_stop: threading.Event | None = None
        self._right_new: threading.Event | None = None
        self._right_buf: deque[_Reading] = deque(maxlen=1)
        self._right_lock = threading.Lock()

        self._connected = False

    # --- Teleoperator API ---
    @property
    def action_features(self) -> dict:
        # 16-length vector combining left and right device outputs
        return {
            "dtype": "float32",
            "shape": (16,),
            "names": {
                "left_vx": 0,
                "left_vy": 1,
                "left_vz": 2,
                "left_ox": 3,
                "left_oy": 4,
                "left_oz": 5,
                "left_b1": 6,
                "left_b2": 7,
                "right_vx": 8,
                "right_vy": 9,
                "right_vz": 10,
                "right_ox": 11,
                "right_oy": 12,
                "right_oz": 13,
                "right_b1": 14,
                "right_b2": 15,
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
            raise DeviceAlreadyConnectedError("BiSpacemouse is already connected.")

        try:
            import pyspacemouse  # noqa: F401
        except Exception as e:
            raise ImportError(
                "pyspacemouse is required for the BiSpacemouse teleoperator.\n"
                "Install with: pip install pyspacemouse"
            ) from e

        # Start left thread
        self._left_stop = threading.Event()
        self._left_new = threading.Event()
        self._left_buf.clear()
        self._left_thread = threading.Thread(target=self._reader_loop_left, name="smouse-left", daemon=True)
        self._left_thread.start()

        # Start right thread
        self._right_stop = threading.Event()
        self._right_new = threading.Event()
        self._right_buf.clear()
        self._right_thread = threading.Thread(target=self._reader_loop_right, name="smouse-right", daemon=True)
        self._right_thread.start()

        # Wait for initial data
        if not (self._left_new.wait(2.0) and self._right_new.wait(2.0)):
            self.disconnect()
            raise RuntimeError("Failed to receive initial readings from both SpaceMouse devices.")

        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def _proc_vec(
        self,
        v: np.ndarray,
        t_scale: float,
        r_scale: float,
        inv_vx: bool,
        inv_vy: bool,
        inv_vz: bool,
        inv_ox: bool,
        inv_oy: bool,
        inv_oz: bool,
        dz: float,
    ) -> np.ndarray:
        vx, vy, vz, ox, oy, oz, b1, b2 = map(float, v[:8])
        # deadzone on translation only
        def dzf(a: float) -> float:
            return 0.0 if abs(a) < dz else a
        vx, vy, vz = dzf(vx), dzf(vy), dzf(vz)
        # inversions
        if inv_vx:
            vx = -vx
        if inv_vy:
            vy = -vy
        if inv_vz:
            vz = -vz
        if inv_ox:
            ox = -ox
        if inv_oy:
            oy = -oy
        if inv_oz:
            oz = -oz
        # scaling
        vx *= t_scale
        vy *= t_scale
        vz *= t_scale
        ox *= r_scale
        oy *= r_scale
        oz *= r_scale
        return np.array([vx, vy, vz, ox, oy, oz, b1, b2], dtype=np.float32)

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError("BiSpacemouse is not connected. Call connect() first.")

        l = self._latest_left()
        r = self._latest_right()

        l_vec = self._proc_vec(
            l.action,
            self.config.translation_scale_left,
            self.config.rotation_scale_left,
            self.config.invert_left_vx,
            self.config.invert_left_vy,
            self.config.invert_left_vz,
            self.config.invert_left_ox,
            self.config.invert_left_oy,
            self.config.invert_left_oz,
            self.config.deadzone_left,
        )
        r_vec = self._proc_vec(
            r.action,
            self.config.translation_scale_right,
            self.config.rotation_scale_right,
            self.config.invert_right_vx,
            self.config.invert_right_vy,
            self.config.invert_right_vz,
            self.config.invert_right_ox,
            self.config.invert_right_oy,
            self.config.invert_right_oz,
            self.config.deadzone_right,
        )

        return {
            "left_vx": float(l_vec[0]),
            "left_vy": float(l_vec[1]),
            "left_vz": float(l_vec[2]),
            "left_ox": float(l_vec[3]),
            "left_oy": float(l_vec[4]),
            "left_oz": float(l_vec[5]),
            "left_b1": float(l_vec[6]),
            "left_b2": float(l_vec[7]),
            "right_vx": float(r_vec[0]),
            "right_vy": float(r_vec[1]),
            "right_vz": float(r_vec[2]),
            "right_ox": float(r_vec[3]),
            "right_oy": float(r_vec[4]),
            "right_oz": float(r_vec[5]),
            "right_b1": float(r_vec[6]),
            "right_b2": float(r_vec[7]),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    def disconnect(self) -> None:
        # Left
        if self._left_thread is not None and self._left_thread.is_alive():
            assert self._left_stop is not None
            self._left_stop.set()
            self._left_thread.join(timeout=2.0)
        self._left_thread = None
        self._left_stop = None
        self._left_new = None
        self._left_buf.clear()

        # Right
        if self._right_thread is not None and self._right_thread.is_alive():
            assert self._right_stop is not None
            self._right_stop.set()
            self._right_thread.join(timeout=2.0)
        self._right_thread = None
        self._right_stop = None
        self._right_new = None
        self._right_buf.clear()

        self._connected = False

    def get_teleop_events(self) -> dict[str, Any]:
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        l = self._latest_left()
        r = self._latest_right()
        # consider both translation and rotation for intervention; use per-side deadzones for translation
        lv = l.action
        rv = r.action
        l_move = any(abs(v) >= self.config.deadzone_left for v in lv[:3]) or bool(lv[6]) or bool(lv[7])
        r_move = any(abs(v) >= self.config.deadzone_right for v in rv[:3]) or bool(rv[6]) or bool(rv[7])
        is_int = l_move or r_move
        return {
            TeleopEvents.IS_INTERVENTION: bool(is_int),
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    # --- Reader helpers ---
    def _reader_loop_left(self) -> None:
        import pyspacemouse

        mouse = pyspacemouse.open(path=self.config.left_device_path) if self.config.left_device_path else pyspacemouse.open()
        target_hz = max(50, int(self.config.fps or 100))
        period = 1.0 / float(target_hz)
        try:
            while self._left_stop is not None and not self._left_stop.is_set():
                t0 = time.perf_counter()
                state = mouse.read()
                now = time.perf_counter()
                # Raw values; inversion/scaling applied in get_action
                vx = float(state.x); vy = float(state.y); vz = float(state.z)
                ox = float(state.pitch); oy = float(state.roll); oz = float(state.yaw)
                b1 = float(state.buttons[0]) if hasattr(state, "buttons") and len(state.buttons) > 0 else 0.0
                b2 = float(state.buttons[1]) if hasattr(state, "buttons") and len(state.buttons) > 1 else 0.0
                with self._left_lock:
                    self._left_buf.append(_Reading(np.array([vx, vy, vz, ox, oy, oz, b1, b2], dtype=np.float32), now))
                    if self._left_new is not None:
                        self._left_new.set()
                dt = time.perf_counter() - t0
                rem = period - dt
                if rem > 0:
                    time.sleep(rem)
        finally:
            try:
                pyspacemouse.close()
            except Exception:
                pass

    def _reader_loop_right(self) -> None:
        import pyspacemouse

        mouse = pyspacemouse.open(path=self.config.right_device_path) if self.config.right_device_path else pyspacemouse.open()
        target_hz = max(50, int(self.config.fps or 100))
        period = 1.0 / float(target_hz)
        try:
            while self._right_stop is not None and not self._right_stop.is_set():
                t0 = time.perf_counter()
                state = mouse.read()
                now = time.perf_counter()
                vx = float(state.x); vy = float(state.y); vz = float(state.z)
                ox = float(state.pitch); oy = float(state.roll); oz = float(state.yaw)
                b1 = float(state.buttons[0]) if hasattr(state, "buttons") and len(state.buttons) > 0 else 0.0
                b2 = float(state.buttons[1]) if hasattr(state, "buttons") and len(state.buttons) > 1 else 0.0
                with self._right_lock:
                    self._right_buf.append(_Reading(np.array([vx, vy, vz, ox, oy, oz, b1, b2], dtype=np.float32), now))
                    if self._right_new is not None:
                        self._right_new.set()
                dt = time.perf_counter() - t0
                rem = period - dt
                if rem > 0:
                    time.sleep(rem)
        finally:
            try:
                pyspacemouse.close()
            except Exception:
                pass

    def _latest_left(self) -> _Reading:
        with self._left_lock:
            if not self._left_buf:
                zeros = np.zeros(8, dtype=np.float32)
                return _Reading(zeros, time.perf_counter())
            return self._left_buf[-1]

    def _latest_right(self) -> _Reading:
        with self._right_lock:
            if not self._right_buf:
                zeros = np.zeros(8, dtype=np.float32)
                return _Reading(zeros, time.perf_counter())
            return self._right_buf[-1]
