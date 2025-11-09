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

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_spacemouse import SpacemouseConfig


@dataclass
class _Reading:
    # 8-length vector: [v_x, v_y, v_z, omega_x, omega_y, omega_z, button_1, button_2]
    action: np.ndarray
    timestamp: float


class SpacemouseTeleop(Teleoperator):
    """SpaceMouse teleoperator.

    Produces simple end-effector deltas (delta_x, delta_y, delta_z) and optional gripper action based on
    the two SpaceMouse buttons. This keeps it compatible with the default identity processors and the
    existing follower robots that accept EE deltas + optional gripper.
    """

    config_class = SpacemouseConfig
    name = "spacemouse"

    def __init__(self, config: SpacemouseConfig):
        super().__init__(config)
        self.config = config

        # Threaded reader state
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._new_data_event: threading.Event | None = None
        self._buffer: deque[_Reading] = deque(maxlen=1)
        self._lock = threading.Lock()

        # Cached connection flag
        self._connected = False

    # -------- Teleoperator interface --------
    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "vx": 0,
                "vy": 1,
                "vz": 2,
                "ox": 3,
                "oy": 4,
                "oz": 5,
                "b1": 6,
                "b2": 7,
            },
        }

    @property
    def feedback_features(self) -> dict:
        # No haptics/LED/audio supported
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError("SpaceMouse is already connected.")

        # Lazy import to avoid hard dependency when not used
        try:
            import pyspacemouse  # noqa: F401
        except Exception as e:  # pragma: no cover - import environment dependent
            raise ImportError(
                "pyspacemouse is required for the SpaceMouse teleoperator.\n"
                "Install with: pip install pyspacemouse"
            ) from e

        # Start reader thread
        self._stop_event = threading.Event()
        self._new_data_event = threading.Event()
        self._buffer.clear()
        self._thread = threading.Thread(target=self._reader_loop, name="smouse-reader", daemon=True)
        self._thread.start()

        # Wait for first sample
        if not self._new_data_event.wait(timeout=2.0):
            self.disconnect()
            raise RuntimeError("Failed to receive initial SpaceMouse reading.")

        self._connected = True

        # Auto-calibrate if desired (no-op by default)
        if calibrate and not self.is_calibrated:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        # No specific calibration for SpaceMouse
        return True

    def calibrate(self) -> None:
        # No calibration needed
        return None

    def configure(self) -> None:
        # No runtime configuration needed
        return None

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "Spacemouse is not connected. Call connect() before get_action()."
            )

        reading = self._latest_reading()
        vec = reading.action.astype(np.float32)
        # Unpack
        vx, vy, vz, ox, oy, oz, b0, b1 = vec.tolist()

        # Deadzone and scaling for translational axes
        dz = float(self.config.deadzone)
        def dzf(v: float) -> float:
            return 0.0 if abs(v) < dz else v

        vx, vy, vz = dzf(vx), dzf(vy), dzf(vz)

        if getattr(self.config, "invert_vx", False):
            vx = -vx
        if getattr(self.config, "invert_vy", False):
            vy = -vy
        if getattr(self.config, "invert_vz", False):
            vz = -vz
        if getattr(self.config, "invert_ox", False):
            ox = -ox
        if getattr(self.config, "invert_oy", False):    
            oy = -oy
        if getattr(self.config, "invert_oz", False):
            oz = -oz

        tscale = float(getattr(self.config, "translation_scale", 1.0))
        vx *= tscale
        vy *= tscale
        vz *= tscale

        # Optional rotation scaling (falls back to translation_scale if not present)
        rscale = float(getattr(self.config, "rotation_scale", 1.0))
        ox *= rscale
        oy *= rscale
        oz *= rscale

        return {
            "vx": float(vx),
            "vy": float(vy),
            "vz": float(vz),
            "ox": float(ox),
            "oy": float(oy),
            "oz": float(oz),
            "b1": float(b0),
            "b2": float(b1),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # SpaceMouse doesn't support feedback
        return None

    def disconnect(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            assert self._stop_event is not None
            self._stop_event.set()
            self._thread.join(timeout=2.0)

        self._thread = None
        self._stop_event = None
        self._new_data_event = None
        self._buffer.clear()
        self._connected = False

    # -------- Optional HIL events --------
    def get_teleop_events(self) -> dict[str, Any]:
        """Basic event set; intervention when any axis exceeds deadzone."""
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        reading = self._latest_reading()
        x, y, z = reading.action[:3]
        dz = float(self.config.deadzone)
        is_intervention = any(abs(v) >= dz for v in (x, y, z))

        return {
            TeleopEvents.IS_INTERVENTION: bool(is_intervention),
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    # -------- Reader thread --------
    def _reader_loop(self) -> None:
        # Import here to avoid making the module import hard-fail when pyspacemouse is missing
        import pyspacemouse

        # Open device (path may be None for default selection)
        if self.config.device_path:
            mouse = pyspacemouse.open(path=self.config.device_path)
        else:
            mouse = pyspacemouse.open()

        target_hz = max(50, int(self.config.fps or 100))
        target_period = 1.0 / float(target_hz)

        try:
            while self._stop_event is not None and not self._stop_event.is_set():
                loop_start = time.perf_counter()
                state = mouse.read()
                now = time.perf_counter()

                # Velocity-based mapping with sign conventions (user-provided reference)
                # [v_x, v_y, v_z, omega_x, omega_y, omega_z, button_1, button_2]
                vx = float(state.x)
                vy = float(state.y)
                vz = float(state.z)
                ox = float(state.pitch)
                oy = float(state.roll)
                oz = float(state.yaw)

                # Buttons as floats (0.0/1.0) to match dtype float32 vector
                b0 = float(getattr(state, "buttons", [0, 0])[0] if hasattr(state, "buttons") and len(state.buttons) > 0 else 0.0)
                b1 = float(getattr(state, "buttons", [0, 0])[1] if hasattr(state, "buttons") and len(state.buttons) > 1 else 0.0)

                vec = np.array([vx, vy, vz, ox, oy, oz, b0, b1], dtype=np.float32)

                with self._lock:
                    self._buffer.append(_Reading(vec, now))
                    if self._new_data_event is not None:
                        self._new_data_event.set()

                # pacing
                elapsed = time.perf_counter() - loop_start
                remaining = target_period - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            try:
                pyspacemouse.close()
            except Exception:
                pass

    def _latest_reading(self) -> _Reading:
        with self._lock:
            if not self._buffer:
                # No data; return zeros to avoid crashes (8-length vector)
                zeros = np.zeros(8, dtype=np.float32)
                return _Reading(zeros, time.perf_counter())
            return self._buffer[-1]
