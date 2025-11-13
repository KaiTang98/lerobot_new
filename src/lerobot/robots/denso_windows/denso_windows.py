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

import json
import select
import socket
import threading
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_denso_windows import DensoWindowsConfig


class DensoWindows(Robot):
    """Denso manipulator controlled by a remote Windows host over TCP.

    This client forwards teleoperation/inference actions to the Windows PC,
    and receives robot state (and possibly auxiliary signals) as JSON lines.

    Expected teleop action keys (from Bi-Spacemouse):
      - left_delta_x/y/z, right_delta_x/y/z, optional left_gripper/right_gripper (0 close,1 stay,2 open)

    Observation includes a flat vector under OBS_STATE and any configured camera frames.
    """

    config_class = DensoWindowsConfig
    name = "denso_windows"

    def __init__(self, config: DensoWindowsConfig):
        super().__init__(config)
        self.config = config

        # Cameras (optional)
        self.cameras = make_cameras_from_configs(config.cameras)

        # TCP client state
        self._sock: socket.socket | None = None
        self._reader_thread: threading.Thread | None = None
        self._reader_stop: threading.Event | None = None

        # Latest remote state cached
        self._last_remote_state: dict[str, Any] = {}

        self._is_connected: bool = False

    # -------------------- Feature descriptors --------------------
    @cached_property
    def _state_ft(self) -> dict[str, type]:
        # Keep human-readable ordering; matches your earlier schema (51 dims)
        keys: list[str] = [
            # A joints (pos 6 + vel 6)
            "curPos_J1_A", "curPos_J2_A", "curPos_J3_A", "curPos_J4_A", "curPos_J5_A", "curPos_J6_A",
            "curVel_J1_A", "curVel_J2_A", "curVel_J3_A", "curVel_J4_A", "curVel_J5_A", "curVel_J6_A",
            # A cartesian (pos 6) + FT (6) + task (1)
            "curPos_x_A", "curPos_y_A", "curPos_z_A", "curPos_roll_A", "curPos_pitch_A", "curPos_yaw_A",
            "curFT_x_A", "curFT_y_A", "curFT_z_A", "curFT_roll_A", "curFT_pitch_A", "curFT_yaw_A",
            "curTask_A",
            # B joints (pos 6 + vel 6)
            "curPos_J1_B", "curPos_J2_B", "curPos_J3_B", "curPos_J4_B", "curPos_J5_B", "curPos_J6_B",
            "curVel_J1_B", "curVel_J2_B", "curVel_J3_B", "curVel_J4_B", "curVel_J5_B", "curVel_J6_B",
            # B cartesian (pos 6) + FT (6) + task (1)
            "curPos_x_B", "curPos_y_B", "curPos_z_B", "curPos_roll_B", "curPos_pitch_B", "curPos_yaw_B",
            "curFT_x_B", "curFT_y_B", "curFT_z_B", "curFT_roll_B", "curFT_pitch_B", "curFT_yaw_B",
            "curTask_B",
            # Optional: mesh error norm (if provided by server)
            # "mesh_error_norm",
        ]
        return dict.fromkeys(keys, float)

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        # Like LeKiwiClient, we expose named state scalars and camera frames; OBS_STATE vector is added at runtime.
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        # Accept deltas and optional grippers from bi_spacemouse.
        base = {
            "left_delta_x": float,
            "left_delta_y": float,
            "left_delta_z": float,
            "left_delta_rx": float,
            "left_delta_ry": float,
            "left_delta_rz": float,
            "right_delta_x": float,
            "right_delta_y": float,
            "right_delta_z": float,
            "right_delta_rx": float,
            "right_delta_ry": float,
            "right_delta_rz": float,
        }
        # Grippers are optional, so advertise but tolerate absence at runtime
        base["left_gripper"] = int
        base["right_gripper"] = int
        return base

    # -------------------- Connection lifecycle --------------------
    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 1) Connect TCP socket to Windows server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        while True:
            try:
                sock.connect((self.config.server_ip, self.config.server_port))
                break
            except (ConnectionRefusedError, TimeoutError, OSError):
                time.sleep(0.2)
        sock.setblocking(False)
        self._sock = sock

        # 2) Start background reader
        self._reader_stop = threading.Event()
        self._reader_thread = threading.Thread(target=self._reader_loop, name="denso-reader", daemon=True)
        self._reader_thread.start()

        # 3) Connect cameras
        for cam in self.cameras.values():
            try:
                cam.connect()
            except Exception:
                # non-fatal: proceed without camera
                pass

        self._is_connected = True

    @property
    def is_calibrated(self) -> bool:
        # remote PC is responsible for any calibration; local side is stateless
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    # -------------------- Background I/O --------------------
    def _reader_loop(self) -> None:
        assert self._sock is not None
        sock = self._sock
        buf = ""
        period = 1.0 / self.config.fps  # expected remote update rate
        try:
            while self._reader_stop is not None and not self._reader_stop.is_set():
                rlist, _, _ = select.select([sock], [], [], period)
                if sock not in rlist:
                    continue
                try:
                    data = sock.recv(65536)
                except BlockingIOError:
                    continue
                except OSError:
                    break
                if not data:
                    break
                buf += data.decode("utf-8", errors="ignore")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Expect keys like: timestamp, r_state_A (list[float]), r_state_B (list[float])
                    # Optionally: r_action_A/B, mesh_error, etc.
                    try:
                        a = np.asarray(msg.get("r_state_A", []), dtype=np.float32)
                        b = np.asarray(msg.get("r_state_B", []), dtype=np.float32)
                        state_vec = np.concatenate([a, b], dtype=np.float32)
                    except Exception:
                        state_vec = np.asarray([], dtype=np.float32)

                    out: dict[str, Any] = {k: 0.0 for k in self._state_ft}
                    # Fill per-key scalar mapping if lengths match expected schema
                    # Safely map joints/cart/cart_ft/task order (best-effort)
                    # If server changes layout, at least OBS_STATE remains valid
                    idx = 0
                    for key in self._state_ft:
                        if idx < state_vec.size:
                            out[key] = float(state_vec[idx])
                        idx += 1
                    out[OBS_STATE] = state_vec

                    # Cache last observation
                    self._last_remote_state = out
        finally:
            try:
                sock.close()
            except Exception:
                pass

    # -------------------- Robot API --------------------
    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs = dict(self._last_remote_state) if self._last_remote_state else {}

        # Attach current camera frames
        for cam_key, cam in self.cameras.items():
            try:
                obs[cam_key] = cam.async_read()
            except Exception:
                obs[cam_key] = None
        return obs

    def _gripper_int_to_buttons(self, v: int) -> tuple[bool, bool]:
        # Map 0(close),1(stay),2(open) -> two "button" booleans expected by your Windows server
        if v == 0:
            return True, False
        if v == 2:
            return False, True
        return False, False

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._is_connected or self._sock is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read bi-spacemouse fields; missing keys default to 0
        lx = float(action.get("left_delta_x", 0.0))
        ly = float(action.get("left_delta_y", 0.0))
        lz = float(action.get("left_delta_z", 0.0))
        lrx = float(action.get("left_delta_rx", 0.0))
        lry = float(action.get("left_delta_ry", 0.0))
        lrz = float(action.get("left_delta_rz", 0.0))

        rx = float(action.get("right_delta_x", 0.0))
        ry = float(action.get("right_delta_y", 0.0))
        rz = float(action.get("right_delta_z", 0.0))
        rrx = float(action.get("right_delta_rx", 0.0))
        rry = float(action.get("right_delta_ry", 0.0))
        rrz = float(action.get("right_delta_rz", 0.0))

        lg = int(action.get("left_gripper", 1))
        rg = int(action.get("right_gripper", 1))

        lbtn0, lbtn1 = self._gripper_int_to_buttons(lg)
        rbtn0, rbtn1 = self._gripper_int_to_buttons(rg)

        # processing delta action [-1, 1]
        lx = np.clip(lx, -1.0, 1.0)
        ly = np.clip(ly, -1.0, 1.0)
        lz = np.clip(lz, -1.0, 1.0)
        lrx = np.clip(lrx, -1.0, 1.0)
        lry = np.clip(lry, -1.0, 1.0)
        lrz = np.clip(lrz, -1.0, 1.0)

        rx = np.clip(rx, -1.0, 1.0)
        ry = np.clip(ry, -1.0, 1.0)
        rz = np.clip(rz, -1.0, 1.0)
        rrx = np.clip(rrx, -1.0, 1.0)
        
        rry = np.clip(rry, -1.0, 1.0)
        rrz = np.clip(rrz, -1.0, 1.0)

        # Windows side expects 6-DoF [x,y,z,roll,pitch,yaw]; we only send translations and zero the rest
        action_A = [lx, ly, 0.0, 0.0, 0.0, 0.0]
        action_B = [rx, ry, 0.0, 0.0, 0.0, 0.0]

        payload = {
            "timestamp": time.time(),
            "task": "teleoperation",
            "sm_A": {"action": action_A, "button": [int(lbtn0), int(lbtn1)]},
            "sm_B": {"action": action_B, "button": [int(rbtn0), int(rbtn1)]},
        }

        try:
            msg = json.dumps(payload) + "\n"
            self._sock.sendall(msg.encode("utf-8"))
        except OSError:
            # connection hiccup; ignore this tick
            pass

        # For dataset compatibility, also return a flat ACTION vector (float32)
        act_vec = np.array([lx, ly, lz, rx, ry, rz, float(lg), float(rg)], dtype=np.float32)
        out = {
            "left_delta_x": lx,
            "left_delta_y": ly,
            "left_delta_z": lz,
            "left_delta_rx": lrx,
            "left_delta_ry": lry,
            "left_delta_rz": lrz,
            "right_delta_x": rx,
            "right_delta_y": ry,
            "right_delta_z": rz,
            "right_delta_rx": rrx,
            "right_delta_ry": rry,
            "right_delta_rz": rrz,
            "left_gripper": lg,
            "right_gripper": rg,
            ACTION: act_vec,
        }
        return out

    def disconnect(self) -> None:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Stop reader
        if self._reader_stop is not None:
            self._reader_stop.set()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            # poke the socket to unblock select
            try:
                if self._sock is not None:
                    self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self._reader_thread.join(timeout=2.0)
        self._reader_thread = None
        self._reader_stop = None

        # Close socket
        try:
            if self._sock is not None:
                self._sock.close()
        finally:
            self._sock = None

        # Disconnect cameras
        for cam in self.cameras.values():
            try:
                if cam.is_connected:
                    cam.disconnect()
            except Exception:
                pass

        self._is_connected = False
