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
from lerobot.utils.constants import ACTION
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_denso_deltapose_force import DensoDeltaPoseForceConfig


class DensoDeltaPoseForce(Robot):
    """Denso manipulator client that forwards delta-pose commands directly.

    This mirrors the Windows TCP protocol used by `DensoWindows`, but the action schema
    here is explicit delta pose for each arm (A=left, B=right):
      - deltapose_l_[x,y,z,rx,ry,rz], deltapose_r_[x,y,z,rx,ry,rz]
      - start_A/end_A, start_B/end_B (latching flags managed host-side if needed)

    Observations expose the named scalar state and any configured camera frames.
    """

    config_class = DensoDeltaPoseForceConfig
    name = "denso_deltapose_force"

    def __init__(self, config: DensoDeltaPoseForceConfig):
        super().__init__(config)
        self.config = config

        # Cameras (optional)
        self.cameras = make_cameras_from_configs(config.cameras)

        # TCP client state
        self._sock: socket.socket | None = None
        self._reader_thread: threading.Thread | None = None
        self._reader_stop: threading.Event | None = None

        # Latest remote state cached
        self._last_remote_state = {}
        # Latest action coming from intime (Windows host), parsed into des* keys
        self._last_remote_action = {}

        self._is_connected = False

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
        ]
        return dict.fromkeys(keys, float)

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        # We expose named state scalars and camera frames (no flat OBS_STATE vector here).
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        # Direct delta pose for left(A) and right(B), plus start/end flags per arm.
        return {
            # action A (left) from quest
            "deltapose_l_x": float, "deltapose_l_y": float, "deltapose_l_z": float, "deltapose_l_rx": float, "deltapose_l_ry": float, "deltapose_l_rz": float,
            # action B (right) from quest
            "deltapose_r_x": float, "deltapose_r_y": float, "deltapose_r_z": float, "deltapose_r_rx": float, "deltapose_r_ry": float, "deltapose_r_rz": float,
            # state change for A/B (use float so dataset schema includes them; values are 0.0 or 1.0)
            "start_A": float, "end_A": float,
            "start_B": float, "end_B": float,
            # buttons control
            "button_A_1": float, "button_A_2": float,
            "button_B_1": float, "button_B_2": float,
            # action A from intime (POS, VEL, VELPure, FT) (24)
            "desPos_x_A": float, "desPos_y_A": float, "desPos_z_A": float,
            "desPos_roll_A": float, "desPos_pitch_A": float, "desPos_yaw_A": float,
            "desVel_x_A": float, "desVel_y_A": float, "desVel_z_A": float,
            "desVel_roll_A": float, "desVel_pitch_A": float, "desVel_yaw_A": float,
            "desVelPure_x_A": float, "desVelPure_y_A": float, "desVelPure_z_A": float,
            "desVelPure_roll_A": float, "desVelPure_pitch_A": float, "desVelPure_yaw_A": float,
            "desFT_x_A": float, "desFT_y_A": float, "desFT_z_A": float,
            "desFT_roll_A": float, "desFT_pitch_A": float, "desFT_yaw_A": float,
            # action B from intime (POS, VEL, VELPure, FT) (24)
            "desPos_x_B": float, "desPos_y_B": float, "desPos_z_B": float,
            "desPos_roll_B": float, "desPos_pitch_B": float, "desPos_yaw_B": float,
            "desVel_x_B": float, "desVel_y_B": float, "desVel_z_B": float,
            "desVel_roll_B": float, "desVel_pitch_B": float, "desVel_yaw_B": float,
            "desVelPure_x_B": float, "desVelPure_y_B": float, "desVelPure_z_B": float,
            "desVelPure_roll_B": float, "desVelPure_pitch_B": float, "desVelPure_yaw_B": float,
            "desFT_x_B": float, "desFT_y_B": float, "desFT_z_B": float,
            "desFT_roll_B": float, "desFT_pitch_B": float, "desFT_yaw_B": float
        }

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
        
        # Add timeout to prevent infinite loop (30 seconds total)
        max_attempts = 150  # 150 * 0.2s = 30 seconds
        attempts = 0
        print(f"[ROBOT] Attempting to connect to Denso server at {self.config.server_ip}:{self.config.server_port}...")
        
        while attempts < max_attempts:
            try:
                sock.connect((self.config.server_ip, self.config.server_port))
                print(f"[ROBOT] Successfully connected to Denso server after {attempts} attempts")
                break
            except (ConnectionRefusedError, TimeoutError, OSError) as e:
                attempts += 1
                if attempts % 25 == 0:  # Print every 5 seconds
                    print(f"[ROBOT] Still trying to connect... ({attempts}/{max_attempts} attempts)")
                time.sleep(0.2)
        else:
            # Timeout reached
            raise ConnectionError(
                f"Failed to connect to Denso robot server at {self.config.server_ip}:{self.config.server_port} "
                f"after {max_attempts} attempts (30 seconds). Please ensure:\n"
                f"  1. Windows server is running on {self.config.server_ip}\n"
                f"  2. Server is listening on port {self.config.server_port}\n"
                f"  3. Network connection is working (can you ping {self.config.server_ip}?)"
            )
        
        sock.setblocking(False)
        self._sock = sock

        # 2) Start background reader
        self._reader_stop = threading.Event()
        self._reader_thread = threading.Thread(target=self._reader_loop, name="denso-reader", daemon=True)
        self._reader_thread.start()

        # 3) Connect cameras and ensure they have warmed up (produced at least one frame).
        for cam_key, cam in self.cameras.items():
            try:
                cam.connect()
                # Wait for at least one successful frame after connect. Use camera's warmup_s
                # as a guideline but allow a small additional buffer.
                warmup_seconds = getattr(cam, "warmup_s", 1.0) or 1.0
                timeout = warmup_seconds + 1.0
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        if getattr(cam, "use_depth", False):
                            cam.async_read_both(timeout_ms=1000)
                        else:
                            cam.async_read(timeout_ms=1000)
                        # Successful read - camera warmed up
                        break
                    except Exception:
                        time.sleep(0.1)
                else:
                    print(f"[ROBOT] Warning: camera {cam_key} did not produce frames within {timeout:.1f}s after connect.")
            except Exception as e:
                # non-fatal: proceed without camera but log a warning
                print(f"[ROBOT] Warning: failed to connect camera {cam_key}: {e}")

        self._is_connected = True

    @property
    def is_calibrated(self) -> bool:
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
                    try:
                        a = np.asarray(msg.get("r_state_A", []), dtype=np.float32)
                        b = np.asarray(msg.get("r_state_B", []), dtype=np.float32)
                        state_vec = np.concatenate([a, b], dtype=np.float32)

                        action_A = np.asarray(msg.get("r_action_A", []), dtype=np.float32)
                        action_B = np.asarray(msg.get("r_action_B", []), dtype=np.float32)
                        action_vec = np.concatenate([action_A, action_B], dtype=np.float32)
                    except Exception:
                        state_vec = np.asarray([], dtype=np.float32)
                        action_vec = np.asarray([], dtype=np.float32)

                    # Build observation dict
                    out: dict[str, Any] = {k: 0.0 for k in self._state_ft}

                    # Fill per-key scalar mapping if lengths match expected schema
                    idx = 0
                    for key in self._state_ft:
                        if idx < state_vec.size:
                            out[key] = float(state_vec[idx])
                        idx += 1

                    # Cache last observation
                    self._last_remote_state = out
                    # Map action_vec into intime action feature keys (desPos/desVel/desVelPure/desFT for A then B)
                    action_keys_order = [
                        # A (24)
                        "desPos_x_A", "desPos_y_A", "desPos_z_A",
                        "desPos_roll_A", "desPos_pitch_A", "desPos_yaw_A",
                        "desVel_x_A", "desVel_y_A", "desVel_z_A",
                        "desVel_roll_A", "desVel_pitch_A", "desVel_yaw_A",
                        "desVelPure_x_A", "desVelPure_y_A", "desVelPure_z_A",
                        "desVelPure_roll_A", "desVelPure_pitch_A", "desVelPure_yaw_A",
                        "desFT_x_A", "desFT_y_A", "desFT_z_A",
                        "desFT_roll_A", "desFT_pitch_A", "desFT_yaw_A",
                        # B (24)
                        "desPos_x_B", "desPos_y_B", "desPos_z_B",
                        "desPos_roll_B", "desPos_pitch_B", "desPos_yaw_B",
                        "desVel_x_B", "desVel_y_B", "desVel_z_B",
                        "desVel_roll_B", "desVel_pitch_B", "desVel_yaw_B",
                        "desVelPure_x_B", "desVelPure_y_B", "desVelPure_z_B",
                        "desVelPure_roll_B", "desVelPure_pitch_B", "desVelPure_yaw_B",
                        "desFT_x_B", "desFT_y_B", "desFT_z_B",
                        "desFT_roll_B", "desFT_pitch_B", "desFT_yaw_B",
                    ]
                    act_out: dict[str, Any] = {}
                    for i, k in enumerate(action_keys_order):
                        if i < action_vec.size:
                            act_out[k] = float(action_vec[i])
                    self._last_remote_action = act_out
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
        # Attach cache under a namespaced key for processors to optionally consume.
        if self._last_remote_action:
            obs["_last_remote_action"] = dict(self._last_remote_action)

        # Attach current camera frames
        # LeRobot convention: flat keys for images, e.g. "camera_l515" for RGB
        for cam_key, cam in self.cameras.items():
            try:
                # Check if camera has depth enabled (use_depth attribute)
                if hasattr(cam, 'use_depth') and cam.use_depth:
                        # Use async_read_both() to get both color and depth. async_read_both
                        # now guarantees that when use_depth=True it will either return
                        # both 'color' and 'depth' or raise a TimeoutError. We still
                        # defensively handle missing keys to avoid KeyError and ensure
                        # observation always contains the sibling depth key.
                        frames = cam.async_read_both()
                        obs[cam_key] = frames.get("color")  # Will become observation.images.camera_l515
                        # Ensure the depth sibling key is always present (None if missing)
                        obs[f"{cam_key}_depth"] = frames.get("depth")
                else:
                    # Single color frame
                    obs[cam_key] = cam.async_read()
            except Exception as e:
                # Log a warning and ensure both rgb and depth sibling keys exist
                print(f"[WARNING] Failed to read from camera {cam_key}: {e}")
                obs[cam_key] = None
                obs[f"{cam_key}_depth"] = None
        return obs

    def send_action(self, action: dict[str, Any], is_infer: bool = False) -> dict[str, Any]:
        if not self._is_connected or self._sock is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if is_infer:
            # Read delta poses; missing keys default to 0
            lx = float(action.get("deltapose_l_x", 0.0))
            ly = float(action.get("deltapose_l_y", 0.0))
            lz = float(action.get("deltapose_l_z", 0.0))
            lrx = float(action.get("deltapose_l_rx", 0.0))
            lry = float(action.get("deltapose_l_ry", 0.0))
            lrz = float(action.get("deltapose_l_rz", 0.0))

            rx = float(action.get("deltapose_r_x", 0.0))
            ry = float(action.get("deltapose_r_y", 0.0))
            rz = float(action.get("deltapose_r_z", 0.0))
            rrx = float(action.get("deltapose_r_rx", 0.0))
            rry = float(action.get("deltapose_r_ry", 0.0))
            rrz = float(action.get("deltapose_r_rz", 0.0))

            lx = float(np.clip(lx, -10.0, 10.0))
            ly = float(np.clip(ly, -10.0, 10.0))
            lz = float(np.clip(lz, -10.0, 10.0))
            lrx = float(np.clip(lrx, -1.0, 1.0))
            lry = float(np.clip(lry, -1.0, 1.0))
            lrz = float(np.clip(lrz, -1.0, 1.0))

            rx = float(np.clip(rx, -10.0, 10.0))
            ry = float(np.clip(ry, -10.0, 10.0))
            rz = float(np.clip(rz, -10.0, 10.0))
            rrx = float(np.clip(rrx, -1.0, 1.0))
            rry = float(np.clip(rry, -1.0, 1.0))
            rrz = float(np.clip(rrz, -1.0, 1.0))

            action_A = [lx, ly, lz, lrx, lry, lrz]
            action_B = [rx, ry, rz, rrx, rry, rrz]

            print(action_A)
            print(action_B)

            payload = {
                "timestamp": time.time(),
                "task": "inference",
                "sm_A": {"action": action_A, "desFz_A": 0.0},
                "sm_B": {"action": action_B, "desFz_B": 0.0},
            }

            start_A = 0
            end_A = 0
            start_B = 0
            end_B = 0
            
        else:
            # Read delta poses; missing keys default to 0
            lx = float(action.get("deltapose_l_x", 0.0))
            ly = float(action.get("deltapose_l_y", 0.0))
            lz = float(action.get("deltapose_l_z", 0.0))
            lrx = float(action.get("deltapose_l_rx", 0.0))
            lry = float(action.get("deltapose_l_ry", 0.0))
            lrz = float(action.get("deltapose_l_rz", 0.0))

            rx = float(action.get("deltapose_r_x", 0.0))
            ry = float(action.get("deltapose_r_y", 0.0))
            rz = float(action.get("deltapose_r_z", 0.0))
            rrx = float(action.get("deltapose_r_rx", 0.0))
            rry = float(action.get("deltapose_r_ry", 0.0))
            rrz = float(action.get("deltapose_r_rz", 0.0))

            # Optional start/end flags (forwarded for host-side latching)
            start_A = int(action.get("start_A", 0))
            end_A = int(action.get("end_A", 0))
            start_B = int(action.get("start_B", 0))
            end_B = int(action.get("end_B", 0))

            lx = float(np.clip(lx, -100.0, 100.0))
            ly = float(np.clip(ly, -100.0, 100.0))
            lz = float(np.clip(lz, -100.0, 100.0))
            lrx = float(np.clip(lrx, -1.0, 1.0))
            lry = float(np.clip(lry, -1.0, 1.0))
            lrz = float(np.clip(lrz, -1.0, 1.0))

            rx = float(np.clip(rx, -100.0, 100.0))
            ry = float(np.clip(ry, -100.0, 100.0))
            rz = float(np.clip(rz, -100.0, 100.0))
            rrx = float(np.clip(rrx, -1.0, 1.0))
            rry = float(np.clip(rry, -1.0, 1.0))
            rrz = float(np.clip(rrz, -1.0, 1.0))

            # Build 6-DoF delta pose arrays
            action_A = [lx, ly, 0.0, 0.0, 0.0, lrz]
            action_B = [rx, ry, 0.0, 0.0, 0.0, rrz]
            # action_A = [lx, ly, lz, 0.0, 0.0, 0.0]
            # action_B = [rx, ry, rz, 0.0, 0.0, 0.0]

            print(action_A)
            print(action_B)

            button_A_1 = int(action.get("button_A_1", 0))
            button_A_2 = int(action.get("button_A_2", 0))
            button_B_1 = int(action.get("button_B_1", 0))
            button_B_2 = int(action.get("button_B_2", 0))
            button_A = [button_A_1, button_A_2]
            button_B = [button_B_1, button_B_2]

            print(button_A)
            print(button_B)

            payload = {
                "timestamp": time.time(),
                "task": "teleoperation",
                "sm_A": {"action": action_A, "start_A": start_A, "end_A": end_A, "button_A": button_A},
                "sm_B": {"action": action_B, "start_B": start_B, "end_B": end_B, "button_B": button_B},
            }

        try:
            msg = json.dumps(payload) + "\n"
            self._sock.sendall(msg.encode("utf-8"))
        except OSError:
            # connection hiccup; ignore this tick
            pass

        # For dataset/logging convenience, also return a flat ACTION vector (float32)
        act_vec = np.array([
            lx, ly, lz, lrx, lry, lrz,
            rx, ry, rz, rrx, rry, rrz,
            float(start_A), float(end_A), float(start_B), float(end_B),
            float(button_A_1), float(button_A_2), float(button_B_1), float(button_B_2)
        ], dtype=np.float32)

        out = {
            "deltapose_l_x": lx,
            "deltapose_l_y": ly,
            "deltapose_l_z": lz,
            "deltapose_l_rx": lrx,
            "deltapose_l_ry": lry,
            "deltapose_l_rz": lrz,
            "deltapose_r_x": rx,
            "deltapose_r_y": ry,
            "deltapose_r_z": rz,
            "deltapose_r_rx": rrx,
            "deltapose_r_ry": rry,
            "deltapose_r_rz": rrz,
            "start_A": start_A,
            "end_A": end_A,
            "start_B": start_B,
            "end_B": end_B,
            "button_A_1": button_A_1,
            "button_A_2": button_A_2,
            "button_B_1": button_B_1,
            "button_B_2": button_B_2,
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