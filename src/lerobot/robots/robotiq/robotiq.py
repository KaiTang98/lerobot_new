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

from typing import Any
import threading
import queue

from lerobot.cameras.utils import make_cameras_from_configs

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_robotiq import RobotiqGripperConfig


class RobotiqGripper(Robot):
    """Standalone Robotiq gripper robot.

    Expected action key: `gripper` with values {0: close, 1: stay, 2: open}.
    For convenience, if `left_gripper` or `right_gripper` are provided instead, those will be used.
    Observation exposes a simple `gripper_state` float in {0.0 (closed), 1.0 (open)}; when unknown, reflects last command.
    """

    config_class = RobotiqGripperConfig
    name = "robotiq_gripper"

    def __init__(self, config: RobotiqGripperConfig):
        super().__init__(config)
        self.config = config
        # Initialize cameras if provided in the config to match other robot implementations
        # so higher-level code (record, teleop) can query `robot.cameras`.
        try:
            self.cameras = make_cameras_from_configs(config.cameras)
        except Exception:
            # If cameras are not provided or fail to initialize, default to empty dict
            self.cameras = {}
        self._connected = False
        self._driver = None
        # Track last commanded state to report in observation
        self._gripper_state: float = 0.0  # 0.0 closed, 1.0 open
        # Background command queue + worker thread to avoid blocking the main loop
        self._cmd_queue: "queue.Queue[str | None]" = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._worker_stop: threading.Event | None = None
        self._state_lock = threading.Lock()

    # --- Robot API ---
    @property
    def observation_features(self) -> dict:
        features: dict[str, Any] = {"gripper_state": float}
        # Include camera feature shapes if cameras are attached in the config
        for cam_key in getattr(self, "cameras", {}):
            try:
                cam_cfg = self.config.cameras[cam_key]
                features[cam_key] = (cam_cfg.height, cam_cfg.width, 3)
            except Exception:
                # If config missing details, skip adding that camera feature
                continue
        return features

    @property
    def action_features(self) -> dict:
        return {"gripper": int}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError("Robotiq gripper is already connected.")
        try:
            from pyrobotiqgripper import RobotiqGripper as _Driver  # type: ignore
        except Exception as e:
            raise ImportError(
                "pyrobotiqgripper is required for the robotiq_gripper robot.\n"
                "Install with: pip install pyrobotiqgripper"
            ) from e

        # Initialize underlying driver (use config params if added later)
        self._driver = _Driver()
        # Activate
        self._driver.activate()
        self._connected = True

        # Start the worker thread
        self._worker_stop = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, name="robotiq-worker", daemon=True)
        self._worker_thread.start()
        # Connect any cameras provided in the config so they are ready for recording
        for cam in getattr(self, "cameras", {}).values():
            try:
                cam.connect()
            except Exception:
                # don't fail the whole connect if a camera can't be opened; log would be nicer
                pass
        if calibrate and not self.is_calibrated:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        # No calibration flow for this simple device
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError("Robotiq gripper is not connected. Call connect() first.")
        # If driver exposes a status, use it; otherwise, rely on last commanded state.
        obs: dict[str, Any] = {"gripper_state": float(self._gripper_state)}

        # Capture images from any attached cameras (non-blocking async_read preferred)
        for cam_key, cam in getattr(self, "cameras", {}).items():
            try:
                obs[cam_key] = cam.async_read()
            except Exception:
                # If async_read fails, skip that camera for this frame
                obs[cam_key] = None

        return obs

    def _extract_gripper_cmd(self, action: dict[str, Any]) -> int:
        if "gripper" in action:
            return int(action["gripper"])  # type: ignore[arg-type]
        if "left_gripper" in action:
            return int(action["left_gripper"])  # type: ignore[arg-type]
        if "right_gripper" in action:
            return int(action["right_gripper"])  # type: ignore[arg-type]
        raise KeyError("Expected 'gripper' in action (or 'left_gripper'/'right_gripper').")

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError("Robotiq gripper is not connected. Call connect() first.")
        assert self._driver is not None
        cmd = self._extract_gripper_cmd(action)
        # Map 0 close, 1 stay, 2 open
        if cmd == 0:
            # enqueue close
            try:
                self._cmd_queue.put_nowait("close")
            except queue.Full:
                # unlikely; drop command
                pass
        elif cmd == 2:
            try:
                self._cmd_queue.put_nowait("open")
            except queue.Full:
                pass
        else:
            # stay -> no command
            pass

        # Return normalized action indicating what was commanded (immediate)
        return {"gripper": int(cmd)}

    def _worker_loop(self) -> None:
        """Background worker that executes open/close commands sequentially."""
        assert self._driver is not None
        while self._worker_stop is not None and not self._worker_stop.is_set():
            try:
                cmd = self._cmd_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if cmd is None:
                break
            try:
                if cmd == "open":
                    try:
                        self._driver.open()
                        with self._state_lock:
                            self._gripper_state = 1.0
                    except Exception:
                        # driver error; ignore but keep loop running
                        pass
                elif cmd == "close":
                    try:
                        self._driver.close()
                        with self._state_lock:
                            self._gripper_state = 0.0
                    except Exception:
                        pass
            finally:
                try:
                    self._cmd_queue.task_done()
                except Exception:
                    pass

    def disconnect(self) -> None:
        # Stop worker thread cleanly
        if self._worker_stop is not None:
            self._worker_stop.set()
        if self._worker_thread is not None:
            # wake up the worker
            try:
                self._cmd_queue.put_nowait(None)
            except Exception:
                pass
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None
        self._worker_stop = None

        self._connected = False
        self._driver = None

        # Disconnect any attached cameras
        for cam in getattr(self, "cameras", {}).values():
            try:
                if cam.is_connected:
                    cam.disconnect()
            except Exception:
                pass
