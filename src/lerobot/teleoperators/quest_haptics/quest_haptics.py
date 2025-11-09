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
from .config_quest_haptics import BiQuestHapticsConfig


@dataclass
class _Pose:
	t: np.ndarray         # translation (3,)
	R: np.ndarray         # rotation matrix (3,3)
	ts: float             # timestamp


def _rotation_vector(R_rel: np.ndarray) -> np.ndarray:
	"""Convert a relative rotation matrix to a Rodrigues rotation vector (rx, ry, rz).

	Returns a 3-vector whose direction is the rotation axis and magnitude is the angle (radians).
	Stable for small angles.
	"""
	# Clamp numerical issues
	tr = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
	theta = np.arccos(tr)
	if theta < 1e-6:
		return np.zeros(3, dtype=np.float32)
	denom = 2.0 * np.sin(theta)
	rx = (R_rel[2, 1] - R_rel[1, 2]) / denom
	ry = (R_rel[0, 2] - R_rel[2, 0]) / denom
	rz = (R_rel[1, 0] - R_rel[0, 1]) / denom
	axis = np.array([rx, ry, rz], dtype=np.float32)
	return axis * float(theta)


class BiQuestHapticsTeleop(Teleoperator):
	"""Bimanual Meta Quest teleoperator with haptics feedback.

	Provides per-hand translation and rotation *deltas* between successive frames:
	  left_delta_x/y/z, left_delta_rx/ry/rz, right_delta_x/y/z, right_delta_rx/ry/rz

	Feedback channel allows setting left/right haptic amplitudes in [0,1].
	Translation deltas are deadzoned by `deadzone_left/right`; rotation deltas are unfiltered.
	"""

	config_class = BiQuestHapticsConfig
	name = "bi_quest_haptics"

	def __init__(self, config: BiQuestHapticsConfig):
		super().__init__(config)
		self.config = config
		self._reader = None  # OculusReader instance
		self._connected = False
		self._last_left: _Pose | None = None
		self._last_right: _Pose | None = None

	# --- Teleoperator API ---
	@property
	def action_features(self) -> dict:
		# 12 DoF deltas (translation + rotation) + 4 button states (boolean as 0/1)
		return {
			"dtype": "float32",
			"shape": (16,),
			"names": {
				"left_delta_x": 0,
				"left_delta_y": 1,
				"left_delta_z": 2,
				"left_delta_rx": 3,
				"left_delta_ry": 4,
				"left_delta_rz": 5,
				"right_delta_x": 6,
				"right_delta_y": 7,
				"right_delta_z": 8,
				"right_delta_rx": 9,
				"right_delta_ry": 10,
				"right_delta_rz": 11,
				"left_X": 12,
				"left_Y": 13,
				"right_A": 14,
				"right_B": 15,
			},
		}

	@property
	def feedback_features(self) -> dict:
		# Two haptic amplitudes (left/right) in [0,1]
		return {
			"dtype": "float32",
			"shape": (2,),
			"names": {"left_haptic_amp": 0, "right_haptic_amp": 1},
		}

	@property
	def is_connected(self) -> bool:
		return self._connected

	def connect(self, calibrate: bool = True) -> None:
		if self.is_connected:
			raise DeviceAlreadyConnectedError("BiQuestHaptics is already connected.")
		try:
			from lerobot.teleoperators.quest_haptics.oculus_reader import OculusReader
		except Exception as e:
			raise ImportError(
				"OculusReader dependency is required for the BiQuestHaptics teleoperator.\n"
				"Ensure pure-python-adb is installed and adb is available on PATH."
			) from e

		self._reader = OculusReader(ip_address=self.config.ip_address, port=self.config.port)

		# Wait briefly for initial transforms from both controllers
		t0 = time.perf_counter()
		ok_left = ok_right = False
		while time.perf_counter() - t0 < 2.0:
			transforms, _ = self._reader.get_transformations_and_buttons()
			ok_left = ok_left or ('l' in transforms and isinstance(transforms.get('l'), np.ndarray))
			ok_right = ok_right or ('r' in transforms and isinstance(transforms.get('r'), np.ndarray))
			if ok_left and ok_right:
				break
			time.sleep(0.01)
		if not (ok_left and ok_right):
			self.disconnect()
			raise RuntimeError("Failed to receive initial transforms from both Quest controllers.")

		self._refresh_last_poses()
		self._connected = True

	@property
	def is_calibrated(self) -> bool:
		return True

	def calibrate(self) -> None:  # simple baseline reset
		self._refresh_last_poses()

	def configure(self) -> None:
		return None

	def _refresh_last_poses(self) -> None:
		assert self._reader is not None
		transforms, _ = self._reader.get_transformations_and_buttons()
		now = time.perf_counter()
		if 'l' in transforms and isinstance(transforms['l'], np.ndarray):
			Tl = np.asarray(transforms['l'], dtype=np.float32)
			self._last_left = _Pose(t=Tl[:3, 3].copy(), R=Tl[:3, :3].copy(), ts=now)
		if 'r' in transforms and isinstance(transforms['r'], np.ndarray):
			Tr = np.asarray(transforms['r'], dtype=np.float32)
			self._last_right = _Pose(t=Tr[:3, 3].copy(), R=Tr[:3, :3].copy(), ts=now)

	def _proc_translation(self, d: np.ndarray, dz: float) -> np.ndarray:
		x, y, z = map(float, d[:3])
		def dzf(a: float) -> float:
			return 0.0 if abs(a) < dz else a
		return np.array([dzf(x), dzf(y), dzf(z)], dtype=np.float32)

	def get_action(self) -> dict[str, Any]:
		if not self.is_connected:
			raise DeviceNotConnectedError("BiQuestHaptics is not connected. Call connect() first.")
		assert self._reader is not None
		transforms, buttons = self._reader.get_transformations_and_buttons()
		now = time.perf_counter()
		out: dict[str, Any] = {}

		# Left
		if 'l' in transforms and isinstance(transforms['l'], np.ndarray):
			Tl = np.asarray(transforms['l'], dtype=np.float32)
			t_l = Tl[:3, 3]; R_l = Tl[:3, :3]
			if self._last_left is None:
				self._last_left = _Pose(t=t_l.copy(), R=R_l.copy(), ts=now)
			d_t = t_l - self._last_left.t
			R_rel = R_l @ self._last_left.R.T
			d_r = _rotation_vector(R_rel)
			self._last_left = _Pose(t=t_l.copy(), R=R_l.copy(), ts=now)
			d_t = self._proc_translation(d_t, self.config.deadzone_left)
			out["left_delta_x"], out["left_delta_y"], out["left_delta_z"] = map(float, d_t)
			out["left_delta_rx"], out["left_delta_ry"], out["left_delta_rz"] = map(float, d_r)
		else:
			for k in ("left_delta_x","left_delta_y","left_delta_z","left_delta_rx","left_delta_ry","left_delta_rz"):
				out[k] = 0.0

		# Right
		if 'r' in transforms and isinstance(transforms['r'], np.ndarray):
			Tr = np.asarray(transforms['r'], dtype=np.float32)
			t_r = Tr[:3, 3]; R_r = Tr[:3, :3]
			if self._last_right is None:
				self._last_right = _Pose(t=t_r.copy(), R=R_r.copy(), ts=now)
			d_t = t_r - self._last_right.t
			R_rel = R_r @ self._last_right.R.T
			d_r = _rotation_vector(R_rel)
			self._last_right = _Pose(t=t_r.copy(), R=R_r.copy(), ts=now)
			d_t = self._proc_translation(d_t, self.config.deadzone_right)
			out["right_delta_x"], out["right_delta_y"], out["right_delta_z"] = map(float, d_t)
			out["right_delta_rx"], out["right_delta_ry"], out["right_delta_rz"] = map(float, d_r)
		else:
			for k in ("right_delta_x","right_delta_y","right_delta_z","right_delta_rx","right_delta_ry","right_delta_rz"):
				out[k] = 0.0


		# Button states (boolean to float 0/1)
		out["left_X"] = 1.0 if bool(buttons.get("X", False)) else 0.0
		out["left_Y"] = 1.0 if bool(buttons.get("Y", False)) else 0.0
		out["right_A"] = 1.0 if bool(buttons.get("A", False)) else 0.0
		out["right_B"] = 1.0 if bool(buttons.get("B", False)) else 0.0

		return out

	def send_feedback(self, feedback: dict[str, Any]) -> None:
		# Expect optional 'left_haptic_amp' and 'right_haptic_amp'
		if not self.is_connected or self._reader is None:
			return None
		try:
			amp_l = float(feedback.get("left_haptic_amp", 0.0))
			amp_r = float(feedback.get("right_haptic_amp", 0.0))
		except Exception:
			return None
		try:
			# Prefer direct haptic api if exposed on reader
			if hasattr(self._reader, "set_haptic_left"):
				self._reader.set_haptic_left(amp_l)
			if hasattr(self._reader, "set_haptic_right"):
				self._reader.set_haptic_right(amp_r)
		except Exception:
			pass
		return None

	def disconnect(self) -> None:
		if self._reader is not None:
			try:
				self._reader.stop()
			except Exception:
				pass
			# Ensure the Android app is closed as well
			try:
				if hasattr(self._reader, "force_stop_app"):
					self._reader.force_stop_app()
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
		# Translation-based intervention (ignore rotation for stability)
		action = self.get_action()
		is_int = (
			abs(action.get("left_delta_x", 0.0)) >= self.config.deadzone_left
			or abs(action.get("left_delta_y", 0.0)) >= self.config.deadzone_left
			or abs(action.get("left_delta_z", 0.0)) >= self.config.deadzone_left
			or abs(action.get("right_delta_x", 0.0)) >= self.config.deadzone_right
			or abs(action.get("right_delta_y", 0.0)) >= self.config.deadzone_right
			or abs(action.get("right_delta_z", 0.0)) >= self.config.deadzone_right
		)
		# Success / terminate mapping (reuse quest defaults if available)
		try:
			assert self._reader is not None
			_, buttons = self._reader.get_transformations_and_buttons()
		except Exception:
			buttons = {}
		terminate = bool(buttons.get('RJ', False) or buttons.get('LJ', False))
		success = bool(buttons.get('RThU', False) or buttons.get('LThU', False))
		return {
			TeleopEvents.IS_INTERVENTION: bool(is_int),
			TeleopEvents.TERMINATE_EPISODE: bool(terminate),
			TeleopEvents.SUCCESS: bool(success),
			TeleopEvents.RERECORD_EPISODE: False,
		}


