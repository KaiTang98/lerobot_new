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


def _R_to_quaternion(R_rel: np.ndarray) -> np.ndarray:
	"""Convert a rotation matrix to a quaternion (x, y, z, w).

	Uses SciPy if available; otherwise applies a numerically stable closed-form
	conversion. Returns a float32 array of shape (4,).
	"""
	R = np.asarray(R_rel, dtype=float)
	# Fast path using SciPy
	try:  # pragma: no cover
		from scipy.spatial.transform import Rotation as _SciRot  # type: ignore
		quat = _SciRot.from_matrix(R).as_quat()  # (x, y, z, w)
		return np.asarray(quat, dtype=np.float32)
	except Exception:
		pass

	# Fallback: explicit quaternion from rotation matrix.
	m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
	m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
	m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
	trace = m00 + m11 + m22
	if trace > 0.0:
		s = np.sqrt(trace + 1.0) * 2.0  # s = 4*qw
		qw = 0.25 * s
		qx = (m21 - m12) / s
		qy = (m02 - m20) / s
		qz = (m10 - m01) / s
	else:
		if m00 > m11 and m00 > m22:
			s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0  # s = 4*qx
			qw = (m21 - m12) / s
			qx = 0.25 * s
			qy = (m01 + m10) / s
			qz = (m02 + m20) / s
		elif m11 > m22:
			s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0  # s = 4*qy
			qw = (m02 - m20) / s
			qx = (m01 + m10) / s
			qy = 0.25 * s
			qz = (m12 + m21) / s
		else:
			s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0  # s = 4*qz
			qw = (m10 - m01) / s
			qx = (m02 + m20) / s
			qy = (m12 + m21) / s
			qz = 0.25 * s
	quat = np.array([qx, qy, qz, qw], dtype=np.float64)
	# Normalize to guard against numerical drift
	quat /= np.linalg.norm(quat) + 1e-12
	return quat.astype(np.float32)


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
		self._init_left: _Pose | None = None
		self._init_right: _Pose | None = None
		self._clear_init_poses()

		# # Activation and reference transforms for relative motion
		self._teleop_flag_left: bool = False
		self._teleop_flag_right: bool = False
		# self._init_operation_T_left: np.ndarray | None = None
		# self._init_operation_T_right: np.ndarray | None = None


	# --- Teleoperator API ---
	@property
	def action_features(self) -> dict:
	
		# 46 DoF (positions + quaternions + buttons/axes)
		return {
			"dtype": "float32",
			"shape": (48,),
			"names": {
				# Position and orientation (quaternion) of left controller
				"l_x": 0,
				"l_y": 1,
				"l_z": 2,
				"l_qx": 3,
				"l_qy": 4,
				"l_qz": 5,
				"l_qw": 6,
				"l_i_x": 7,
				"l_i_y": 8,
				"l_i_z": 9,
				"l_i_qx": 10,
				"l_i_qy": 11,
				"l_i_qz": 12,
				"l_i_qw": 13,

				# Position and orientation (quaternion) of right controller
				"r_x": 14,
				"r_y": 15,
				"r_z": 16,
				"r_qx": 17,
				"r_qy": 18,
				"r_qz": 19,
				"r_qw": 20,
				"r_i_x": 21,
				"r_i_y": 22,
				"r_i_z": 23,
				"r_i_qx": 24,
				"r_i_qy": 25,
				"r_i_qz": 26,
				"r_i_qw": 27,

				# Joystick and trigger states
				"l_joystick_x": 28,
				"l_joystick_y": 29,
				"l_trigger_up": 30,
				"l_trigger_down": 31,
				"r_joystick_x": 32,
				"r_joystick_y": 33,
				"r_trigger_up": 34,
				"r_trigger_down": 35,

				# Button states
				"l_X": 36,
				"l_Y": 37,
				"l_joystick_press": 38,
				"l_trigger_up_press": 39,
				"l_trigger_down_press": 40,
				"r_A": 41,
				"r_B": 42,
				"r_joystick_press": 43,
				"r_trigger_up_press": 44,
				"r_trigger_down_press": 45,
				"l_teleop_active": 46,
				"r_teleop_active": 47,
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
			time.sleep(0.1)
		if not (ok_left and ok_right):
			self.disconnect()
			raise RuntimeError("Failed to receive initial transforms from both Quest controllers.")

		self._clear_init_poses()
		self._connected = True


	@property
	def is_calibrated(self) -> bool:
		return True


	def calibrate(self) -> None:  # simple baseline reset
		self._refresh_last_poses()


	def configure(self) -> None:
		return None


	# def _refresh_last_poses(self) -> None:
	# 	assert self._reader is not None
	# 	transforms, _ = self._reader.get_transformations_and_buttons()
	# 	now = time.perf_counter()
	# 	if 'l' in transforms and isinstance(transforms['l'], np.ndarray):
	# 		Tl = np.asarray(transforms['l'], dtype=np.float32)
	# 		self._last_left = _Pose(t=Tl[:3, 3].copy(), R=Tl[:3, :3].copy(), ts=now)
	# 	if 'r' in transforms and isinstance(transforms['r'], np.ndarray):
	# 		Tr = np.asarray(transforms['r'], dtype=np.float32)
	# 		self._last_right = _Pose(t=Tr[:3, 3].copy(), R=Tr[:3, :3].copy(), ts=now)


	def _clear_init_poses(self) -> None:
		self._init_left = None
		self._init_right = None


	def get_action(self) -> dict[str, Any]:
		if not self.is_connected:
			raise DeviceNotConnectedError("BiQuestHaptics is not connected. Call connect() first.")
		assert self._reader is not None

		transforms, buttons = self._reader.get_transformations_and_buttons()
		now = time.perf_counter()
		out: dict[str, Any] = {}

		# Left controller absolute pose (position + quaternion) and initial reference
		if 'l' in transforms and isinstance(transforms['l'], np.ndarray):
			Tl = np.asarray(transforms['l'], dtype=np.float32)
			pos_l = Tl[:3, 3]
			R_l = Tl[:3, :3]
			quat_l = _R_to_quaternion(R_l)  # (qx,qy,qz,qw)
			out["l_x"], out["l_y"], out["l_z"] = map(float, pos_l)
			out["l_qx"], out["l_qy"], out["l_qz"], out["l_qw"] = map(float, quat_l)
		else:
			for k in ("l_x","l_y","l_z","l_qx","l_qy","l_qz","l_qw","l_i_x","l_i_y","l_i_z","l_i_qx","l_i_qy","l_i_qz","l_i_qw"):
				out[k] = 0.0

		# Right controller absolute pose
		if 'r' in transforms and isinstance(transforms['r'], np.ndarray):
			Tr = np.asarray(transforms['r'], dtype=np.float32)
			pos_r = Tr[:3, 3]
			R_r = Tr[:3, :3]
			quat_r = _R_to_quaternion(R_r)
			out["r_x"], out["r_y"], out["r_z"] = map(float, pos_r)
			out["r_qx"], out["r_qy"], out["r_qz"], out["r_qw"] = map(float, quat_r)
		else:
			for k in ("r_x","r_y","r_z","r_qx","r_qy","r_qz","r_qw","r_i_x","r_i_y","r_i_z","r_i_qx","r_i_qy","r_i_qz","r_i_qw"):
				out[k] = 0.0

		# Left controller button states (boolean to float 0/1)
		out["l_X"] = 1.0 if bool(buttons.get("X", False)) else 0.0
		out["l_Y"] = 1.0 if bool(buttons.get("Y", False)) else 0.0
		out["l_joystick_press"] = 1.0 if bool(buttons.get("LJ", False)) else 0.0
		out["l_trigger_up_press"] = 1.0 if bool(buttons.get("LG", False)) else 0.0
		out["l_trigger_down_press"] = 1.0 if bool(buttons.get("LTr", False)) else 0.0
		
		# Right controller button states	
		out["r_A"] = 1.0 if bool(buttons.get("A", False)) else 0.0
		out["r_B"] = 1.0 if bool(buttons.get("B", False)) else 0.0
		out["r_joystick_press"] = 1.0 if bool(buttons.get("RJ", False)) else 0.0
		out["r_trigger_up_press"] = 1.0 if bool(buttons.get("RG", False)) else 0.0
		out["r_trigger_down_press"] = 1.0 if bool(buttons.get("RTr", False)) else 0.0

		# print("l_trigger_up_press:", out["l_trigger_up_press"])
		# print("l_trigger_down_press:", out["l_trigger_down_press"])
		# print("r_trigger_up_press:", out["r_trigger_up_press"])
		# print("r_trigger_down_press:", out["r_trigger_down_press"])

		# when "l_X" is pressed, start left controller pose control, store initial pose
		if out["l_X"] == 1.0:
			self._teleop_flag_left = True
			if self._init_left is None:
				if 'l' in transforms and isinstance(transforms['l'], np.ndarray):
					Tl = np.asarray(transforms['l'], dtype=np.float32)
					self._init_left = _Pose(t=Tl[:3, 3].copy(), R=Tl[:3, :3].copy(), ts=now)

		# when "r_A" is pressed, start right controller pose control, store initial pose
		if out["r_A"] == 1.0:
			self._teleop_flag_right = True
			if self._init_right is None:
				if 'r' in transforms and isinstance(transforms['r'], np.ndarray):
					Tr = np.asarray(transforms['r'], dtype=np.float32)
					self._init_right = _Pose(t=Tr[:3, 3].copy(), R=Tr[:3, :3].copy(), ts=now)

		if self._teleop_flag_left:
			out["l_teleop_active"] = 1.0
		if self._teleop_flag_right:
			out["r_teleop_active"] = 1.0

		# send the initial reference poses if available
		if self._init_left is not None:
			out["l_i_x"], out["l_i_y"], out["l_i_z"] = map(float, self._init_left.t)
			quat_init_l = _R_to_quaternion(self._init_left.R)
			out["l_i_qx"], out["l_i_qy"], out["l_i_qz"], out["l_i_qw"] = map(float, quat_init_l)
		else:
			for k in ("l_i_x","l_i_y","l_i_z","l_i_qx","l_i_qy","l_i_qz","l_i_qw"):
				out[k] = 0.0

		if self._init_right is not None:
			out["r_i_x"], out["r_i_y"], out["r_i_z"] = map(float, self._init_right.t)
			quat_init_r = _R_to_quaternion(self._init_right.R)
			out["r_i_qx"], out["r_i_qy"], out["r_i_qz"], out["r_i_qw"] = map(float, quat_init_r)
		else:
			for k in ("r_i_x","r_i_y","r_i_z","r_i_qx","r_i_qy","r_i_qz","r_i_qw"):
				out[k] = 0.0

		# clear initial poses when "l_Y" is pressed
		if out["l_Y"] == 1.0:
			self._init_left = None
			self._teleop_flag_left = False
		if out["r_B"] == 1.0:
			self._init_right = None
			self._teleop_flag_right = False

		return out


	def send_feedback(self, feedback: dict[str, Any]) -> None:
		# Expect optional 'left_haptic_amp' and 'right_haptic_amp'
		if not self.is_connected or self._reader is None:
			return None
		try:
			fx_left = float(feedback.get("curFT_x_A", 0.0))
			fy_left = float(feedback.get("curFT_y_A", 0.0))
			fz_left = float(feedback.get("curFT_z_A", 0.0))
			fx_right = float(feedback.get("curFT_x_B", 0.0))
			fy_right = float(feedback.get("curFT_y_B", 0.0))
			fz_right = float(feedback.get("curFT_z_B", 0.0))

			amp_l = np.clip(np.max([abs(fx_left), abs(fy_left), abs(fz_left)]) / self.config.haptic_force_scale, 0.0, 1.0)
			amp_r = np.clip(np.max([abs(fx_right), abs(fy_right), abs(fz_right)]) / self.config.haptic_force_scale, 0.0, 1.0)
			# amp_l = float(feedback.get("left_haptic_amp", 0.0))
			# amp_r = float(feedback.get("right_haptic_amp", 0.0))
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


