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

"""Utility to sanity-check bimanual SpaceMouse input.

Example:

    python -m lerobot.teleoperators.bi_spacemouse.utils_bi_spacemouse \
        --left_device_path /dev/hidraw5 \
        --right_device_path /dev/hidraw6 \
        --fps 100 \
        --duration 10 \
        --translation_scale_left 1.5 \
        --rotation_scale_left 0.7 \
        --invert_left_vx --invert_right_ox

Prints the processed action dict at ~5 Hz plus an intervention flag.
Press Ctrl+C (SIGINT) to stop early.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

from .config_bi_spacemouse import BiSpacemouseConfig
from .bi_spacemouse import BiSpacemouseTeleop
from ..utils import TeleopEvents
from ..teleoperator import Teleoperator
from lerobot.utils.robot_utils import busy_wait


def format_action(action: dict[str, float]) -> str:
    # Expected keys per new schema (8 per side)
    left_keys = ["left_vx", "left_vy", "left_vz", "left_ox", "left_oy", "left_oz", "left_b1", "left_b2"]
    right_keys = ["right_vx", "right_vy", "right_vz", "right_ox", "right_oy", "right_oz", "right_b1", "right_b2"]
    parts: list[str] = []
    for group in (left_keys, right_keys):
        for k in group:
            if k in action:
                v = action[k]
                if isinstance(v, (int, float)):
                    parts.append(f"{k}={v: .3f}")
                else:
                    parts.append(f"{k}={v}")
    return " | ".join(parts)


def print_loop(teleop: Teleoperator, fps: int, duration: Optional[float]) -> None:
    teleop.connect()
    print("Connected to Bi-Spacemouse. Reading...")
    start = time.perf_counter()
    target_period = 1.0 / max(1, fps)
    n = 0

    try:
        while True:
            loop_t0 = time.perf_counter()

            action = teleop.get_action()
            events = {}
            if hasattr(teleop, "get_teleop_events"):
                try:
                    events = teleop.get_teleop_events()  # type: ignore[attr-defined]
                except Exception:
                    events = {}

            evt_str = ""
            if events:
                is_int = events.get(TeleopEvents.IS_INTERVENTION, False)
                evt_str = f" | intervention={bool(is_int)}"

            n += 1
            if n % max(1, fps // 5) == 0:  # print ~5 times per second
                print(f"{format_action(action)}{evt_str}")

            elapsed = time.perf_counter() - loop_t0
            busy_wait(max(0.0, target_period - elapsed))

            if duration is not None and (time.perf_counter() - start) >= duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
        used = time.perf_counter() - start
        avg_hz = n / used if used > 0 else float("nan")
        print(f"Disconnected. Samples={n}, avg rate={avg_hz:.1f} Hz")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bimanual SpaceMouse print-loop utility")
    # Device paths & timing
    p.add_argument("--left_device_path", type=str, default=None, help="Left OS device path (e.g. /dev/hidrawX)")
    p.add_argument("--right_device_path", type=str, default=None, help="Right OS device path (e.g. /dev/hidrawY)")
    p.add_argument("--fps", type=int, default=100, help="Loop frequency (Hz)")
    p.add_argument("--duration", type=float, default=None, help="Stop after N seconds (default: infinite)")

    # Scaling
    p.add_argument("--translation_scale_left", type=float, default=1.0, help="Left translation scale")
    p.add_argument("--translation_scale_right", type=float, default=1.0, help="Right translation scale")
    p.add_argument("--rotation_scale_left", type=float, default=1.0, help="Left rotation scale")
    p.add_argument("--rotation_scale_right", type=float, default=1.0, help="Right rotation scale")

    # Deadzones
    p.add_argument("--deadzone_left", type=float, default=0.0, help="Left translation deadzone")
    p.add_argument("--deadzone_right", type=float, default=0.0, help="Right translation deadzone")


    args = p.parse_args(argv)

    cfg = BiSpacemouseConfig(
        left_device_path=args.left_device_path,
        right_device_path=args.right_device_path,
        fps=max(50, int(args.fps)),
        translation_scale_left=float(args.translation_scale_left),
        translation_scale_right=float(args.translation_scale_right),
        rotation_scale_left=float(args.rotation_scale_left),
        rotation_scale_right=float(args.rotation_scale_right),
        deadzone_left=float(args.deadzone_left),
        deadzone_right=float(args.deadzone_right),
    )

    teleop = BiSpacemouseTeleop(cfg)
    try:
        print_loop(teleop, fps=cfg.fps, duration=args.duration)
        return 0
    except ImportError as e:
        print(str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
