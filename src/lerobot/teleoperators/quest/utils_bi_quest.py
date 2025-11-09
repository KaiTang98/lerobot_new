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

"""Small utility to sanity-check bimanual Meta Quest teleoperation.

Run directly:

    python -m lerobot.teleoperators.quest.utils_bi_quest \
        --fps 100 \
        --duration 60

For wireless ADB, add: --ip_address 192.168.x.y --port 5555

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

from .config_quest import BiQuestConfig
from .quest import BiQuestTeleop
from ..utils import TeleopEvents
from ..teleoperator import Teleoperator
from lerobot.utils.robot_utils import busy_wait


def format_action(action: dict[str, float]) -> str:
    keys = [
        "left_delta_x",
        "left_delta_y",
        "left_delta_z",
        "left_gripper",
        "right_delta_x",
        "right_delta_y",
        "right_delta_z",
        "right_gripper",
    ]
    parts: list[str] = []
    for k in keys:
        if k in action:
            v = action[k]
            if isinstance(v, (int, float)):
                parts.append(f"{k}={v: .3f}")
            else:
                parts.append(f"{k}={v}")
    return " | ".join(parts)


def print_loop(teleop: Teleoperator, fps: int, duration: Optional[float]) -> None:
    teleop.connect()
    print("Connected to Quest (bimanual). Reading...")
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
            # print ~5 times per second
            if n % max(1, fps // 5) == 0:
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
    p = argparse.ArgumentParser(description="Bi-Quest print-loop utility")
    p.add_argument("--ip_address", type=str, default=None, help="Wireless ADB IP (None for USB)")
    p.add_argument("--port", type=int, default=5555, help="Wireless ADB port")
    p.add_argument("--fps", type=int, default=100, help="Loop frequency")
    p.add_argument("--duration", type=float, default=None, help="Stop after N seconds (default: infinite)")
    p.add_argument("--left_scale", type=float, default=1.0, help="Left translation scale factor")
    p.add_argument("--right_scale", type=float, default=1.0, help="Right translation scale factor")
    p.add_argument("--left_deadzone", type=float, default=0.0, help="Left axis deadzone")
    p.add_argument("--right_deadzone", type=float, default=0.0, help="Right axis deadzone")
    p.add_argument("--invert_left_x", action="store_true", help="Invert left X axis")
    p.add_argument("--invert_left_y", action="store_true", help="Invert left Y axis")
    p.add_argument("--invert_left_z", action="store_true", help="Invert left Z axis")
    p.add_argument("--invert_right_x", action="store_true", help="Invert right X axis")
    p.add_argument("--invert_right_y", action="store_true", help="Invert right Y axis")
    p.add_argument("--invert_right_z", action="store_true", help="Invert right Z axis")
    p.add_argument("--no-gripper", dest="use_gripper", action="store_false", help="Disable gripper outputs")
    p.set_defaults(use_gripper=True)

    args = p.parse_args(argv)

    cfg = BiQuestConfig(
        ip_address=args.ip_address,
        port=int(args.port),
        fps=max(50, int(args.fps)),
        use_gripper=bool(args.use_gripper),
        translation_scale_left=float(args.left_scale),
        translation_scale_right=float(args.right_scale),
        invert_left_x=bool(args.invert_left_x),
        invert_left_y=bool(args.invert_left_y),
        invert_left_z=bool(args.invert_left_z),
        invert_right_x=bool(args.invert_right_x),
        invert_right_y=bool(args.invert_right_y),
        invert_right_z=bool(args.invert_right_z),
        deadzone_left=float(args.left_deadzone),
        deadzone_right=float(args.right_deadzone),
    )

    teleop = BiQuestTeleop(cfg)
    try:
        print_loop(teleop, fps=cfg.fps, duration=args.duration)
        return 0
    except ImportError as e:
        print(str(e))
        print("Hint: pip install pure-python-adb; ensure adb is installed and on PATH")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
