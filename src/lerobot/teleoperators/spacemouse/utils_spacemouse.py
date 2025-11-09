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

"""Small utility to sanity-check SpaceMouse input.

Run directly:

    python -m lerobot.teleoperators.spacemouse.utils_spacemouse \
        --device_path /dev/hidraw5 \
        --fps 100 \
        --duration 10 \
        --translation_scale 1.0 \
        --deadzone 0.02

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

from .config_spacemouse import SpacemouseConfig
from .spacemouse import SpacemouseTeleop
from ..utils import TeleopEvents
from ..teleoperator import Teleoperator
from lerobot.utils.robot_utils import busy_wait


def format_action(action: dict[str, float]) -> str:
    # Match the current spacemouse.py output keys
    keys = [
        "vx",
        "vy",
        "vz",
        "ox",
        "oy",
        "oz",
        "b1",
        "b2",
    ]
    parts: list[str] = []
    for k in keys:
        if k in action:
            v = action[k]
            try:
                parts.append(f"{k}={float(v): .3f}")
            except Exception:
                parts.append(f"{k}={v}")
    return " | ".join(parts)


def print_loop(teleop: Teleoperator, fps: int, duration: Optional[float]) -> None:
    teleop.connect()
    print("Connected to SpaceMouse. Reading...")
    start = time.perf_counter()
    target_period = 1.0 / max(1, fps)
    n = 0

    try:
        while True:
            loop_t0 = time.perf_counter()

            action = teleop.get_action()
            # Optional: also fetch teleop events
            events = {}
            if hasattr(teleop, "get_teleop_events"):
                try:
                    events = teleop.get_teleop_events()  # type: ignore[attr-defined]
                except Exception:
                    events = {}

            # Print both action and a compact event summary
            evt_str = ""
            if events:
                is_int = events.get(TeleopEvents.IS_INTERVENTION, False)
                evt_str = f" | intervention={bool(is_int)}"

            n += 1
            if n % max(1, fps // 5) == 0:  # print ~5 times per second
                print(f"{format_action(action)}{evt_str}")

            # pacing
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
    p = argparse.ArgumentParser(description="SpaceMouse print-loop utility")
    p.add_argument("--device_path", type=str, default=None, help="OS device path (e.g. /dev/hidrawX)")
    p.add_argument("--fps", type=int, default=100, help="Loop frequency")
    p.add_argument("--duration", type=float, default=None, help="Stop after N seconds (default: infinite)")
    p.add_argument("--translation_scale", type=float, default=1.0, help="Scale for v_x/v_y/v_z")
    p.add_argument("--deadzone", type=float, default=0.0, help="Deadzone below which values are zeroed")

    args = p.parse_args(argv)

    cfg = SpacemouseConfig(
        device_path=args.device_path,
        fps=max(50, int(args.fps)),
        translation_scale=float(args.translation_scale),
        deadzone=float(args.deadzone),
    )

    teleop = SpacemouseTeleop(cfg)
    try:
        print_loop(teleop, fps=cfg.fps, duration=args.duration)
        return 0
    except ImportError as e:
        print(str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
