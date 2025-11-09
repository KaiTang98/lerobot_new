#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility to sanity‑check bimanual Quest teleoperation with haptics.

Example:

    python -m lerobot.teleoperators.quest_haptics.utils_quest_haptics \
        --fps 100 --duration 15 --ip_address 192.168.x.y --port 5555

Shows per‑frame translation & rotation deltas plus button states.
Sends a haptic pulse (0.5 left/right) every 1 second.
Press Ctrl+C to stop early.
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

from .config_quest_haptics import BiQuestHapticsConfig
from .quest_haptics import BiQuestHapticsTeleop
from ..utils import TeleopEvents
from ..teleoperator import Teleoperator
from lerobot.utils.robot_utils import busy_wait


def _format_action(action: dict[str, float], names: list[str]) -> str:
    parts: list[str] = []
    for k in names:
        v = action.get(k, 0.0)
        if isinstance(v, (int, float)):
            parts.append(f"{k}={v: .3f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)


def print_loop(teleop: Teleoperator, fps: int, duration: Optional[float]) -> None:
    teleop.connect()
    print("Connected to Quest (haptics). Reading...")
    start = time.perf_counter()
    target_period = 1.0 / max(1, fps)
    n = 0
    last_haptic = start

    # Resolve action names from teleop definition
    names_map = getattr(teleop, "action_features", {}).get("names", {})  # type: ignore
    ordered_names = [k for k, _ in sorted(names_map.items(), key=lambda kv: kv[1])] if names_map else []

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

            # Haptic pulse every 1 second (amplitude 0.5 both hands)
            now = time.perf_counter()
            if now - last_haptic >= 1.0:
                teleop.send_feedback({"left_haptic_amp": 0.5, "right_haptic_amp": 0.5})
                last_haptic = now

            evt_str = ""
            if events:
                is_int = events.get(TeleopEvents.IS_INTERVENTION, False)
                evt_str = f" | intervention={bool(is_int)}"

            n += 1
            # print ~5 times per second
            if n % max(1, fps // 5) == 0:
                print(f"{_format_action(action, ordered_names)}{evt_str}")

            elapsed = time.perf_counter() - loop_t0
            busy_wait(max(0.0, target_period - elapsed))

            if duration is not None and (time.perf_counter() - start) >= duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        # teleop.send_feedback({"left_haptic_amp": 0.0, "right_haptic_amp": 0.0})
        # time.sleep(1)  # allow haptic commands to be sent
        teleop.disconnect()
        used = time.perf_counter() - start
        avg_hz = n / used if used > 0 else float("nan")
        print(f"Disconnected. Samples={n}, avg rate={avg_hz:.1f} Hz")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bi-Quest Haptics print-loop utility")
    p.add_argument("--ip_address", type=str, default=None, help="Wireless ADB IP (None for USB)")
    p.add_argument("--port", type=int, default=5555, help="Wireless ADB port")
    p.add_argument("--fps", type=int, default=100, help="Loop frequency (Hz)")
    p.add_argument("--duration", type=float, default=None, help="Stop after N seconds (default: infinite)")
    p.add_argument("--deadzone_left", type=float, default=0.0, help="Left translation deadzone")
    p.add_argument("--deadzone_right", type=float, default=0.0, help="Right translation deadzone")

    args = p.parse_args(argv)

    cfg = BiQuestHapticsConfig(
        ip_address=args.ip_address,
        port=int(args.port),
        fps=max(50, int(args.fps)),
        deadzone_left=float(args.deadzone_left),
        deadzone_right=float(args.deadzone_right),
    )

    teleop = BiQuestHapticsTeleop(cfg)
    try:
        print_loop(teleop, fps=cfg.fps, duration=args.duration)
        return 0
    except ImportError as e:
        print(str(e))
        print("Hint: pip install pure-python-adb; ensure adb is installed and on PATH")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
