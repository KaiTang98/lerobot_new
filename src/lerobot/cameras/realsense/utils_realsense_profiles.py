#!/usr/bin/env python

"""List supported RealSense color/depth stream profiles for a given device.

Example:
  python -m lerobot.cameras.realsense.utils_realsense_profiles --serial f1181599
"""

from __future__ import annotations

import argparse
import sys

try:
    import pyrealsense2 as rs  # type: ignore
except Exception as e:
    print(f"[ERROR] pyrealsense2 not available: {e}")
    sys.exit(2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List supported RealSense stream profiles")
    p.add_argument("--serial", required=True, help="Serial number OR unique device name.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        sys.exit(3)

    device = None
    # Try match by serial first, then by name
    for dev in devices:
        if dev.get_info(rs.camera_info.serial_number) == args.serial:
            device = dev
            break
    if device is None:
        for dev in devices:
            if dev.get_info(rs.camera_info.name) == args.serial:
                device = dev
                break
    if device is None:
        print(f"Device '{args.serial}' not found. Available:")
        for dev in devices:
            print(f"  - {dev.get_info(rs.camera_info.name)} (SN={dev.get_info(rs.camera_info.serial_number)})")
        sys.exit(4)

    print(f"Device: {device.get_info(rs.camera_info.name)} (SN={device.get_info(rs.camera_info.serial_number)})")
    sensors = device.query_sensors()
    for sensor in sensors:
        try:
            sensor_name = sensor.get_info(rs.camera_info.name)
        except Exception:
            sensor_name = "<unknown>"
        print(f"\nSensor: {sensor_name}")
        profiles = sensor.get_stream_profiles()
        color_profiles = []
        depth_profiles = []
        for p in profiles:
            if not p.is_video_stream_profile():
                continue
            vp = p.as_video_stream_profile()
            entry = (vp.stream_type(), vp.format(), vp.width(), vp.height(), vp.fps())
            if vp.stream_type() == rs.stream.color:
                color_profiles.append(entry)
            elif vp.stream_type() == rs.stream.depth:
                depth_profiles.append(entry)

        if color_profiles:
            print("  Color profiles:")
            # Deduplicate and sort
            seen = set()
            for st, fmt, w, h, f in sorted(color_profiles, key=lambda e: (e[2], e[3], e[4])):
                key = (w, h, f, fmt.name)
                if key in seen:
                    continue
                seen.add(key)
                print(f"    - {w}x{h} @{f} fps, format={fmt.name}")
        if depth_profiles:
            print("  Depth profiles:")
            seen = set()
            for st, fmt, w, h, f in sorted(depth_profiles, key=lambda e: (e[2], e[3], e[4])):
                key = (w, h, f, fmt.name)
                if key in seen:
                    continue
                seen.add(key)
                print(f"    - {w}x{h} @{f} fps, format={fmt.name}")


if __name__ == "__main__":
    main()
