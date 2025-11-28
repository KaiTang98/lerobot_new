#!/usr/bin/env python

"""Utility to benchmark RealSense camera FPS (sync vs async).

Example usages:

  # Basic synchronous read benchmark for 5s
  python -m lerobot.cameras.realsense.utils_realsense_fps \
      --serial f1181599 --duration 5

  # Asynchronous benchmark at requested 30 fps, display frames
  python -m lerobot.cameras.realsense.utils_realsense_fps \
      --serial f1181599 --mode async --fps 30 --width 640 --height 480 --duration 5 --display

  # Depth + color synchronous benchmark
  python -m lerobot.cameras.realsense.utils_realsense_fps \
      --serial f1181599 --use-depth --duration 4

Notes:
  - "sync" uses blocking `read()` calls (highest per-call latency, measures end-to-end capture).
  - "async" uses `async_read()` which waits on the background thread for a *new* frame (current implementation still blocks until fresh frame).
  - Warmup is shortened by default (1s) to reduce startup delay; you can override with --warmup-s.
  - If pyrealsense2 is not installed or the device is absent, the script will exit with a clear error message.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import List

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # display disabled if OpenCV not present

from .configuration_realsense import RealSenseCameraConfig
from .camera_realsense import RealSenseCamera


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark RealSense camera FPS.")
    p.add_argument("--serial", required=True, help="Serial number OR unique device name.")
    p.add_argument("--width", type=int, default=640, help="Capture width.")
    p.add_argument("--height", type=int, default=480, help="Capture height.")
    p.add_argument("--fps", type=int, default=30, help="Requested FPS.")
    p.add_argument("--duration", type=float, default=5.0, help="Benchmark duration in seconds.")
    p.add_argument("--mode", choices=["sync", "async"], default="sync", help="Read mode: sync or async.")
    p.add_argument("--use-depth", action="store_true", help="Enable depth stream as well.")
    p.add_argument("--display", action="store_true", help="Display frames in a window (may reduce FPS).")
    p.add_argument("--warmup-s", type=float, default=1.0, help="Warmup time in seconds before benchmarking.")
    p.add_argument("--no-warmup", action="store_true", help="Disable warmup entirely.")
    p.add_argument("--verbose", action="store_true", help="Print per-frame timings.")
    return p.parse_args()


def _maybe_display(win: str, frame: np.ndarray, display: bool) -> None:
    if not display:
        return
    if cv2 is None:
        return
    try:
        cv2.imshow(win, frame)
        cv2.waitKey(1)
    except Exception:
        pass


def benchmark_sync(cam: RealSenseCamera, duration: float, display: bool, verbose: bool) -> dict:
    t0 = time.perf_counter()
    frame_count = 0
    frame_times: List[float] = []
    while True:
        loop_start = time.perf_counter()
        frame = cam.read(timeout_ms=500)
        frame_count += 1
        _maybe_display("RealSense-sync", frame, display)
        loop_dt = time.perf_counter() - loop_start
        frame_times.append(loop_dt)
        if verbose:
            print(f"Frame {frame_count:04d} dt={loop_dt*1e3:.2f}ms")
        if time.perf_counter() - t0 >= duration:
            break
    elapsed = time.perf_counter() - t0
    return _summarize(elapsed, frame_count, frame_times, mode="sync")


def benchmark_async(cam: RealSenseCamera, duration: float, display: bool, verbose: bool) -> dict:
    t0 = time.perf_counter()
    frame_count = 0
    frame_times: List[float] = []
    while True:
        loop_start = time.perf_counter()
        frame = cam.async_read(timeout_ms=1000)
        frame_count += 1
        _maybe_display("RealSense-async", frame, display)
        loop_dt = time.perf_counter() - loop_start
        frame_times.append(loop_dt)
        if verbose:
            print(f"Frame {frame_count:04d} wait+copy={loop_dt*1e3:.2f}ms")
        if time.perf_counter() - t0 >= duration:
            break
    elapsed = time.perf_counter() - t0
    return _summarize(elapsed, frame_count, frame_times, mode="async")


def _summarize(elapsed: float, frames: int, frame_times: List[float], mode: str) -> dict:
    fps = frames / elapsed if elapsed > 0 else 0.0
    avg_ms = statistics.mean(frame_times) * 1e3 if frame_times else 0.0
    p95_ms = (np.percentile(frame_times, 95) * 1e3) if frame_times else 0.0
    return {
        "mode": mode,
        "frames": frames,
        "elapsed_s": elapsed,
        "fps": fps,
        "avg_frame_ms": avg_ms,
        "p95_frame_ms": p95_ms,
    }


def main() -> None:
    args = parse_args()

    try:
        config = RealSenseCameraConfig(
            serial_number_or_name=args.serial,
            fps=args.fps,
            width=args.width,
            height=args.height,
            use_depth=args.use_depth,
            warmup_s=args.warmup_s,
        )
        cam = RealSenseCamera(config)
    except Exception as e:
        print(f"[ERROR] Failed to initialize camera config: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        cam.connect(warmup=not args.no_warmup)
    except Exception as e:
        print(f"[ERROR] Failed to connect to camera: {e}", file=sys.stderr)
        sys.exit(3)

    print(f"Connected to {cam} (requested {args.width}x{args.height}@{args.fps}fps, mode={args.mode})")
    if args.display and cv2 is None:
        print("[WARN] OpenCV not available; --display ignored.")

    try:
        if args.mode == "sync":
            stats = benchmark_sync(cam, args.duration, args.display, args.verbose)
        else:
            stats = benchmark_async(cam, args.duration, args.display, args.verbose)
    finally:
        try:
            cam.disconnect()
        except Exception:
            pass
        if args.display and cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    print("\n=== RealSense FPS Benchmark ===")
    print(f"Mode:            {stats['mode']}")
    print(f"Frames captured: {stats['frames']}")
    print(f"Elapsed (s):     {stats['elapsed_s']:.3f}")
    print(f"Effective FPS:   {stats['fps']:.2f}")
    print(f"Avg frame wait:  {stats['avg_frame_ms']:.2f} ms")
    print(f"P95 frame wait:  {stats['p95_frame_ms']:.2f} ms")

    # Simple heuristic hints
    target_fps = args.fps
    if stats['fps'] < target_fps * 0.7:
        print("[HINT] Observed FPS << requested. Potential causes: USB bandwidth, depth enabled, display overhead, blocking async_read design.")
    elif stats['fps'] < target_fps * 0.9:
        print("[HINT] Slight FPS drop vs requested; consider reducing resolution or disabling depth.")


if __name__ == "__main__":
    main()
