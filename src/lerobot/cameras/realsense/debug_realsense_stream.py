import argparse
import time

import cv2
import numpy as np

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug RealSense RGB+Depth streaming")
    parser.add_argument("--serial", type=str, default="f1181599", help="RealSense serial number or name")
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show OpenCV windows (requires GUI-enabled OpenCV build)",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run (0 = run until q/ESC)")
    parser.add_argument("--timeout_ms", type=float, default=1000.0, help="Async first-frame timeout (ms)")
    args = parser.parse_args()

    # Adjust this to your camera's serial or name if needed
    serial_or_name = args.serial

    config = RealSenseCameraConfig(
        serial_number_or_name=serial_or_name,
        use_depth=True,
        color_mode=ColorMode.BGR,
        width=640,
        height=480,
        fps=30,
    )

    cam = RealSenseCamera(config)
    # Disable warmup to avoid early read() failures; we'll rely on async loop.
    cam.connect(warmup=False)

    print("Connected RealSense:", cam)

    start = time.time()
    last_print = 0.0
    frame_count = 0
    fps_frame_count = 0
    fps_start = start

    def _safe_imshow(name: str, img: np.ndarray) -> None:
        if not args.display:
            return
        try:
            cv2.imshow(name, img)
        except cv2.error as e:
            # GUI not available (headless OpenCV build). Fall back to console-only.
            print(f"OpenCV GUI not available; disabling display. ({e})")
            args.display = False

    try:
        while True:
            # Use asynchronous API to test async_read_both()
            try:
                frame_dict = cam.async_read_both(timeout_ms=args.timeout_ms)
            except Exception as e:  # noqa: BLE001
                print("async_read_both failed, skipping frame:", e)
                time.sleep(0.01)
                continue

            color = frame_dict.get("color")
            depth = frame_dict.get("depth")
            
            frame_count += 1
            fps_frame_count += 1

            # Print shapes and FPS occasionally for debugging
            now = time.time()
            if now - last_print > 1.0:
                elapsed_fps = now - fps_start
                current_fps = fps_frame_count / elapsed_fps if elapsed_fps > 0 else 0
                color_shape = None if color is None else tuple(color.shape)
                depth_shape = None if depth is None else tuple(depth.shape)
                print(f"color={color_shape} depth={depth_shape} | FPS={current_fps:.1f}")
                last_print = now
                # Reset FPS counter for next window
                fps_frame_count = 0
                fps_start = now

            # Visualize color
            if color is not None:
                _safe_imshow("RealSense Color (async)", color)

            # Visualize depth as a colormap
            if depth is not None:
                # Normalize depth for visualization
                depth_vis = depth.astype(np.float32)
                # Clip far values for better contrast if needed
                max_val = np.percentile(depth_vis, 95)
                if max_val <= 0:
                    max_val = 1.0
                depth_vis = np.clip(depth_vis, 0, max_val)
                depth_vis = (depth_vis / max_val * 255.0).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                _safe_imshow("RealSense Depth (async)", depth_color)

            if args.display:
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    # ESC or q to quit
                    break

            if args.duration > 0 and (now - start) >= args.duration:
                break

            # Small sleep to avoid maxing out CPU
            time.sleep(0.001)

    finally:
        # Print final stats
        elapsed_total = time.time() - start
        avg_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
        print(f"\n=== Session Summary ===")
        print(f"Total frames: {frame_count}")
        print(f"Duration: {elapsed_total:.2f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        
        cam.disconnect()
        if args.display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()
