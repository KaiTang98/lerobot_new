#!/usr/bin/env python3
"""Test SAM2 integration with RealSense camera for fabric segmentation.

This script tests the SAM2CameraRunner with your actual L515 camera.
It will display the RGB image with the fabric mask overlaid.
"""

import sys
import os
import argparse

# Add external/sam2 to path so we can import sam2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "external/sam2"))

import numpy as np
import cv2

from external.sam2_api import create_sam2_camera_runner
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig


def main():
    parser = argparse.ArgumentParser(description="Test SAM2 with RealSense")
    parser.add_argument("--serial", type=str, default="f1181599", help="RealSense serial number")
    parser.add_argument("--checkpoint", type=str, 
                       default="external/sam2/checkpoints/sam2.1_hiera_tiny.pt",
                       help="SAM2 checkpoint path")
    parser.add_argument("--point-x", type=int, default=320, help="Initial prompt point X")
    parser.add_argument("--point-y", type=int, default=240, help="Initial prompt point Y")
    parser.add_argument("--use-box", action="store_true", help="Use box prompt instead of point")
    parser.add_argument("--box", type=int, nargs=4, default=[200, 150, 440, 330],
                       help="Box prompt [x1 y1 x2 y2]")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    args = parser.parse_args()
    
    print("=== SAM2 + RealSense Integration Test ===\n")
    
    # 1. Initialize RealSense camera
    print(f"1. Connecting to RealSense camera {args.serial}...")
    config = RealSenseCameraConfig(
        serial_number_or_name=args.serial,
        fps=30,
        width=640,
        height=480,
        use_depth=False,  # We only need RGB for SAM2
        color_mode=ColorMode.RGB,
    )
    camera = RealSenseCamera(config)
    camera.connect()
    print(f"   ✓ Connected: {camera}\n")
    
    # 2. Create SAM2 runner
    print("2. Creating SAM2 camera runner...")
    if args.use_box:
        print(f"   Using box prompt: {args.box}")
        sam_runner = create_sam2_camera_runner(
            checkpoint_path=args.checkpoint,
            model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
            initial_prompt_box=args.box,
            device="cuda",
        )
    else:
        print(f"   Using point prompt: [{args.point_x}, {args.point_y}]")
        sam_runner = create_sam2_camera_runner(
            checkpoint_path=args.checkpoint,
            model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
            initial_prompt_point=[args.point_x, args.point_y],
            device="cuda",
        )
    print("   ✓ SAM2 runner created\n")
    
    # 3. Run tracking loop
    print(f"3. Running SAM2 tracking for {args.duration} seconds...")
    print("   Saving images every 100 frames to 'sam2_output/' directory\n")
    
    # Create output directory
    output_dir = "sam2_output"
    os.makedirs(output_dir, exist_ok=True)
    
    import time
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            # Get RGB frame
            rgb = camera.read()
            if rgb is None:
                print("   Warning: Failed to read frame, retrying...")
                continue
            
            # Run SAM2
            t0 = time.time()
            mask = sam_runner(rgb)
            sam_time = time.time() - t0
            
            # Create visualization
            # Convert RGB to BGR for OpenCV saving
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            overlay = bgr.copy()
            
            # Apply green overlay on mask
            overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
            
            # Draw prompt on first frame
            if frame_count == 0:
                if args.use_box:
                    x1, y1, x2, y2 = args.box
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                else:
                    cv2.circle(overlay, (args.point_x, args.point_y), 5, (0, 0, 255), -1)
            
            # Add text info
            cv2.putText(overlay, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"SAM2: {sam_time*1000:.1f}ms ({1/sam_time:.1f} FPS)", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Mask pixels: {mask.sum()}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save every 100 frames
            if frame_count % 100 == 0:
                filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(filename, overlay)
                print(f"   Saved: {filename} (SAM2={sam_time*1000:.1f}ms, mask_pixels={mask.sum()})")
            
            # Print stats every 30 frames
            if frame_count % 30 == 0 and frame_count % 100 != 0:
                print(f"   Frame {frame_count}: SAM2={sam_time*1000:.1f}ms, "
                      f"mask_pixels={mask.sum()}")
            
            frame_count += 1
            
            # Check duration
            if time.time() - start_time > args.duration:
                break
    
    except KeyboardInterrupt:
        print("\n   Interrupted by user")
    
    finally:
        # Cleanup
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n4. Test Summary:")
        print(f"   Total frames: {frame_count}")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   SAM2 initialized: {sam_runner.initialized}")
        print(f"   Saved images: {output_dir}/frame_*.jpg")
        
        camera.disconnect()
        
        print("\n✓ Test complete!")
        print(f"✓ Check saved images in: {output_dir}/")



if __name__ == "__main__":
    main()
