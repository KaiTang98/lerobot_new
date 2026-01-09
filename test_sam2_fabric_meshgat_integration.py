#!/usr/bin/env python3
"""Integration test for SAM2 + FabricPointCloud + MeshGAT pipeline.

This script demonstrates the complete pipeline:
1. RealSense L515 captures RGB + depth
2. SAM2 segments the fabric
3. FabricPointCloud converts masked depth to centered pointcloud
4. MeshGAT predicts fabric mesh vertices

Run with:
    python test_sam2_fabric_meshgat_integration.py --help
"""

import sys
import os
import argparse

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "external/sam2"))

import numpy as np
import time

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.processor.factory import make_denso_meshgat_robot_observation_processor


def main():
    parser = argparse.ArgumentParser(description="Test complete SAM2 + MeshGAT pipeline")
    
    # Camera args
    parser.add_argument("--serial", type=str, default="f1181599",
                       help="RealSense serial number")
    
    # SAM2 args
    parser.add_argument("--sam2-checkpoint", type=str,
                       default="external/sam2/checkpoints/sam2.1_hiera_tiny.pt",
                       help="SAM2 checkpoint path")
    parser.add_argument("--sam2-point", type=int, nargs=2, default=[320, 240],
                       help="SAM2 initial point [x y]")
    
    # MeshGAT args
    parser.add_argument("--meshgat-checkpoint", type=str, required=True,
                       help="MeshGAT checkpoint path")
    parser.add_argument("--meshgat-config", type=str, required=True,
                       help="MeshGAT config path")
    parser.add_argument("--meshgat-template", type=str, default=None,
                       help="MeshGAT template path")
    
    # Processing args
    parser.add_argument("--target-points", type=int, default=1024,
                       help="Number of pointcloud points")
    parser.add_argument("--num-frames", type=int, default=100,
                       help="Number of frames to test")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SAM2 + FabricPointCloud + MeshGAT Integration Test")
    print("="*60)
    print()
    
    # 1. Initialize RealSense with depth
    print("1. Initializing RealSense L515...")
    config = RealSenseCameraConfig(
        serial_number_or_name=args.serial,
        fps=30,
        width=640,
        height=480,
        use_depth=True,  # ← Important: we need depth!
        color_mode=ColorMode.RGB,
    )
    camera = RealSenseCamera(config)
    camera.connect(warmup=False)  # Disable warmup to avoid early read failures
    
    # Get intrinsics and depth scale
    intrinsics = camera.config.camera_intrinsics
    depth_scale = camera.config.depth_scale
    
    print(f"   ✓ Connected: {camera}")
    print(f"   Intrinsics:\n{intrinsics}")
    print(f"   Depth scale: {depth_scale} m/unit")
    print()
    
    # 2. Create processor pipeline
    print("2. Creating processor pipeline...")
    print(f"   SAM2: {args.sam2_checkpoint}")
    print(f"   MeshGAT: {args.meshgat_checkpoint}")
    print(f"   Initial point: {args.sam2_point}")
    print(f"   Target points: {args.target_points}")
    
    processor = make_denso_meshgat_robot_observation_processor(
        # MeshGAT args
        checkpoint_path=args.meshgat_checkpoint,
        config_path=args.meshgat_config,
        template_path=args.meshgat_template,
        device="cuda",
        input_key="pcl",
        output_key="mesh_vertices",
        # FabricPointCloud + SAM2 args
        enable_pointcloud=True,
        camera_intrinsics=intrinsics,
        depth_scale=depth_scale,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_initial_point=args.sam2_point,
        target_num_points=args.target_points,
    )
    
    print("   ✓ Pipeline created with steps:")
    for i, step in enumerate(processor.steps):
        print(f"      {i+1}. {step.__class__.__name__}")
    print()
    
    # 3. Run test loop
    print(f"3. Processing {args.num_frames} frames...")
    print()
    
    timings = {
        "camera_read": [],
        "sam2": [],
        "pointcloud": [],
        "meshgat": [],
        "total": [],
    }
    
    for frame_idx in range(args.num_frames):
        t_start = time.time()
        
        # Read RGB + depth from camera using async_read_both (more reliable)
        # Retry a few times if needed
        t0 = time.time()
        rgb, depth = None, None
        for retry in range(3):
            try:
                result = camera.async_read_both(timeout_ms=1000)  # 1 second timeout
                rgb = result.get("color")
                depth = result.get("depth")
                if rgb is not None and depth is not None:
                    break
            except Exception as e:
                if retry < 2:
                    print(f"   Frame {frame_idx}: Read attempt {retry+1} failed, retrying...")
                    time.sleep(0.1)
                else:
                    print(f"   Frame {frame_idx}: Failed after 3 attempts: {e}")
        
        if rgb is None or depth is None:
            print(f"   Frame {frame_idx}: Failed to read RGB/depth, skipping...")
            continue
        
        t_read = time.time() - t0
        
        # Create observation dict
        observation = {
            "rgb": rgb,
            "depth": depth,
            "joint_pos": np.zeros(6, dtype=np.float32),  # Dummy state
        }
        
        # Process through pipeline
        t0 = time.time()
        try:
            processed_obs = processor(observation)
            t_process = time.time() - t0
        except Exception as e:
            print(f"   Frame {frame_idx}: Processing failed: {e}")
            continue
        
        t_total = time.time() - t_start
        
        # Extract results
        pcl = processed_obs.get("pcl")
        mesh_vertices = processed_obs.get("mesh_vertices")
        
        # Record timings
        timings["camera_read"].append(t_read)
        timings["total"].append(t_total)
        
        # Print progress
        if frame_idx % 10 == 0 or frame_idx < 5:
            print(f"   Frame {frame_idx}:")
            print(f"      Camera: {t_read*1000:.1f}ms")
            print(f"      Pipeline: {t_process*1000:.1f}ms")
            print(f"      Total: {t_total*1000:.1f}ms ({1/t_total:.1f} FPS)")
            if pcl is not None:
                print(f"      Pointcloud: shape={pcl.shape}, mean={pcl.mean(axis=0)}")
            if mesh_vertices is not None:
                print(f"      Mesh: shape={mesh_vertices.shape}, "
                      f"range=[{mesh_vertices.min():.3f}, {mesh_vertices.max():.3f}]")
            print()
    
    # 4. Summary
    print()
    print("4. Performance Summary:")
    print(f"   Frames processed: {len(timings['total'])}")
    
    if timings["total"]:
        avg_camera = np.mean(timings["camera_read"]) * 1000
        avg_total = np.mean(timings["total"]) * 1000
        avg_fps = 1.0 / np.mean(timings["total"])
        
        print(f"   Average camera read: {avg_camera:.1f}ms")
        print(f"   Average total time: {avg_total:.1f}ms")
        print(f"   Average FPS: {avg_fps:.1f}")
        
        # Check if real-time capable (30 FPS = 33.3ms per frame)
        if avg_total < 33.3:
            print(f"   ✓ Real-time capable at 30 FPS!")
        else:
            print(f"   ⚠ Not real-time at 30 FPS (need <33.3ms, got {avg_total:.1f}ms)")
    
    print()
    
    # 5. Cleanup
    camera.disconnect()
    print("✓ Test complete!")


if __name__ == "__main__":
    main()
