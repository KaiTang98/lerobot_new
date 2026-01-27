#!/usr/bin/env python
"""
Main application for standalone mesh detection.

Usage:
    python main.py --config config.yaml
    python main.py --camera-serial f1181599 --device cuda
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import yaml

from camera_manager import CameraManager
from mesh_pipeline import MeshPipeline
from visualizer import MeshVisualizer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Standalone Mesh Detection System")
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file"
    )
    
    # Override options
    parser.add_argument("--camera-serial", type=str, help="Camera serial number")
    parser.add_argument("--width", type=int, help="Frame width")
    parser.add_argument("--height", type=int, help="Frame height")
    parser.add_argument("--fps", type=int, help="Camera FPS")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Processing device")
    parser.add_argument("--no-3d", action="store_true", help="Disable 3D visualization")
    parser.add_argument("--save-dir", type=str, help="Output directory for saved meshes")
    
    return parser.parse_args()


def main():
    """Main application loop."""
    args = parse_args()
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Using default configuration")
        config = {}
    else:
        config = load_config(config_path)
    
    # Apply CLI overrides
    camera_cfg = config.get('camera', {})
    if args.camera_serial:
        camera_cfg['serial'] = args.camera_serial
    if args.width:
        camera_cfg['width'] = args.width
    if args.height:
        camera_cfg['height'] = args.height
    if args.fps:
        camera_cfg['fps'] = args.fps
    
    processing_cfg = config.get('processing', {})
    if args.device:
        processing_cfg['device'] = args.device
    
    viz_cfg = config.get('visualization', {})
    if args.no_3d:
        viz_cfg['enable_3d'] = False
    
    output_cfg = config.get('output', {})
    if args.save_dir:
        output_cfg['save_dir'] = args.save_dir
    
    # Extract configuration values with defaults
    camera_serial = camera_cfg.get('serial', 'f1181599')
    camera_color_width = camera_cfg.get('color_width', camera_cfg.get('width', 640))
    camera_color_height = camera_cfg.get('color_height', camera_cfg.get('height', 480))
    camera_depth_width = camera_cfg.get('depth_width', camera_cfg.get('width', 640))
    camera_depth_height = camera_cfg.get('depth_height', camera_cfg.get('height', 480))
    camera_fps = camera_cfg.get('fps', 30)
    
    sam2_cfg = config.get('sam2', {})
    sam2_checkpoint = sam2_cfg.get('checkpoint', 'external/sam2/checkpoints/sam2.1_hiera_tiny.pt')
    sam2_model_cfg = sam2_cfg.get('model_config', 'configs/sam2.1/sam2.1_hiera_t.yaml')
    sam2_initial_point = sam2_cfg.get('initial_point', [320, 240])
    sam2_initial_box = sam2_cfg.get('initial_box', None)
    sam2_auto_detect = sam2_cfg.get('auto_detect', False)
    
    fabric_detection_cfg = config.get('fabric_detection', {})
    tracking_cfg = config.get('tracking', {})
    
    # Extract depth filtering parameters (for filtered depth visualization)
    depth_min = fabric_detection_cfg.get('depth_min', 0.2)
    depth_max = fabric_detection_cfg.get('depth_max', 1.0)
    
    meshgat_cfg = config.get('meshgat', {})
    meshgat_checkpoint = meshgat_cfg.get('checkpoint')
    meshgat_config = meshgat_cfg.get('config')
    meshgat_template = meshgat_cfg.get('template', None)
    target_num_points = meshgat_cfg.get('target_num_points', 1024)
    
    device = processing_cfg.get('device', 'cuda')
    resample_method = processing_cfg.get('resample_method', 'random')
    processing_width = processing_cfg.get('processing_width', None)
    processing_height = processing_cfg.get('processing_height', None)
    
    enable_2d = viz_cfg.get('enable_2d', True)
    enable_3d = viz_cfg.get('enable_3d', True)
    show_original_rgb = viz_cfg.get('show_original_rgb', False)
    show_original_depth = viz_cfg.get('show_original_depth', False)
    show_filtered_depth = viz_cfg.get('show_filtered_depth', False)
    apply_static_mask_to_viz = viz_cfg.get('apply_static_mask_to_visualization', False)
    mesh_color = tuple(viz_cfg.get('mesh_color', [0.8, 0.2, 0.2]))
    pcl_color = tuple(viz_cfg.get('pointcloud_color', [0.2, 0.8, 0.2]))
    
    save_dir = output_cfg.get('save_dir', 'output')
    record_visualization = output_cfg.get('record_visualization', False)
    video_fps = output_cfg.get('video_fps', 30)
    
    # Make save_dir relative to this script's directory (mesh_detection/)
    script_dir = Path(__file__).parent
    save_dir = script_dir / save_dir
    
    # Validate required paths
    if not meshgat_checkpoint or not meshgat_config:
        print("Error: MeshGAT checkpoint and config paths are required")
        print("Please specify in config.yaml or use --config option")
        return 1
    
    # Create output directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Standalone Mesh Detection System")
    print("=" * 60)
    print(f"Camera: {camera_serial}")
    print(f"  Capture: Color {camera_color_width}x{camera_color_height}, Depth {camera_depth_width}x{camera_depth_height} @ {camera_fps}fps")
    if processing_width and processing_height:
        print(f"  Processing: Resized to {processing_width}x{processing_height}")
    else:
        print(f"  Processing: Native resolution")
    print(f"Device: {device}")
    print(f"SAM2: {sam2_checkpoint}")
    print(f"MeshGAT: {meshgat_checkpoint}")
    print(f"Output: {save_dir}")
    print("=" * 60)
    
    # Initialize camera
    print("\n[1/3] Initializing camera...")
    try:
        camera = CameraManager(
            serial=camera_serial,
            color_width=camera_color_width,
            color_height=camera_color_height,
            depth_width=camera_depth_width,
            depth_height=camera_depth_height,
            fps=camera_fps,
            use_depth=True,
        )
        camera.start()
        intrinsics = camera.get_intrinsics()
        depth_scale = camera.get_depth_scale()
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return 1
    
    # Initialize pipeline
    print("\n[2/3] Initializing processing pipeline...")
    try:
        # Initialize fabric detector if auto-detect is enabled
        fabric_detector = None
        static_mask_for_viz = None
        if sam2_auto_detect:
            from fabric_detector import FabricDetector, FabricDetectorConfig
            detector_config = FabricDetectorConfig(
                depth_min=fabric_detection_cfg.get('depth_min', 0.2),
                depth_max=fabric_detection_cfg.get('depth_max', 1.0),
                robot_saturation_max=fabric_detection_cfg.get('robot_saturation_threshold', 0.3),
                robot_value_range=(0.0, 1.0),
                table_saturation_max=fabric_detection_cfg.get('table_saturation_threshold', 0.2),
                table_value_min=0.7,
                table_depth_std_max=fabric_detection_cfg.get('table_flatness_threshold', 0.02),
                min_fabric_area=fabric_detection_cfg.get('min_fabric_area', 1000),
                min_confidence=tracking_cfg.get('min_confidence', 0.3),
                static_mask_path=fabric_detection_cfg.get('static_mask_path'),
                enable_debug=fabric_detection_cfg.get('enable_debug', False),
            )
            fabric_detector = FabricDetector(config=detector_config, depth_scale=depth_scale)
            # Get static mask for visualization filtering if enabled
            if apply_static_mask_to_viz and fabric_detector.static_mask is not None:
                static_mask_for_viz = fabric_detector.static_mask
                print(f"  Static mask will be applied to visualization")
            print(f"  Auto-detect mode: ENABLED (fabric detector initialized)")
        else:
            print(f"  Auto-detect mode: DISABLED (using fixed point {sam2_initial_point})")
        
        pipeline = MeshPipeline(
            sam2_checkpoint=sam2_checkpoint,
            sam2_model_cfg=sam2_model_cfg,
            meshgat_checkpoint=meshgat_checkpoint,
            meshgat_config=meshgat_config,
            camera_intrinsics=intrinsics,
            depth_scale=depth_scale,
            initial_point=sam2_initial_point,
            initial_box=sam2_initial_box,
            meshgat_template=meshgat_template,
            target_num_points=target_num_points,
            resample_method=resample_method,
            device=device,
            auto_detect=sam2_auto_detect,
            fabric_detector=fabric_detector,
            tracking_config=tracking_cfg,
        )
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        camera.stop()
        return 1
    
    # Initialize visualizer
    print("\n[3/3] Initializing visualizer...")
    try:
        visualizer = MeshVisualizer(
            window_name="Mesh Detection",
            enable_3d=enable_3d and enable_2d,  # Need 2D for event loop even if only 3D wanted
            mesh_color=mesh_color,
            pcl_color=pcl_color,
            show_original_rgb=show_original_rgb,
            show_original_depth=show_original_depth,
            show_filtered_depth=show_filtered_depth,
            depth_scale=depth_scale,
        )
        visualizer._enable_2d = enable_2d  # Store 2D flag for conditional display
        print(f"  2D visualization: {'ENABLED' if enable_2d else 'DISABLED'}")
        print(f"  3D visualization: {'ENABLED' if enable_3d else 'DISABLED'}")
        print(f"  Original RGB: {'ENABLED' if show_original_rgb else 'DISABLED'}")
        print(f"  Original Depth: {'ENABLED' if show_original_depth else 'DISABLED'}")
        print(f"  Filtered Depth: {'ENABLED' if show_filtered_depth else 'DISABLED'} (range: {depth_min:.2f}-{depth_max:.2f}m)")
        if apply_static_mask_to_viz and static_mask_for_viz is not None:
            print(f"  Static mask filter: ENABLED (fixed objects removed from visualization)")
        elif apply_static_mask_to_viz:
            print(f"  Static mask filter: DISABLED (no static mask loaded)")
        print(f"  Depth scale: {depth_scale:.6f} m/unit")
    except Exception as e:
        print(f"Failed to initialize visualizer: {e}")
        camera.stop()
        return 1
    
    # Initialize video writer if recording is enabled
    video_writer = None
    if record_visualization:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = save_dir / f"visualization_{timestamp}.mp4"
        print(f"\n📹 Video recording ENABLED")
        print(f"   Output: {video_path}")
        print(f"   FPS: {video_fps}")
        # VideoWriter will be initialized on first frame when we know the resolution
    
    print("\n" + "=" * 60)
    print("System ready! Press Q to quit, SPACE to pause, R to reset")
    print("=" * 60 + "\n")
    
    # Main processing loop
    frame_count = 0
    save_count = 0
    
    try:
        while not visualizer.should_close():
            # Start total timing
            t_loop_start = time.perf_counter()
            
            # Get frame from camera
            t_camera_start = time.perf_counter()
            try:
                rgb, depth = camera.get_frame(timeout=1.0)
            except TimeoutError:
                print("Frame timeout, skipping...")
                continue
            t_camera = time.perf_counter() - t_camera_start
            
            # Skip processing if paused
            if visualizer.is_paused():
                if enable_2d:
                    visualizer.update_2d(rgb, None, 0, "PAUSED")
                time.sleep(0.1)
                continue
            
            # Resize frames if processing resolution is specified
            t_resize_start = time.perf_counter()
            if processing_width and processing_height:
                import cv2
                rgb_proc = cv2.resize(rgb, (processing_width, processing_height), interpolation=cv2.INTER_LINEAR)
                depth_proc = cv2.resize(depth, (processing_width, processing_height), interpolation=cv2.INTER_NEAREST)
            else:
                rgb_proc = rgb
                depth_proc = depth
            t_resize = time.perf_counter() - t_resize_start
            
            # Process frame through pipeline
            t_pipeline_start = time.perf_counter()
            try:
                result = pipeline.process_frame(rgb_proc, depth_proc)
            except Exception as e:
                print(f"Processing error: {e}")
                continue
            t_pipeline = time.perf_counter() - t_pipeline_start
            
            # Update visualizations
            fps = visualizer.compute_fps()
            
            # Start visualization timing
            t_viz_start = time.perf_counter()
            
            # Resize mask back to original resolution if we resized for processing
            if processing_width and processing_height:
                import cv2
                mask_viz = cv2.resize(result['mask'].astype(np.uint8), 
                                     (rgb.shape[1], rgb.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                mask_viz = result['mask']
            
            # Apply static mask filter to visualization if enabled
            if apply_static_mask_to_viz and static_mask_for_viz is not None:
                # Resize static mask to match visualization mask if needed
                if static_mask_for_viz.shape != mask_viz.shape:
                    static_mask_resized = cv2.resize(
                        static_mask_for_viz.astype(np.uint8),
                        (mask_viz.shape[1], mask_viz.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                else:
                    static_mask_resized = static_mask_for_viz
                
                # Filter out static objects from visualization
                mask_viz = mask_viz & ~static_mask_resized
            
            # Downscale for display if high-res capture (performance optimization)
            # Visualization operations (mask overlay, cv2.imshow) are VERY slow on high-res images
            # E.g., 1920x1080 mask overlay takes ~43ms vs 6ms for 640x480 (7x slower!)
            DISPLAY_MAX_SIZE = 640  # Max dimension for display
            if rgb.shape[1] > DISPLAY_MAX_SIZE or rgb.shape[0] > DISPLAY_MAX_SIZE:
                # Calculate downscale factor
                scale = min(DISPLAY_MAX_SIZE / rgb.shape[1], DISPLAY_MAX_SIZE / rgb.shape[0])
                display_w = int(rgb.shape[1] * scale)
                display_h = int(rgb.shape[0] * scale)
                
                # Downscale for display
                rgb_display = cv2.resize(rgb, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
                if mask_viz is not None:
                    mask_display = cv2.resize(mask_viz.astype(np.uint8), (display_w, display_h), 
                                            interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    mask_display = None
            else:
                rgb_display = rgb
                mask_display = mask_viz
            
            # Calculate timing BEFORE visualization calls
            t_loop_end = time.perf_counter()
            t_total = t_loop_end - t_loop_start
            pct_camera = (t_camera / t_total) * 100
            pct_resize = (t_resize / t_total) * 100
            pct_pipeline = (t_pipeline / t_total) * 100
            # Viz percentage will be updated after viz calls
            
            # Build status with mesh statistics and timing breakdown
            mesh_verts = result['mesh_vertices']
            if len(mesh_verts) > 0:
                mesh_mean = mesh_verts.mean(axis=0)
                mesh_std = mesh_verts.std(axis=0)
                status = (f"FPS: {1000/t_total/1000:.1f} | Total: {t_total*1000:.1f}ms "
                         f"(Cam: {t_camera*1000:.1f}ms {pct_camera:.0f}%, "
                         f"Resize: {t_resize*1000:.1f}ms {pct_resize:.0f}%, "
                         f"Pipeline: {t_pipeline*1000:.1f}ms {pct_pipeline:.0f}%) | "
                         f"PCL: {len(result['pointcloud'])} | Mesh: {len(mesh_verts)} verts")
            else:
                status = (f"FPS: {1000/t_total/1000:.1f} | Total: {t_total*1000:.1f}ms "
                         f"(Cam: {t_camera*1000:.1f}ms {pct_camera:.0f}%, "
                         f"Resize: {t_resize*1000:.1f}ms {pct_resize:.0f}%, "
                         f"Pipeline: {t_pipeline*1000:.1f}ms {pct_pipeline:.0f}%) | "
                         f"Mesh: 0 verts")
            
            # Add tracking info if in auto-detect mode
            if sam2_auto_detect and 'tracking_info' in result:
                t_info = result['tracking_info']
                if t_info.get('initialized'):
                    status += f" | Track: Q={t_info.get('quality_score', 0):.2f} IoU={t_info.get('iou', 0):.2f}"
                    if t_info.get('lost'):
                        status += " [LOST]"
                else:
                    status += " | Track: WAITING..."
            
            # Update visualizations (respecting enable flags)
            if enable_2d:
                visualizer.update_2d(rgb_display, mask_display, fps, status)
            if enable_3d:
                visualizer.update_3d(result['pointcloud'], result['mesh_vertices'])
            
            # Show original RGB and depth if requested
            visualizer.update_original_rgb(rgb)
            visualizer.update_original_depth(depth)
            visualizer.update_filtered_depth(depth, depth_min, depth_max)
            
            # Record visualization to video if enabled
            if record_visualization:
                # Create visualization frame (RGB with mask overlay)
                viz_frame = visualizer.get_2d_frame(rgb_display, mask_display, status)
                
                if viz_frame is not None:
                    # Initialize video writer on first frame
                    if video_writer is None:
                        import cv2
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        frame_h, frame_w = viz_frame.shape[:2]
                        video_writer = cv2.VideoWriter(str(video_path), fourcc, video_fps, (frame_w, frame_h))
                        if video_writer.isOpened():
                            print(f"   Video writer initialized: {frame_w}x{frame_h}")
                        else:
                            print(f"   ⚠️ Failed to initialize video writer")
                            record_visualization = False
                    
                    # Write frame
                    if video_writer is not None and video_writer.isOpened():
                        video_writer.write(cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR))
            
            # Measure actual visualization time
            t_viz = time.perf_counter() - t_viz_start
            t_total_final = time.perf_counter() - t_loop_start
            
            # Recalculate percentages with actual viz time
            pct_camera = (t_camera / t_total_final) * 100
            pct_resize = (t_resize / t_total_final) * 100
            pct_pipeline = (t_pipeline / t_total_final) * 100
            pct_viz = (t_viz / t_total_final) * 100
            
            frame_count += 1
            
            # Print mesh statistics every 30 frames
            if frame_count % 30 == 0:
                mesh_verts = result['mesh_vertices']
                pcl = result['pointcloud']
                print(f"\n[Frame {frame_count}]")
                print(f"  FPS: {1000/t_total_final/1000:.1f} (matches 1000/{t_total_final*1000:.1f}ms)")
                print(f"  Total: {t_total_final*1000:.1f}ms")
                print(f"    Camera:    {t_camera*1000:.1f}ms ({pct_camera:.1f}%)")
                print(f"    Resize:    {t_resize*1000:.1f}ms ({pct_resize:.1f}%)")
                print(f"    Pipeline:  {t_pipeline*1000:.1f}ms ({pct_pipeline:.1f}%)")
                print(f"    Viz:       {t_viz*1000:.1f}ms ({pct_viz:.1f}%)")
                print(f"  Pointcloud: {len(pcl)} points")
                print(f"  Mesh: {len(mesh_verts)} vertices")
                if len(mesh_verts) > 0:
                    print(f"  Mesh mean: {mesh_verts.mean(axis=0)}")
                    print(f"  Mesh std: {mesh_verts.std(axis=0)}")
                    print(f"  Mesh bounds: min={mesh_verts.min(axis=0)}, max={mesh_verts.max(axis=0)}")
            
            # Auto-save if enabled
            if output_cfg.get('auto_save', False):
                save_frame(save_dir, save_count, result)
                save_count += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print(f"\nProcessed {frame_count} frames")
        
        # Close video writer if recording was enabled
        if video_writer is not None:
            video_writer.release()
            print(f"📹 Video saved: {video_path}")
            print(f"   Frames recorded: {frame_count}")
        
        visualizer.close()
        camera.stop()
        print("Shutdown complete")
    
    return 0


def save_frame(save_dir: str, frame_id: int, result: dict) -> None:
    """Save frame results to disk.
    
    Args:
        save_dir: Output directory
        frame_id: Frame ID
        result: Processing results dict
    """
    frame_dir = os.path.join(save_dir, f"frame_{frame_id:06d}")
    os.makedirs(frame_dir, exist_ok=True)
    
    # Save pointcloud
    pcl_path = os.path.join(frame_dir, "pointcloud.npy")
    np.save(pcl_path, result['pointcloud'])
    
    # Save mesh
    mesh_path = os.path.join(frame_dir, "mesh_vertices.npy")
    np.save(mesh_path, result['mesh_vertices'])
    
    # Save mask
    mask_path = os.path.join(frame_dir, "mask.npy")
    np.save(mask_path, result['mask'])
    
    print(f"Saved frame {frame_id} to {frame_dir}")


if __name__ == "__main__":
    exit(main())
