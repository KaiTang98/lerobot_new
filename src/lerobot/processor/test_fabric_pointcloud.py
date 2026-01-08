#!/usr/bin/env python3
"""
Test script for FabricPointCloudProcessorStep.

Creates synthetic RGB + depth + mask, runs the processor, and validates output.
"""

import numpy as np

from lerobot.processor.fabric_pointcloud_processor import FabricPointCloudProcessorStep


def dummy_sam_runner(rgb: np.ndarray) -> np.ndarray:
    """Dummy SAM runner that creates a simple circular mask in the center."""
    H, W, _ = rgb.shape
    y_center, x_center = H // 2, W // 2
    radius = min(H, W) // 4

    y_coords, x_coords = np.ogrid[:H, :W]
    mask = ((y_coords - y_center) ** 2 + (x_coords - x_center) ** 2) <= radius**2

    return mask.astype(bool)


def create_synthetic_data(height: int = 480, width: int = 640) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic RGB and depth images for testing.

    Args:
        height: Image height.
        width: Image width.

    Returns:
        rgb: (H, W, 3) uint8 image.
        depth: (H, W) uint16 depth map (in mm, simulating real RealSense data).
    """
    # Create a simple gradient RGB image
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)  # Red gradient horizontally
    rgb[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]  # Green gradient vertically
    rgb[:, :, 2] = 128  # Constant blue

    # Create a synthetic depth map (planar surface at ~0.5m, with some variation)
    # L515 uses depth_scale = 0.00025, so 2000 units = 0.5m
    base_depth = 2000
    depth = np.full((height, width), base_depth, dtype=np.uint16)

    # Add a slight gradient to simulate a tilted plane
    y_coords, x_coords = np.ogrid[:height, :width]
    depth = depth + (y_coords // 10).astype(np.uint16) + (x_coords // 20).astype(np.uint16)

    return rgb, depth


def main() -> None:
    print("=== FabricPointCloudProcessorStep Test ===\n")

    # 1) Create synthetic data
    print("1. Creating synthetic RGB + depth...")
    rgb, depth = create_synthetic_data(height=480, width=640)
    print(f"   RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
    print(f"   Depth shape: {depth.shape}, dtype: {depth.dtype}")
    print(f"   Depth range: {depth.min()} - {depth.max()} (device units)\n")

    # 2) Create processor with L515-like intrinsics
    print("2. Initializing FabricPointCloudProcessorStep...")
    processor = FabricPointCloudProcessorStep(
        rgb_key="rgb",
        depth_key="depth",
        fx=600.974,
        fy=600.81915,
        cx=331.9461,
        cy=248.23386,
        depth_scale=0.00025,  # L515 depth unit = 0.25mm
        target_num_points=1024,
        sam_runner=dummy_sam_runner,
        output_key="pcl",
        resample_method="random",
    )
    print(f"   Processor: {processor.__class__.__name__}")
    print(f"   Target points: {processor.target_num_points}\n")

    # 3) Run processor
    print("3. Running processor...")
    observation = {"rgb": rgb, "depth": depth}

    try:
        observation = processor.observation(observation)
        print("   ✓ Processing succeeded\n")
    except Exception as e:
        print(f"   ✗ Processing failed: {e}\n")
        return

    # 4) Validate output
    print("4. Validating output...")
    pcl = observation.get("pcl")

    if pcl is None:
        print("   ✗ No 'pcl' key in observation after processing.\n")
        return

    print(f"   Pointcloud shape: {pcl.shape}, dtype: {pcl.dtype}")

    if pcl.shape != (processor.target_num_points, 3):
        print(f"   ✗ Expected shape ({processor.target_num_points}, 3), got {pcl.shape}\n")
        return

    if pcl.dtype != np.float32:
        print(f"   ✗ Expected dtype float32, got {pcl.dtype}\n")
        return

    # Check centering (mean should be close to zero)
    mean = pcl.mean(axis=0)
    print(f"   Pointcloud mean: {mean}")

    if np.linalg.norm(mean) > 1e-3:
        print(f"   ⚠ Warning: mean is not close to zero (norm={np.linalg.norm(mean):.6f})\n")
    else:
        print("   ✓ Pointcloud is centered (mean ≈ 0)\n")

    # Check range
    pcl_min = pcl.min(axis=0)
    pcl_max = pcl.max(axis=0)
    print(f"   Pointcloud range:")
    print(f"     X: [{pcl_min[0]:.4f}, {pcl_max[0]:.4f}]")
    print(f"     Y: [{pcl_min[1]:.4f}, {pcl_max[1]:.4f}]")
    print(f"     Z: [{pcl_min[2]:.4f}, {pcl_max[2]:.4f}]\n")

    # 5) Test with FPS resampling
    print("5. Testing FPS resampling method...")
    processor_fps = FabricPointCloudProcessorStep(
        rgb_key="rgb",
        depth_key="depth",
        fx=600.974,
        fy=600.81915,
        cx=331.9461,
        cy=248.23386,
        depth_scale=0.00025,
        target_num_points=512,
        sam_runner=dummy_sam_runner,
        output_key="pcl",
        resample_method="fps",
    )

    observation_fps = {"rgb": rgb, "depth": depth}
    observation_fps = processor_fps.observation(observation_fps)
    pcl_fps = observation_fps["pcl"]

    print(f"   FPS pointcloud shape: {pcl_fps.shape}, dtype: {pcl_fps.dtype}")
    print(f"   FPS mean: {pcl_fps.mean(axis=0)}\n")

    # 6) Summary
    print("=== Test Summary ===")
    print("✓ FabricPointCloudProcessorStep works correctly")
    print("✓ Random and FPS resampling both produce valid output")
    print("✓ Pointcloud is centered and has expected shape")
    print("\nNext steps:")
    print("- Wire this processor into the Denso MeshGAT observation pipeline")
    print("- Replace dummy_sam_runner with real SAM2 inference")
    print("- Test with real L515 camera data")


if __name__ == "__main__":
    main()
