# MeshGAT Real-time Teleop TODO

Step-by-step plan to integrate a real-time MeshGAT fabric perception pipeline into teleoperation and recording.

## 1. RealSense + robot observation plumbing

- [done] 1.1 Add L515 support to the Denso follower robot implementation
  - Initialize a RealSense `pipeline`, configure color + depth streams, and start streaming.
  - Align depth to color (e.g. via `rs.align(rs.stream.color)`) so RGB and depth share the same pixel coordinates.
- [done] 1.2 Expose camera data in `robot.get_observation()`
  - Return `"rgb"` (H x W x 3 uint8) and `"depth"` (H x W uint16/float) in the `RobotObservation` dict.
  - Keep other robot state fields unchanged.
- [done] 1.3 Capture and store camera intrinsics + depth scale
  - Read `(fx, fy, cx, cy)` and `depth_scale` from RealSense and either:
    - store them as attributes on the robot object (preferred), and
    - ensure they can be passed into processors via config.
- [done] 1.4 Smoke test: simple script to call `robot.get_observation()`
  - Verify that RGB and depth shapes are correct and frames are being updated at the desired FPS.

## 2. FabricPointCloudProcessorStep (RGB+depth → pcl)

- [done] 2.1 Implement `FabricPointCloudProcessorStep` in `lerobot.processor`
  - Subclass `ObservationProcessorStep` and register via `ProcessorStepRegistry`.
  - Constructor config:
    - `rgb_key`, `depth_key` (e.g. `"rgb"`, `"depth"`).
    - camera intrinsics `(fx, fy, cx, cy)` and `depth_scale`.
    - `target_num_points` (from MeshGAT config).
    - `sam_runner` callable: `rgb_np -> mask_np`.
  - `observation()`:
    - Run `sam_runner(rgb_np)` to get a binary fabric mask.
    - Project masked depth to camera-frame XYZ using intrinsics.
    - Center points and resample to `target_num_points`.
    - Write result to `observation["pcl"]` as `(N, 3)` float32.
- [done] 2.2 Add `FabricPointCloudProcessorStep` to `lerobot/processor/__init__.py`
  - Export in `__all__` so it can be imported from `lerobot.processor`.
- [done] 2.3 Unit test: synthetic depth + mask
  - Create a small test that constructs a fake `rgb`, `depth`, and a dummy `sam_runner` returning a known mask.
  - Check that `observation["pcl"]` has shape `(N, 3)`, is centered (mean ~ 0), and respects the masked region.

## 3. SAM2 Integration (for fabric segmentation)

- [done] 3.1 Add SAM2 as git submodule
  - `git submodule add https://github.com/facebookresearch/segment-anything-2.git external/sam2/sam2`
  - Install: `cd external/sam2/sam2 && pip install -e .`
- [done] 3.2 Download SAM2 checkpoint
  - Create `external/sam2/checkpoints/` directory
  - Download model: `sam2_hiera_large.pt` or `sam2_hiera_small.pt` (for speed)
- [done] 3.3 Create SAM2 API wrapper (done: `external/sam2/api.py`)
  - Functions: `load_sam2_predictor()`, `create_sam2_runner()`, `create_simple_sam2_runner()`
- [done] 3.4 Test SAM2 integration
  - Create test script with synthetic RGB image
  - Verify sam_runner returns correct mask shape
- [done] 3.5 Determine fabric bounding box prompt
  - Manually identify fabric region in your camera view
  - Record box_prompt: `[x1, y1, x2, y2]` coordinates

## 4. Wire FabricPointCloud + MeshGAT into Denso observation pipeline

- [done] 4.1 Create `make_denso_meshgat_robot_observation_processor` factory
  - Location: `src/lerobot/processor/factory.py`
  - Steps:
    - `DensoDeltaPoseStripRemoteActionStep()` - cleanup
    - `FabricPointCloudProcessorStep(sam_runner=...)` - RGB+depth → pcl
    - `MeshGATObservationProcessorStep(...)` - pcl → mesh_vertices
  - Pass camera intrinsics, depth scale, SAM2 config, MeshGAT config
- [ ] 4.2 Config plumbing for all parameters
  - Camera intrinsics + depth_scale: from `robot.camera.config` (already available)
  - SAM2: checkpoint path + box_prompt (add to robot config)
  - MeshGAT: checkpoint + config paths (add to robot config).
    - MeshGAT checkpoint, config, and template paths.
  - Make sure `make_teleop_robot_processors` (or a variant) can construct the pipeline with these values.
- [ ] 3.3 Integration test: offline script
  - Write a small script that:
    - Instantiates the Denso robot with L515,
    - Builds the MeshGAT observation pipeline,
    - Runs a few iterations of `obs -> obs_processed` without teleop,
    - Asserts that `"pcl"` and `"mesh_vertices"` appear in `obs_processed` with expected shapes.

## 4. Teleop and recording integration

- [ ] 4.1 Specialize `make_teleop_robot_processors` for Denso+MeshGAT
  - For the desired `robot.type` / `teleop.type` combo, return a tuple where
    `robot_observation_processor` is the MeshGAT-enabled one.
- [ ] 4.2 Update dataset feature spec to include `mesh_vertices`
  - Extend the dataset feature aggregation so `mesh_vertices` is registered as an observation state feature.
  - Verify that `obs_state["mesh_vertices"]` is present when recording with `lerobot-record`.
- [ ] 4.3 Teleop smoke test
  - Use `lerobot-teleoperate` / `lerobot-record` with the new config.
  - Confirm that control loop FPS remains acceptable and no exceptions are raised.

## 5. GPU and performance optimizations

- [ ] 5.1 Move heavy ops to GPU
  - Ensure SAM2 runs on GPU.
  - Optionally implement depth→pointcloud and resampling in torch instead of numpy.
  - Keep MeshGAT on the same device and minimize CPU↔GPU transfers.
- [ ] 5.2 Resolution and frequency tuning
  - Add options to run SAM2 at downsampled resolution and/or lower frequency than the control loop.
  - Benchmark end-to-end latency and adjust camera resolution, SAM2 settings, and `target_num_points`.
- [ ] 5.3 Profiling and logging
  - Add simple timing logs around:
    - camera capture,
    - SAM2 inference,
    - FabricPointCloudProcessorStep,
    - MeshGAT forward.
  - Use this to guide further optimizations if needed.

## 6. Optional: offline / dataset preprocessing

- [ ] 6.1 CLI or script to precompute MeshGAT vertices for a dataset
  - Reuse FabricPointCloud and MeshGAT processors in an offline loop.
  - Store `mesh_vertices` into an existing dataset to avoid real-time computation during training.
- [ ] 6.2 Tests for offline pipeline
  - Minimal test to run the offline script on a tiny dataset shard and check that new features are written correctly.
