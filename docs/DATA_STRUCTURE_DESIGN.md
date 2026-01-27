# LeRobot Data Structure Design for MeshGAT Integration

## Overview
This document explains the data flow from camera → robot observation → processor pipeline → dataset for the Denso DeltaPose Force robot with MeshGAT mesh prediction.

## 1. Camera Layer Data Structures

### RealSense Camera Methods

```python
# Method 1: async_read() - Color only
color_frame: NDArray = camera.async_read()  # Shape: (H, W, 3), dtype: uint8

# Method 2: async_read_both() - Color + Depth
frames: dict = camera.async_read_both()
# Returns: {"color": NDArray[H, W, 3], "depth": NDArray[H, W, 3]}
# - color: RGB image (uint8)
# - depth: Colorized depth visualization (uint8) or raw depth (uint16)
```

### Camera Configuration (YAML)
```yaml
cameras:
  camera_l515:
    type: intelrealsense
    serial_number_or_name: f1181599
    width: 640
    height: 480
    fps: 30
    use_depth: true  # ← Enables depth capture
    color_mode: RGB
```

---

## 2. Robot Observation Layer

### Current Issue (BEFORE Fix)
```python
# denso_deltapose_force.py - get_observation()
obs["camera_l515"] = cam.async_read()  # ❌ Only returns color!
# Result: obs = {
#     "curPos_J1_A": 0.0,
#     ...  # 51 state scalars
#     "camera_l515": NDArray[480, 640, 3]  # color only
# }
```

### Proposed Fix (AFTER)
```python
# Check if camera has depth enabled
if hasattr(cam, 'use_depth') and cam.use_depth:
    frames = cam.async_read_both()  # ✅ Get both color and depth
    obs[cam_key] = frames  # Keep nested structure for backward compatibility
    # Flatten for processors:
    obs["rgb"] = frames["color"]     # Direct access for FabricPointCloudProcessor
    obs["depth"] = frames["depth"]   # Direct access for FabricPointCloudProcessor
else:
    obs[cam_key] = cam.async_read()  # Single color frame

# Result: obs = {
#     "curPos_J1_A": 0.0,
#     ...  # 51 state scalars
#     "camera_l515": {"color": NDArray, "depth": NDArray},  # Nested (backward compat)
#     "rgb": NDArray[480, 640, 3],   # Flattened (for processors)
#     "depth": NDArray[480, 640, 3], # Flattened (for processors)
# }
```

---

## 3. Processor Pipeline Data Structures

### Pipeline Flow
```
RobotObservation → EnvTransition → ProcessorSteps → EnvTransition → RobotObservation
```

### EnvTransition Structure
```python
from lerobot.processor.types import EnvTransition

transition = EnvTransition(
    observation={
        # State features (51 scalars)
        "curPos_J1_A": torch.tensor(0.0),
        ...
        # Visual features (processors expect flat keys)
        "rgb": torch.tensor(shape=[480, 640, 3], dtype=torch.uint8),
        "depth": torch.tensor(shape=[480, 640, 3], dtype=torch.uint8),
        # Processed features (added by processors)
        "pcl": torch.tensor(shape=[1024, 3], dtype=torch.float32),  # Added by FabricPointCloud
        "mesh_vertices": torch.tensor(shape=[N, 3], dtype=torch.float32),  # Added by MeshGAT
    },
    action={...},
    reward=None,
    done=False,
    success=False,
)
```

### Processor Steps

#### Step 1: DensoDeltaPoseStripRemoteActionStep
```python
# Removes _last_remote_action from observation
observation.pop("_last_remote_action", None)
```

#### Step 2: FabricPointCloudProcessorStep
```python
# Input: observation["rgb"], observation["depth"]
# Process: SAM2 segmentation → 3D pointcloud generation
# Output: observation["pcl"] = torch.tensor(shape=[1024, 3], dtype=float32)

# Configuration in factory.py:
FabricPointCloudProcessorStep(
    rgb_key="rgb",          # ← Expects flat key!
    depth_key="depth",      # ← Expects flat key!
    fx=600.974,
    fy=600.819,
    cx=331.946,
    cy=248.233,
    depth_scale=0.000250,   # L515: 0.25mm per unit
    target_num_points=1024,
    sam_runner=sam2_runner,
    output_key="pcl",
)
```

#### Step 3: MeshGATObservationProcessorStep
```python
# Input: observation["pcl"]
# Process: MeshGAT graph neural network inference
# Output: observation["mesh_vertices"] = torch.tensor(shape=[N, 3], dtype=float32)

# Configuration:
MeshGATObservationProcessorStep(
    checkpoint_path="/path/to/meshgat.pt",
    config_path="/path/to/config.yaml",
    device="cuda",
    input_key="pcl",
    output_key="mesh_vertices",
)
```

---

## 4. Dataset Storage Structure

### Feature Schema (PolicyFeature)
```python
from lerobot.configs.types import PolicyFeature, FeatureType

# State features (51 scalars)
PolicyFeature(key="curPos_J1_A", type=FeatureType.STATE, shape=(1,))
...

# Visual features
PolicyFeature(key="rgb", type=FeatureType.VISUAL, shape=(480, 640, 3))
PolicyFeature(key="depth", type=FeatureType.VISUAL, shape=(480, 640, 3))

# NEW: 3D geometric features
PolicyFeature(key="pcl", type=FeatureType.PCL, shape=(1024, 3))
PolicyFeature(key="mesh_vertices", type=FeatureType.MESH, shape=(None, 3))  # Variable N
```

### LeRobotDataset Metadata
```python
# dataset.meta structure
{
    "robot_type": "denso_deltapose_force",
    "fps": 30,
    "codebase_version": "v3.0.0",
    "features": {
        # Observation features
        "observation.state.curPos_J1_A": PolicyFeature(...),
        ...
        "observation.images.rgb": PolicyFeature(...),  # Video encoded
        "observation.images.depth": PolicyFeature(...),  # Video encoded
        "observation.state.pcl": PolicyFeature(type=FeatureType.PCL, shape=(1024, 3)),
        "observation.state.mesh_vertices": PolicyFeature(type=FeatureType.MESH, shape=(None, 3)),
        
        # Action features (68 dimensions)
        "action.deltapose_l_x": PolicyFeature(...),
        ...
    },
    "stats": {
        "observation.state.pcl": {
            "mean": torch.tensor([0.0, 0.0, 0.5], shape=[1, 3]),
            "std": torch.tensor([0.1, 0.1, 0.1], shape=[1, 3]),
            "min": torch.tensor([-0.5, -0.5, 0.0]),
            "max": torch.tensor([0.5, 0.5, 1.0]),
        },
        "observation.state.mesh_vertices": {
            # Stats for mesh vertices
        },
    },
    "episodes": [
        {"episode_id": 0, "length": 1500, "timestamp": "2026-01-09T16:00:00"},
        ...
    ],
}
```

### On-Disk Storage
```
data/lerobot/${HF_USER}/denso_fabric_manipulation/
├── meta/
│   ├── info.json              # Dataset info (fps, robot_type, codebase_version)
│   ├── tasks.jsonl            # Task descriptions per episode
│   ├── episodes.jsonl         # Episode metadata (length, timestamp)
│   └── stats.safetensors      # Normalization statistics
├── data/
│   ├── chunk-000/
│   │   ├── observation.state.curPos_J1_A.parquet
│   │   ├── ...
│   │   ├── observation.state.pcl.parquet         # Pointcloud data
│   │   ├── observation.state.mesh_vertices.parquet  # Mesh vertices
│   │   ├── action.deltapose_l_x.parquet
│   │   └── ...
│   └── chunk-001/
│       └── ...
└── videos/
    ├── chunk-000/
    │   ├── observation.images.rgb.mp4      # H.264 encoded
    │   └── observation.images.depth.mp4    # H.264 encoded
    └── chunk-001/
        └── ...
```

---

## 5. Key Design Decisions

### Why Flatten Camera Structure?
**Problem:** Processors expect `observation["rgb"]` but camera returns `observation["camera_l515"]["color"]`

**Solution:** Robot's `get_observation()` flattens structure:
```python
obs["camera_l515"] = {"color": ..., "depth": ...}  # Keep for backward compatibility
obs["rgb"] = obs["camera_l515"]["color"]           # Add flat access
obs["depth"] = obs["camera_l515"]["depth"]         # Add flat access
```

**Benefits:**
- ✅ Processors work with simple flat keys
- ✅ Backward compatible with existing code expecting `obs["camera_key"]`
- ✅ Clear separation: `camera_l515` for raw data, `rgb/depth` for processing

### Why Add PCL and MESH Feature Types?
**Problem:** Pointcloud and mesh data don't fit existing feature types (STATE, VISUAL, ACTION)

**Solution:** Add dedicated feature types in `lerobot/configs/types.py`:
```python
class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ACTION = "ACTION"
    PCL = "PCL"      # ← Point cloud (N, 3) float32
    MESH = "MESH"    # ← Mesh vertices (N, 3) float32
```

**Benefits:**
- ✅ Explicit type identification in dataset schema
- ✅ Enables specialized normalization/preprocessing for 3D data
- ✅ Clear distinction from 2D images (VISUAL) and 1D scalars (STATE)

### Variable-Length Mesh Vertices
**Challenge:** Mesh output has variable number of vertices (N changes per frame)

**Solution:** Use `shape=(None, 3)` in PolicyFeature:
```python
PolicyFeature(
    key="mesh_vertices",
    type=FeatureType.MESH,
    shape=(None, 3),  # ← None indicates variable dimension
)
```

**Storage:** Parquet format handles variable-length arrays natively.

---

## 6. Data Flow Summary

```
┌─────────────────┐
│ RealSense L515  │
│  640x480 @ 30Hz │
└────────┬────────┘
         │ async_read_both()
         ↓
┌─────────────────────────────────────────┐
│ Robot Observation                       │
│ {                                       │
│   "curPos_J1_A": 0.0, ...  (51 state)  │
│   "camera_l515": {"color", "depth"},   │
│   "rgb": NDArray[480,640,3],           │  ← Flattened
│   "depth": NDArray[480,640,3]          │  ← Flattened
│ }                                       │
└────────┬────────────────────────────────┘
         │ to_transition()
         ↓
┌─────────────────────────────────────────┐
│ EnvTransition                           │
│ observation: {                          │
│   "curPos_J1_A": Tensor, ...           │
│   "rgb": Tensor[480,640,3],            │
│   "depth": Tensor[480,640,3]           │
│ }                                       │
└────────┬────────────────────────────────┘
         │ FabricPointCloudProcessorStep
         ↓
┌─────────────────────────────────────────┐
│ EnvTransition (after pointcloud)        │
│ observation: {                          │
│   ...,                                  │
│   "pcl": Tensor[1024, 3]  ← Added      │
│ }                                       │
└────────┬────────────────────────────────┘
         │ MeshGATObservationProcessorStep
         ↓
┌─────────────────────────────────────────┐
│ EnvTransition (after mesh)              │
│ observation: {                          │
│   ...,                                  │
│   "pcl": Tensor[1024, 3],              │
│   "mesh_vertices": Tensor[N, 3]  ← Added│
│ }                                       │
└────────┬────────────────────────────────┘
         │ to_observation()
         ↓
┌─────────────────────────────────────────┐
│ Robot Observation (processed)           │
│ Ready for dataset.add_frame()           │
└────────┬────────────────────────────────┘
         │ dataset.add_frame(frame)
         ↓
┌─────────────────────────────────────────┐
│ LeRobotDataset Storage                  │
│ ├── observation.state.*.parquet         │
│ ├── observation.state.pcl.parquet       │
│ ├── observation.state.mesh_vertices.parquet│
│ ├── videos/observation.images.rgb.mp4   │
│ └── videos/observation.images.depth.mp4 │
└─────────────────────────────────────────┘
```

---

## 7. Implementation Checklist

- [ ] **Fix robot.get_observation()** to use `async_read_both()` when `use_depth=True`
- [ ] **Flatten camera structure** to expose `rgb` and `depth` at top level
- [ ] **Update camera configuration** to set intrinsics and depth_scale after connection
- [ ] **Test processor pipeline** with real camera data
- [ ] **Verify dataset schema** includes PCL and MESH features
- [ ] **Check normalization stats** are computed for 3D features
- [ ] **Validate parquet storage** handles variable-length mesh vertices

---

## 8. Common Debugging Commands

```bash
# Test camera directly
python -m lerobot.cameras.realsense.debug_realsense_stream --serial f1181599 --duration 5

# Check dataset schema
python -c "from lerobot.datasets import LeRobotDataset; ds = LeRobotDataset('data/lerobot/${HF_USER}/denso_fabric_manipulation'); print(ds.meta.features)"

# Inspect single frame
python -c "from lerobot.datasets import LeRobotDataset; ds = LeRobotDataset('...'); frame = ds[0]; print({k: v.shape for k, v in frame.items()})"

# Visualize dataset
lerobot-dataset-viz --repo-id ${HF_USER}/denso_fabric_manipulation --episode-index 0
```

---

## 9. Next Steps

1. Apply the `get_observation()` fix to `denso_deltapose_force.py`
2. Run recording to test complete pipeline
3. Inspect first recorded episode to verify data structure
4. Add unit tests for camera flattening logic
5. Document camera configuration best practices
