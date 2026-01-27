# MeshGAT Feature Naming Convention for LeRobot

## Your Proposed Structure
```python
observation.images.camera_l515.rgb      # ❌ Non-standard nesting
observation.images.camera_l515.depth    # ❌ Non-standard nesting
observation.images.mesh                 # ❓ Mesh is not an image
observation.pcl                         # ❌ Missing observation prefix
observation.mesh.fabric                 # ❓ Needs clarification
```

## LeRobot v3 Conventions (from codebase analysis)

### **Standard Feature Keys**
```python
# Defined in src/lerobot/utils/constants.py
OBS_STR = "observation"
OBS_STATE = "observation.state"          # Scalar/vector state data
OBS_IMAGES = "observation.images"        # Visual data (video-encoded)
OBS_ENV_STATE = "observation.environment_state"
ACTION = "action"
```

### **Naming Patterns from Code**
```python
# Images: observation.images.<camera_name>
"observation.images.laptop"              # ✅ Standard
"observation.images.phone"               # ✅ Standard
"observation.images.camera_l515"         # ✅ Your RGB camera

# State: observation.state or observation.state.<specific>
"observation.state"                      # ✅ Flat vector [N,]
"observation.state.ee.x"                 # ✅ Named scalar
"observation.state.j1.pos"               # ✅ Named scalar
```

### **Dataset Storage Structure**
```
data/lerobot/repo_id/
├── data/
│   └── chunk-000/
│       ├── observation.state.parquet          # Scalars/vectors
│       ├── observation.state.ee.x.parquet     # Individual states
│       └── action.parquet
└── videos/
    └── observation.images.laptop/
        └── chunk-000/
            └── file-000.mp4                   # H.264 encoded
```

---

## **Recommended Design for MeshGAT**

### **Option 1: Flat Structure (RECOMMENDED)**

```python
# Images (video-encoded, stored in videos/)
observation.images.camera_l515           # RGB image (480, 640, 3) uint8
observation.images.camera_l515_depth     # Depth visualization (480, 640, 3) uint8

# 3D Geometry (parquet-encoded, stored in data/)
observation.state.pcl                    # Point cloud (1024, 3) float32
observation.state.mesh_vertices          # Mesh vertices (N, 3) float32
observation.state.mesh_faces             # Mesh faces (M, 3) int32 [optional]
```

**Why?**
- ✅ Follows LeRobot conventions exactly
- ✅ Clear separation: images → videos/, state → parquet
- ✅ No nesting issues with processors
- ✅ Existing code expects flat keys like `observation.state.*`

### **Option 2: Nested Structure (Alternative)**

```python
# Images
observation.images.camera_l515           # RGB
observation.images.camera_l515_depth     # Depth

# 3D Geometry (grouped by modality)
observation.pcl.fabric                   # Point cloud (1024, 3)
observation.mesh.fabric_vertices         # Mesh vertices (N, 3)
observation.mesh.fabric_faces            # Mesh faces (M, 3)
```

**Why?**
- ✅ Semantic grouping (pcl, mesh as separate modalities)
- ⚠️  Requires adding new top-level prefixes (`observation.pcl`, `observation.mesh`)
- ⚠️  More changes to existing code
- ✅ Better for multiple objects (e.g., `observation.mesh.fabric`, `observation.mesh.garment`)

---

## **Detailed Comparison**

| Aspect | Option 1: Flat | Option 2: Nested |
|--------|----------------|------------------|
| **Follows existing conventions** | ✅ Exact match | ⚠️ Extends conventions |
| **Code changes required** | Minimal | Moderate (add new prefixes) |
| **Multi-object support** | `observation.state.pcl_fabric`<br>`observation.state.pcl_garment` | ✅ `observation.pcl.fabric`<br>`observation.pcl.garment` |
| **Processor compatibility** | ✅ Works with existing | Needs new prefix handling |
| **Dataset storage** | `data/chunk-000/observation.state.pcl.parquet` | `data/chunk-000/observation.pcl.fabric.parquet` |
| **Type system** | Uses `FeatureType.PCL` & `MESH` | Same |

---

## **Implementation Details**

### **Option 1: Flat Structure (Recommended)**

#### Robot get_observation()
```python
def get_observation(self) -> dict[str, Any]:
    obs = dict(self._last_remote_state)  # State scalars
    
    # Camera frames (flattened for processors)
    for cam_key, cam in self.cameras.items():
        if hasattr(cam, 'use_depth') and cam.use_depth:
            frames = cam.async_read_both()
            obs[f"camera_{cam_key}"] = frames["color"]        # RGB
            obs[f"camera_{cam_key}_depth"] = frames["depth"]  # Depth
        else:
            obs[cam_key] = cam.async_read()
    
    return obs
```

#### Processor Naming
```python
# FabricPointCloudProcessorStep
FabricPointCloudProcessorStep(
    rgb_key="camera_l515",          # Input: observation["camera_l515"]
    depth_key="camera_l515_depth",  # Input: observation["camera_l515_depth"]
    output_key="pcl",               # Output: observation["pcl"]
    ...
)

# After processing, dataset features will be:
# observation.images.camera_l515       (from robot observation)
# observation.images.camera_l515_depth (from robot observation)
# observation.state.pcl                (from processor)
```

#### MeshGATProcessorStep
```python
MeshGATObservationProcessorStep(
    input_key="pcl",              # Input: observation["pcl"]
    output_key="mesh_vertices",   # Output: observation["mesh_vertices"]
    ...
)

# Dataset feature: observation.state.mesh_vertices
```

#### Dataset Features Schema
```python
features = {
    # Images (video-encoded)
    "observation.images.camera_l515": PolicyFeature(
        type=FeatureType.VISUAL, 
        shape=(480, 640, 3),
        dtype="video",
    ),
    "observation.images.camera_l515_depth": PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(480, 640, 3),
        dtype="video",
    ),
    
    # 3D Geometry (parquet-encoded)
    "observation.state.pcl": PolicyFeature(
        type=FeatureType.PCL,
        shape=(1024, 3),
        dtype="float32",
    ),
    "observation.state.mesh_vertices": PolicyFeature(
        type=FeatureType.MESH,
        shape=(None, 3),  # Variable N
        dtype="float32",
    ),
}
```

---

### **Option 2: Nested Structure (If Multi-Object)**

#### Robot Observation
```python
def get_observation(self) -> dict[str, Any]:
    obs = dict(self._last_remote_state)
    
    # Still flatten cameras for processor compatibility
    for cam_key, cam in self.cameras.items():
        if hasattr(cam, 'use_depth') and cam.use_depth:
            frames = cam.async_read_both()
            obs[f"{cam_key}"] = frames["color"]
            obs[f"{cam_key}_depth"] = frames["depth"]
        else:
            obs[cam_key] = cam.async_read()
    
    return obs
```

#### Processor Outputs
```python
# FabricPointCloudProcessorStep
output_key="pcl_fabric"  # Output: observation["pcl_fabric"]

# MeshGATProcessorStep  
output_key="mesh_fabric"  # Output: observation["mesh_fabric"]
```

#### Dataset Features (Requires New Constants)
```python
# Add to src/lerobot/utils/constants.py
OBS_PCL = "observation.pcl"
OBS_MESH = "observation.mesh"

# Dataset features
features = {
    "observation.images.camera_l515": ...,
    "observation.images.camera_l515_depth": ...,
    
    # New top-level prefixes
    "observation.pcl.fabric": PolicyFeature(
        type=FeatureType.PCL,
        shape=(1024, 3),
    ),
    "observation.mesh.fabric": PolicyFeature(
        type=FeatureType.MESH,
        shape=(None, 3),
    ),
}
```

#### Code Changes Required
```python
# src/lerobot/datasets/utils.py - hw_to_dataset_features()
def hw_to_dataset_features(...):
    # Add handling for PCL and MESH prefixes
    pcl_fts = {k: v for k, v in hw_features.items() if k.startswith("pcl_")}
    mesh_fts = {k: v for k, v in hw_features.items() if k.startswith("mesh_")}
    
    for key, shape in pcl_fts.items():
        features[f"{prefix}.pcl.{key[4:]}"] = {  # Strip "pcl_" prefix
            "dtype": "float32",
            "shape": shape,
        }
    
    for key, shape in mesh_fts.items():
        features[f"{prefix}.mesh.{key[5:]}"] = {  # Strip "mesh_" prefix
            "dtype": "float32",
            "shape": shape,
        }
```

---

## **Final Recommendation**

### **Use Option 1: Flat Structure**

```python
# ✅ RECOMMENDED NAMING:
observation.images.camera_l515           # RGB (video)
observation.images.camera_l515_depth     # Depth (video)
observation.state.pcl                    # Point cloud (parquet)
observation.state.mesh_vertices          # Mesh vertices (parquet)
```

**Reasons:**
1. ✅ **Zero changes** to existing LeRobot conventions
2. ✅ **Works immediately** with existing processors and dataset code
3. ✅ **Clear storage separation**: images → MP4, state → parquet
4. ✅ **Type system ready**: `FeatureType.PCL` and `MESH` already added
5. ✅ **Simple processor pipeline**: flat keys, no nesting issues

**When to use Option 2:**
- You have **multiple objects** to track (e.g., fabric, garment, towel)
- You want **semantic grouping** in feature names
- You're willing to **extend LeRobot's constants** and update dataset utilities

---

## **Implementation Checklist for Option 1**

- [x] ✅ Add `FeatureType.PCL` and `MESH` to enum
- [ ] Update `robot.get_observation()` to flatten camera structure:
  ```python
  obs["camera_l515"] = frames["color"]
  obs["camera_l515_depth"] = frames["depth"]
  ```
- [ ] Configure `FabricPointCloudProcessorStep`:
  ```python
  rgb_key="camera_l515"
  depth_key="camera_l515_depth"
  output_key="pcl"
  ```
- [ ] Configure `MeshGATObservationProcessorStep`:
  ```python
  input_key="pcl"
  output_key="mesh_vertices"
  ```
- [ ] Verify dataset schema includes:
  - `observation.images.camera_l515` (video)
  - `observation.images.camera_l515_depth` (video)
  - `observation.state.pcl` (parquet, type=PCL)
  - `observation.state.mesh_vertices` (parquet, type=MESH)

---

## **Example: How Features Flow**

```
┌──────────────────────────────────────────────────────┐
│ Robot Observation (raw dict from robot.get_obs())   │
│ {                                                    │
│   "curPos_J1_A": 0.0, ...          (51 state)      │
│   "camera_l515": NDArray[480,640,3],      # RGB     │
│   "camera_l515_depth": NDArray[480,640,3], # Depth  │
│ }                                                    │
└────────────────┬─────────────────────────────────────┘
                 │ hw_to_dataset_features()
                 ↓
┌──────────────────────────────────────────────────────┐
│ Initial Dataset Features                             │
│ {                                                    │
│   "observation.state": (51,),        # State vector │
│   "observation.images.camera_l515": (480,640,3),    │
│   "observation.images.camera_l515_depth": (480,640,3)│
│ }                                                    │
└────────────────┬─────────────────────────────────────┘
                 │ FabricPointCloudProcessorStep
                 ↓
┌──────────────────────────────────────────────────────┐
│ After Pointcloud Processing                          │
│ {                                                    │
│   ... (all above features)                           │
│   "observation.state.pcl": (1024, 3),  ← Added     │
│ }                                                    │
└────────────────┬─────────────────────────────────────┘
                 │ MeshGATObservationProcessorStep
                 ↓
┌──────────────────────────────────────────────────────┐
│ Final Dataset Features                               │
│ {                                                    │
│   "observation.state": (51,),                        │
│   "observation.images.camera_l515": (480,640,3),    │
│   "observation.images.camera_l515_depth": (480,640,3)│
│   "observation.state.pcl": (1024, 3),               │
│   "observation.state.mesh_vertices": (N, 3),  ← Added│
│ }                                                    │
└────────────────┬─────────────────────────────────────┘
                 │ LeRobotDataset.add_frame()
                 ↓
┌──────────────────────────────────────────────────────┐
│ On-Disk Storage                                      │
│ data/lerobot/repo_id/                                │
│ ├── data/chunk-000/                                  │
│ │   ├── observation.state.parquet          (51,)    │
│ │   ├── observation.state.pcl.parquet      (1024,3) │
│ │   └── observation.state.mesh_vertices.parquet (N,3)│
│ └── videos/                                          │
│     ├── observation.images.camera_l515/              │
│     │   └── chunk-000/file-000.mp4                   │
│     └── observation.images.camera_l515_depth/        │
│         └── chunk-000/file-000.mp4                   │
└──────────────────────────────────────────────────────┘
```

---

## **Summary**

**Your Question:**
> I want to make it `observation.images.cam_key.rgb` or `.depth` or other image such as mesh.
> We will also have `observation.pcl` or `observation.mesh.fabric`.
> Is this a good way?

**Answer:**
- ❌ `observation.images.cam_key.rgb` - **Too nested**, processors expect flat keys
- ❌ `observation.pcl` - **Missing `observation.state` prefix** (LeRobot convention)
- ✅ `observation.images.camera_l515` - **Perfect** (RGB image)
- ✅ `observation.images.camera_l515_depth` - **Perfect** (depth image)
- ✅ `observation.state.pcl` - **Perfect** (pointcloud data)
- ✅ `observation.state.mesh_vertices` - **Perfect** (mesh vertices)

**Final Structure (Recommended):**
```python
observation.images.camera_l515           # RGB video
observation.images.camera_l515_depth     # Depth video
observation.state.pcl                    # Pointcloud parquet
observation.state.mesh_vertices          # Mesh parquet
```

This follows LeRobot conventions exactly, requires minimal code changes, and works seamlessly with the existing dataset infrastructure! 🎯
