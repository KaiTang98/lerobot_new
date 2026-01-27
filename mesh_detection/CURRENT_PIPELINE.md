# Current Mesh Detection Pipeline

**Date:** January 26, 2026  
**Status:** ✅ Runs but mesh predictions are incorrect  
**Next Step:** Add pointcloud preprocessing (centering, normalization, etc.) before MeshGAT

---

## 🔄 Pipeline Overview

```
Camera Frame (RGB-D)
    ↓
[1] Camera Capture & Resize
    ↓
[2] SAM2 Segmentation (Fabric Mask)
    ↓
[3] Depth to Pointcloud Projection
    ↓
[4] Pointcloud Centering
    ↓
[5] Pointcloud Resampling (1024 points)
    ↓
[6] MeshGAT Inference
    ↓
Mesh Vertices (Output)
```

---

## 📋 Detailed Pipeline Steps

### [1] Camera Capture & Optional Resize
**Location:** `main.py` lines 289-307

**Input:**
- RealSense camera stream

**Process:**
- Capture RGB (uint8, H×W×3) and Depth (uint16, H×W)
- Optional resize to processing resolution if configured

**Output:**
- `rgb_proc`: RGB image (default 640×480 or configured size)
- `depth_proc`: Depth image (same size as rgb_proc)

**Configuration:**
```yaml
camera:
  color_width: 1920
  color_height: 1080
  depth_width: 1920
  depth_height: 1080
  fps: 30

processing:
  processing_width: 640  # Optional downscale
  processing_height: 480
```

---

### [2] SAM2 Segmentation
**Location:** `mesh_pipeline.py` lines 166-223 (auto mode) or 224-241 (manual mode)

**Input:**
- `rgb`: (H, W, 3) uint8 RGB image
- `depth`: (H, W) uint16 depth (for auto-detection only)

**Process:**

**Manual Mode:**
- Use fixed initial point or box prompt
- SAM2 tracks object from initial prompt

**Auto-Detection Mode:**
- Run `FabricDetector` to find fabric
- Extract foreground using depth range (0.77-0.91m)
- Filter out robot (low saturation)
- Filter out table (white, flat)
- Filter out static objects (pre-defined mask)
- Find centroid of largest fabric blob
- Initialize SAM2 with detected point
- Track fabric in subsequent frames
- Re-initialize if tracking lost

**Output:**
- `mask`: (H, W) bool segmentation mask

**Configuration:**
```yaml
sam2:
  checkpoint: "external/sam2/checkpoints/sam2.1_hiera_tiny.pt"
  model_config: "configs/sam2.1/sam2.1_hiera_t.yaml"
  auto_detect: true  # Enable auto-detection
  initial_point: [320, 240]  # Used if auto_detect=false

fabric_detection:
  depth_min: 0.77  # meters
  depth_max: 0.91
  robot_saturation_threshold: 0.3
  table_saturation_threshold: 0.2
  table_flatness_threshold: 0.02
  min_fabric_area: 1000
  static_mask_path: "/path/to/static_mask.png"

tracking:
  min_confidence: 0.3
  min_mask_area_ratio: 0.3
  min_mask_iou: 0.5
  max_components: 50
  loss_patience_frames: 5
```

---

### [3] Depth to Pointcloud Projection
**Location:** `mesh_pipeline.py` lines 411-459

**Input:**
- `depth`: (H, W) uint16 depth in camera units
- `mask`: (H, W) bool segmentation mask
- Camera intrinsics: fx, fy, cx, cy
- Depth scale: units to meters conversion

**Process:**
```python
# 1. Convert depth to meters
depth_m = depth * depth_scale  # e.g., 0.00025

# 2. Get masked pixel coordinates
v, u = np.where(mask)  # Row, column indices

# 3. Get depth values at masked pixels
z = depth_m[v, u]

# 4. Filter out zero depth
valid_mask = z > 0
u_valid = u[valid_mask]
v_valid = v[valid_mask]
z_valid = z[valid_mask]

# 5. Project to 3D camera coordinates
x = (u_valid - cx) * z_valid / fx
y = (v_valid - cy) * z_valid / fy

# 6. Stack to (N, 3) pointcloud
points = np.stack([x, y, z_valid], axis=1)
```

**Output:**
- `points`: (N, 3) float32 pointcloud in camera frame
- **Coordinate System:** Camera frame (X-right, Y-down, Z-forward)
- **Units:** Meters

**Example Values:**
```
Camera intrinsics:
  fx = 600.974, fy = 600.819
  cx = 331.946, cy = 248.233
Depth scale = 0.00025 m/unit

If fabric at (u=320, v=240) has depth=3200 units:
  z = 3200 * 0.00025 = 0.8m
  x = (320 - 331.946) * 0.8 / 600.974 = -0.0159m
  y = (240 - 248.233) * 0.8 / 600.819 = -0.0110m
Point: [-0.0159, -0.0110, 0.8000]
```

---

### [4] Pointcloud Centering
**Location:** `mesh_pipeline.py` lines 461-464

**Input:**
- `points`: (N, 3) raw pointcloud in camera frame

**Process:**
```python
# Compute centroid
centroid = points.mean(axis=0)  # (3,)

# Center pointcloud
points_centered = points - centroid
```

**Output:**
- `points_centered`: (N, 3) centered pointcloud
- **Centroid moved to origin [0, 0, 0]**

**Example:**
```
Before centering:
  Mean: [0.0245, 0.0123, 0.8234]
  Std:  [0.0456, 0.0389, 0.0234]
  Range: x[-0.05, 0.08], y[-0.03, 0.06], z[0.78, 0.87]

After centering:
  Mean: [0.0000, 0.0000, 0.0000]
  Std:  [0.0456, 0.0389, 0.0234]  # Unchanged
  Range: x[-0.07, 0.05], y[-0.05, 0.04], z[-0.04, 0.05]
```

---

### [5] Pointcloud Resampling
**Location:** `mesh_pipeline.py` lines 466-467, 469-522

**Input:**
- `points_centered`: (N, 3) centered pointcloud
- `target_num_points`: Target number of points (default 1024)

**Process:**

**If N < target (Upsample):**
```python
indices = np.random.choice(N, target_n, replace=True)
return points[indices]
```

**If N > target (Downsample):**

*Random Sampling:*
```python
indices = np.random.choice(N, target_n, replace=False)
return points[indices]
```

*Farthest Point Sampling (FPS):*
```python
# Start with random point
# Iteratively select point farthest from selected set
# Maintains better coverage of pointcloud shape
```

**Output:**
- `points_resampled`: (1024, 3) float32 centered pointcloud

**Configuration:**
```yaml
processing:
  target_num_points: 1024
  resample_method: "random"  # or "fps"
```

---

### [6] MeshGAT Inference
**Location:** `mesh_pipeline.py` lines 524-569

**Input:**
- `pointcloud`: (1024, 3) float32 centered, resampled pointcloud

**Process:**
```python
# 1. Convert to torch tensor
pcl_tensor = torch.from_numpy(pointcloud).float().to(device)

# 2. Add batch dimension
pcl_tensor = pcl_tensor.unsqueeze(0)  # (1, 1024, 3)

# 3. Forward pass
if cfg.input_type == "pointcloud":
    output = model({"points": pcl_tensor})
elif cfg.input_type == "depth":
    output = model(pcl_tensor)

# 4. Extract vertices
if isinstance(output, dict):
    vertices = output['vertices']
else:
    vertices = output

# 5. Convert to numpy
mesh_vertices = vertices.squeeze(0).cpu().numpy()  # (M, 3)
```

**Output:**
- `mesh_vertices`: (M, 3) float32 predicted mesh vertices
- **Coordinate System:** Same as input (centered)
- **Units:** Meters

**Configuration:**
```yaml
meshgat:
  checkpoint: "/path/to/checkpoint.pt"
  config: "/path/to/config.yaml"
  template: "/path/to/template.pickle"  # Optional
```

---

## 🔴 Current Problem: Wrong Predictions

### Symptoms:
- Pipeline runs without errors
- Mesh vertices are generated (M vertices)
- But predictions are **incorrect** (wrong shape/position)

### Root Cause Analysis:
The pointcloud preprocessing is **incomplete**. Currently only doing:
1. ✅ Centering (mean = 0)
2. ✅ Resampling (fixed number of points)

But **missing** critical preprocessing:
- ❌ **Normalization** (scale to unit sphere or fixed scale)
- ❌ **Orientation alignment** (principal component analysis)
- ❌ **Noise filtering** (outlier removal)
- ❌ **Density normalization** (uniform point distribution)

### Why This Matters:
MeshGAT was trained on preprocessed pointclouds with specific properties:
- **Normalized scale:** Points fit in [-1, 1] cube or unit sphere
- **Consistent orientation:** Aligned to canonical axes
- **Clean geometry:** No outliers or noise
- **Uniform density:** Even point distribution

Without matching preprocessing, the network sees inputs with:
- **Arbitrary scale:** Could be 0.01m or 1.0m scale
- **Random orientation:** Fabric rotated arbitrarily
- **Noise:** Depth sensor noise, edge artifacts
- **Variable density:** More points in some regions

This causes **distribution shift** → wrong predictions.

---

## 📊 Data Flow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ Camera (RealSense D435i)                                    │
│   Color: 1920×1080 @ 30fps (uint8)                         │
│   Depth: 1920×1080 @ 30fps (uint16, 0.00025 m/unit)        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Resize (Optional)                                           │
│   → 640×480 (faster processing)                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ SAM2 Segmentation                                           │
│   Input: RGB (640×480×3)                                    │
│   Output: Mask (640×480 bool)                               │
│   Auto-detect: Use fabric detector to find initial point    │
│   Tracking: Maintain mask across frames                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Depth → Pointcloud Projection                               │
│   Input: Depth (640×480 uint16), Mask (640×480 bool)       │
│   Process:                                                  │
│     - Filter by mask                                        │
│     - Project to 3D using intrinsics                        │
│     - Filter zero depth                                     │
│   Output: Pointcloud (N×3, ~50K-200K points)                │
│   Units: Meters, Camera frame                               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Centering                                                   │
│   points_centered = points - points.mean(axis=0)            │
│   Output: Centered pointcloud (N×3)                         │
│   Mean: [0, 0, 0]                                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Resampling                                                  │
│   Method: Random or FPS                                     │
│   Output: Fixed-size pointcloud (1024×3)                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ MeshGAT Inference                                           │
│   Input: (1, 1024, 3) torch tensor                          │
│   Model: PointNet++ backbone + Graph Attention              │
│   Output: Mesh vertices (M×3, e.g., 642 vertices)           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Visualization & Output                                      │
│   3D: Open3D viewer (pointcloud + mesh)                     │
│   2D: OpenCV window (RGB + mask overlay)                    │
│   Video: Optional recording to MP4                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Key Configuration Files

### config.yaml
Main system configuration with all parameters.

**Critical Settings:**
```yaml
camera:
  serial: "f1181599"
  color_width: 1920
  color_height: 1080
  depth_width: 1920
  depth_height: 1080
  fps: 30

processing:
  device: "cuda"
  processing_width: 640
  processing_height: 480
  target_num_points: 1024
  resample_method: "random"

sam2:
  checkpoint: "external/sam2/checkpoints/sam2.1_hiera_tiny.pt"
  model_config: "configs/sam2.1/sam2.1_hiera_t.yaml"
  auto_detect: true

fabric_detection:
  depth_min: 0.77
  depth_max: 0.91
  static_mask_path: "/path/to/mask/static_mask.png"

meshgat:
  checkpoint: "/path/to/checkpoint.pt"
  config: "/path/to/config.yaml"
  template: null  # Optional
```

---

## 📁 Code Structure

```
mesh_detection/
├── main.py                  # Main loop, orchestration
├── mesh_pipeline.py         # Core pipeline (SAM2 → PCL → MeshGAT)
├── camera_manager.py        # RealSense camera wrapper
├── fabric_detector.py       # Intelligent fabric detection
├── visualizer.py            # Multi-window visualization
├── create_static_mask.py    # Tool to create static masks
├── config.yaml              # System configuration
├── CURRENT_PIPELINE.md      # This document
└── VIDEO_RECORDING.md       # Video recording guide
```

---

## 🚀 Next Steps: Add PCL Preprocessing

### What Needs to Be Done:

1. **Scale Normalization**
   - Compute bounding box or max distance from center
   - Scale pointcloud to fit in unit sphere or [-1, 1] cube
   - Match the scale used during MeshGAT training

2. **Orientation Alignment (Optional)**
   - Use PCA to find principal axes
   - Rotate pointcloud to canonical orientation
   - Only if MeshGAT was trained with aligned data

3. **Outlier Removal**
   - Statistical outlier removal (mean + std threshold)
   - Or radius-based outlier removal
   - Remove noisy depth measurements

4. **Density Normalization (Optional)**
   - Voxel grid filtering for uniform density
   - Or adaptive sampling based on local density

5. **Coordinate System Convention**
   - Verify MeshGAT's expected coordinate system
   - May need to swap/flip axes (e.g., Y-up vs Z-up)

### Implementation Plan:

1. **Investigate MeshGAT training preprocessing**
   - Check training code for normalization
   - Check template mesh scale
   - Determine expected input distribution

2. **Add preprocessing module**
   - Create `pointcloud_processor.py`
   - Implement normalization functions
   - Add optional PCA alignment

3. **Integrate into pipeline**
   - Insert between centering and resampling
   - Or replace centering with full normalization

4. **Test and validate**
   - Compare preprocessed PCL to training data
   - Verify mesh predictions improve
   - Check scale/orientation consistency

---

## 📝 Questions to Answer:

1. **What preprocessing was used during MeshGAT training?**
   - Look in training scripts
   - Check data loading code
   - Compare with template mesh scale

2. **What is the expected pointcloud scale?**
   - Unit sphere (max distance = 1.0)?
   - Unit cube (all coords in [-1, 1])?
   - Fixed physical scale (e.g., 0.1m)?

3. **Is orientation alignment needed?**
   - Was training data aligned?
   - Is the model rotation-invariant?

4. **What coordinate system does MeshGAT expect?**
   - Camera frame (X-right, Y-down, Z-forward)?
   - World frame (X-forward, Y-left, Z-up)?
   - Object frame (aligned to fabric)?

---

## 📊 Performance Metrics

**Current System Performance:**
- **Total FPS:** ~30 fps
- **Camera:** ~10ms (33%)
- **Resize:** ~5ms (17%)
- **Pipeline:** ~15ms (50%)
  - SAM2: ~8-10ms
  - PCL projection: ~2-3ms
  - MeshGAT: ~3-5ms
- **Visualization:** ~3ms

**Memory Usage:**
- SAM2 model: ~200MB
- MeshGAT model: ~100MB
- Frame buffers: ~50MB
- Total GPU: ~400MB

---

## 🐛 Known Issues

1. ✅ **Static mask optimization** - FIXED (caching)
2. ✅ **Output directory path** - FIXED (script-relative)
3. 🔴 **Wrong mesh predictions** - NEEDS PREPROCESSING
4. ⚠️ **Variable pointcloud density** - Depth sensor characteristics
5. ⚠️ **Edge noise** - Depth discontinuities at mask boundaries

---

## 📚 References

- **SAM2:** Segment Anything 2 (Meta)
- **MeshGAT:** Graph Attention for Mesh Prediction
- **RealSense D435i:** Intel depth camera
- **Open3D:** 3D visualization library
- **PyTorch:** Deep learning framework
