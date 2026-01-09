# SAM2 + FabricPointCloud + MeshGAT Data Flow

This document traces data through the complete pipeline, showing types, dimensions, and CPU/GPU transfers.

---

## 📸 **1. Camera Capture (RealSense L515)**

**Location**: CPU (Host Memory)

```
RealSenseCamera.async_read_both()
  ↓
RGB:   np.ndarray[uint8]     shape=(480, 640, 3)   device=CPU   [color image]
Depth: np.ndarray[uint16]    shape=(480, 640)      device=CPU   [depth in sensor units]
```

**Notes:**
- RGB: uint8 values [0, 255]
- Depth: uint16 raw sensor values (multiply by depth_scale=0.00025 for meters)
- Both arrays are contiguous NumPy arrays in system RAM

---

## 🎯 **2. SAM2 Segmentation**

### 2a. Input Conversion
**Location**: CPU → GPU

```
rgb: np.ndarray[uint8]     shape=(480, 640, 3)   device=CPU
  ↓ [copy to GPU]
rgb_gpu: torch.Tensor[uint8]     shape=(480, 640, 3)   device=CUDA
  ↓ [normalize to float32]
rgb_gpu: torch.Tensor[float32]   shape=(480, 640, 3)   device=CUDA   [values 0.0-1.0]
```

### 2b. SAM2 Forward Pass
**Location**: GPU

```
SAM2CameraRunner.__call__(rgb)
  ↓ [with torch.autocast(dtype=bfloat16)]
  ↓ [first frame: load_first_frame() + add_new_prompt()]
  ↓ [subsequent: track()]
  ↓
mask_logits: torch.Tensor[float32]   shape=(1, 480, 640)   device=CUDA
  ↓ [threshold > 0.0]
mask_gpu: torch.Tensor[bool]         shape=(480, 640)      device=CUDA
```

### 2c. Output Conversion
**Location**: GPU → CPU

```
mask_gpu: torch.Tensor[bool]     shape=(480, 640)   device=CUDA
  ↓ [.cpu().numpy()]
mask: np.ndarray[bool]            shape=(480, 640)   device=CPU
```

**Summary:**
- Input: RGB uint8 (CPU) → GPU
- Processing: bfloat16 inference on GPU
- Output: Boolean mask (CPU)
- Time: ~17ms

---

## ☁️ **3. FabricPointCloud Processor**

### 3a. Input Preparation
**Location**: CPU

```
rgb:   np.ndarray[uint8]      shape=(480, 640, 3)   device=CPU
depth: np.ndarray[uint16]     shape=(480, 640)      device=CPU
mask:  np.ndarray[bool]       shape=(480, 640)      device=CPU
  ↓ [depth to meters: depth * depth_scale]
depth_m: np.ndarray[float32]  shape=(480, 640)      device=CPU   [meters]
  ↓ [apply mask]
masked_depth: np.ndarray[float32]  shape=(N_masked,)   device=CPU  [N_masked = mask.sum()]
```

### 3b. Depth to 3D Points
**Location**: CPU

```
Camera Intrinsics:
  fx = 600.974, fy = 600.819
  cx = 331.946, cy = 248.234

For each masked pixel (u, v):
  Z = depth_m[v, u]
  X = (u - cx) * Z / fx
  Y = (v - cy) * Z / fy
  
  ↓
points_3d: np.ndarray[float32]    shape=(N_masked, 3)   device=CPU   [XYZ in meters]
```

### 3c. Point Cloud Sampling
**Location**: CPU

```
points_3d: np.ndarray[float32]    shape=(N_masked, 3)   device=CPU   [variable N]
  ↓ [random sample or FPS downsample]
sampled_points: np.ndarray[float32]   shape=(1024, 3)   device=CPU   [fixed size]
```

### 3d. Centering (Zero-Mean)
**Location**: CPU

```
sampled_points: np.ndarray[float32]   shape=(1024, 3)   device=CPU
  ↓ [subtract mean]
centroid = sampled_points.mean(axis=0)    shape=(3,)
centered_points = sampled_points - centroid
  ↓
pcl: np.ndarray[float32]   shape=(1024, 3)   device=CPU   [mean ≈ (0, 0, 0)]
```

**Summary:**
- Input: RGB + Depth + Mask (all CPU)
- Processing: Depth→3D projection, sampling, centering (CPU)
- Output: Centered pointcloud (CPU)
- Time: ~2ms

---

## 🧠 **4. MeshGAT Prediction**

### 4a. Input Conversion
**Location**: CPU → GPU

```
pcl: np.ndarray[float32]       shape=(1024, 3)      device=CPU
  ↓ [torch.from_numpy()]
pcl_tensor: torch.Tensor[float32]   shape=(1024, 3)   device=CPU
  ↓ [.unsqueeze(0)]
pcl_batch: torch.Tensor[float32]    shape=(1, 1024, 3)   device=CPU   [add batch dim]
  ↓ [.to(device="cuda")]
pcl_gpu: torch.Tensor[float32]      shape=(1, 1024, 3)   device=CUDA
```

### 4b. MeshGAT Forward Pass
**Location**: GPU

```
Input:
  batch = {"points": pcl_gpu}
  pcl_gpu: torch.Tensor[float32]   shape=(1, 1024, 3)   device=CUDA

MeshGAT Architecture:
  PointNet++ Encoder
    ↓ [set abstraction layers]
  global_features: torch.Tensor[float32]   shape=(1, D_feat)   device=CUDA
    ↓ [message passing on latent graph]
  Graph Attention Network (GAT)
    ↓ [N_layers × (edge update + node update with attention)]
    ↓ [uses torch_scatter.scatter_add() and scatter_softmax()]
  latent_graph: torch.Tensor[float32]   shape=(1, V_latent, D_node)   device=CUDA
    ↓ [MLP decoder]
  mesh_vertices: torch.Tensor[float32]   shape=(1, 442, 3)   device=CUDA

Output:
  pred_mesh: torch.Tensor[float32]   shape=(1, 442, 3)   device=CUDA
```

### 4c. Output Conversion
**Location**: GPU → CPU

```
pred_mesh: torch.Tensor[float32]   shape=(1, 442, 3)   device=CUDA
  ↓ [.squeeze(0)]
pred_mesh: torch.Tensor[float32]   shape=(442, 3)      device=CUDA
  ↓ [.cpu().numpy()]
mesh_vertices: np.ndarray[float32]  shape=(442, 3)      device=CPU   [XYZ in meters]
```

**Summary:**
- Input: Pointcloud (CPU) → GPU
- Processing: PointNet++ + GAT (GPU, fp32)
- Output: Mesh vertices (CPU)
- Time: ~10-13ms

---

## 📦 **5. Final Output Package**

**Location**: CPU (ready for robot observation dict)

```
observation = {
    "observation.images.camera_rgb": np.ndarray[uint8]    shape=(480, 640, 3)   device=CPU,
    "observation.images.camera_depth": np.ndarray[uint16] shape=(480, 640)      device=CPU,
    "pcl": np.ndarray[float32]                            shape=(1024, 3)       device=CPU,
    "mesh_vertices": np.ndarray[float32]                  shape=(442, 3)        device=CPU,
}
```

---

## 🔄 **Complete Pipeline Summary**

```
                                CPU                                         GPU
                                ───                                         ───

┌──────────────────┐
│  RealSense L515  │
│ RGB + Depth Read │
└────────┬─────────┘
         │ RGB: (480,640,3) uint8
         │ Depth: (480,640) uint16
         ▼
┌────────────────────┐
│   SAM2 Segment     │  ──────────────────────────→  ┌──────────────────────┐
│  rgb → mask        │     Copy RGB to GPU            │  SAM2 Inference      │
└────────┬───────────┘  ←──────────────────────────  │  bfloat16            │
         │                 Copy mask to CPU            │  track() ~17ms       │
         │ mask: (480,640) bool                        └──────────────────────┘
         ▼
┌────────────────────────┐
│  FabricPointCloud      │
│  • depth → 3D points   │  (All on CPU)
│  • mask filter         │  ~2ms
│  • sample 1024 pts     │
│  • center (zero mean)  │
└────────┬───────────────┘
         │ pcl: (1024,3) float32
         ▼
┌────────────────────┐
│  MeshGAT Predict   │  ──────────────────────────→  ┌──────────────────────┐
│  pcl → mesh        │     Copy pcl to GPU            │  MeshGAT Forward     │
└────────┬───────────┘  ←──────────────────────────  │  PointNet++ + GAT    │
         │                 Copy mesh to CPU            │  fp32  ~10-13ms      │
         │ mesh: (442,3) float32                       └──────────────────────┘
         ▼
┌────────────────────────┐
│  Robot Observation     │
│  Dict Output           │
│  • rgb, depth          │
│  • pcl, mesh_vertices  │
└────────────────────────┘
```

---

## 🎯 **Memory Transfer Analysis**

| Transfer | Direction | Data | Size | Frequency |
|----------|-----------|------|------|-----------|
| RGB → SAM2 | CPU→GPU | uint8 image | 921 KB | Every frame |
| SAM2 mask → CPU | GPU→CPU | bool mask | 307 KB | Every frame |
| Pointcloud → MeshGAT | CPU→GPU | float32 pcl | 12 KB | Every frame |
| MeshGAT mesh → CPU | GPU→CPU | float32 mesh | 5 KB | Every frame |
| **Total per frame** | **Bidirectional** | | **~1.2 MB** | **30 FPS** |

**Bandwidth usage**: 1.2 MB × 30 FPS = **36 MB/s** (negligible for PCIe 3.0: ~16 GB/s)

---

## 🔍 **Data Type Evolution**

```
RGB Journey:
  uint8 (CPU) → float32 (GPU) → [SAM2] → bool mask (GPU) → bool mask (CPU)

Depth Journey:
  uint16 (CPU) → float32 meters (CPU) → masked float32 (CPU) 
  → 3D points (CPU) → sampled points (CPU) → centered pcl (CPU)
  → float32 tensor (GPU) → [MeshGAT] → mesh vertices (GPU) → mesh vertices (CPU)

Key Conversions:
  1. uint8 → float32: normalization for neural networks
  2. uint16 → float32: depth units to meters
  3. bfloat16: SAM2 inference (lower precision for speed)
  4. bool: binary segmentation mask
  5. float32: standard precision for 3D geometry
```

---

## ⚡ **Performance Hotspots**

| Component | Time | Location | Bottleneck |
|-----------|------|----------|------------|
| SAM2 | 17ms | GPU | ⚠️ 50% of total time |
| MeshGAT | 10-13ms | GPU | 30% of total time |
| Pointcloud | 2ms | CPU | Minimal |
| Memory Transfers | <1ms | PCIe | Negligible |
| **Total** | **~32ms** | | **30 FPS capable ✅** |

---

## 💡 **Key Insights**

1. **GPU is well-utilized**: Both major components (SAM2, MeshGAT) run on GPU
2. **Memory transfers are minimal**: Only small tensors move between CPU/GPU
3. **CPU work is lightweight**: Pointcloud conversion is fast (<2ms)
4. **Data stays in fp32**: Except SAM2 which uses bfloat16 for inference
5. **No unnecessary copies**: Data flows efficiently through the pipeline

---

## 🚀 **Optimization Notes**

If you need to optimize further:

1. **Keep data on GPU longer**: Could keep pointcloud on GPU if MeshGAT accepts it
2. **Batch processing**: Process multiple frames together (trades latency for throughput)
3. **Quantization**: Could quantize MeshGAT to int8 (may lose accuracy)
4. **Async transfers**: Overlap CPU-GPU transfers with computation (already mostly async)
5. **Lower resolution**: Reduce camera resolution if acceptable (e.g., 480x360)

Currently the pipeline is well-optimized and achieves real-time 30 FPS! 🎉
