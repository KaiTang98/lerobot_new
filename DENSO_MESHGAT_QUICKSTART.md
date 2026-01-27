# Quick Start: Recording with Denso DeltaPose Force + MeshGAT

## 🚀 TL;DR

**With MeshGAT (fabric mesh prediction):**
```bash
lerobot-record --config configs/denso_deltapose_force_meshgat.yaml
```

**Without MeshGAT (standard recording):**
```bash
lerobot-record --config configs/denso_deltapose_force_standard.yaml
```

---

## 📋 Prerequisites

1. **Hardware:**
   - Denso robot with Windows server running (192.168.2.100:12345)
   - RealSense L515 camera (serial: f1181599) connected to Linux PC
   - Quest 3 headset with haptics

2. **Software:**
   - LeRobot environment activated: `conda activate lerobot_new`
   - SAM2 checkpoint downloaded: `external/sam2/checkpoints/sam2.1_hiera_tiny.pt`
   - MeshGAT checkpoint: `/home/ktang/ws/data/mesh_gat/collar_pcl/checkpoint/finalbestmodel_1999_0.00645.pt`

---

## 🎯 Option 1: With MeshGAT (Fabric Manipulation)

### Using YAML Config (Recommended):

```bash
# 1. Edit the config if needed
nano configs/denso_deltapose_force_meshgat.yaml

# 2. Run recording
lerobot-record --config configs/denso_deltapose_force_meshgat.yaml
```

### Using Command Line:

```bash
lerobot-record \
  --robot.type=denso_deltapose_force \
  --robot.server_ip=192.168.2.100 \
  --robot.server_port=12345 \
  --robot.fps=100 \
  --robot.cameras='{
    camera_l515: {
      type: realsense,
      serial_number: f1181599,
      width: 640,
      height: 480,
      fps: 30,
      use_depth: true,
      color_mode: RGB
    }
  }' \
  --robot.enable_meshgat=true \
  --robot.meshgat_camera_key=camera_l515 \
  --robot.meshgat_checkpoint_path=/home/ktang/ws/data/mesh_gat/collar_pcl/checkpoint/finalbestmodel_1999_0.00645.pt \
  --robot.meshgat_config_path=/home/ktang/ws/data/mesh_gat/collar_pcl/checkpoint/config.yaml \
  --robot.sam2_checkpoint_path=external/sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --robot.sam2_initial_point='[320, 240]' \
  --robot.meshgat_target_num_points=1024 \
  --teleop.type=bi_quest_haptics \
  --dataset.repo_id=${HF_USER}/denso_fabric_manipulation \
  --dataset.single_task="Fold the fabric using both arms" \
  --dataset.num_episodes=50 \
  --display_data=true
```

### What Gets Recorded:

The dataset will contain these observations per frame:
- **Robot state**: Joint positions, velocities, forces
- **RGB image**: `observation.images.camera_l515` (640x480)
- **Depth image**: `observation.images.camera_l515_depth` (640x480)
- **Point cloud**: `observation.pcl` (1024 points, XYZ)
- **Mesh vertices**: `observation.mesh_vertices` (442 vertices, XYZ) ✅

**Pipeline:** Camera → SAM2 Segmentation → Pointcloud → MeshGAT → Mesh Prediction

**Performance:** ~30 FPS (see `SAM2_MESHGAT_PERFORMANCE_RESULTS.md`)

---

## 🎯 Option 2: Without MeshGAT (Standard Recording)

### Using YAML Config:

```bash
lerobot-record --config configs/denso_deltapose_force_standard.yaml
```

### Using Command Line:

```bash
lerobot-record \
  --robot.type=denso_deltapose_force \
  --robot.server_ip=192.168.2.100 \
  --robot.server_port=12345 \
  --robot.cameras='{
    camera_l515: {
      type: realsense,
      serial_number: f1181599,
      width: 640,
      height: 480,
      fps: 30,
      use_depth: false,
      color_mode: RGB
    }
  }' \
  --robot.enable_meshgat=false \
  --teleop.type=bi_quest_haptics \
  --dataset.repo_id=${HF_USER}/denso_manipulation \
  --dataset.single_task="Pick and place objects" \
  --dataset.num_episodes=50
```

### What Gets Recorded:

- **Robot state**: Joint positions, velocities, forces
- **RGB image**: `observation.images.camera_l515` (640x480)
- No depth, no pointcloud, no mesh ✅

**Performance:** ~100 FPS (robot control loop)

---

## ⚙️ Configuration Parameters

### MeshGAT Parameters:

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `enable_meshgat` | No | `false` | Enable mesh prediction pipeline |
| `meshgat_camera_key` | If enabled | - | Which camera to use (must have depth) |
| `meshgat_checkpoint_path` | If enabled | - | Path to MeshGAT model checkpoint |
| `meshgat_config_path` | If enabled | - | Path to MeshGAT config YAML |
| `sam2_checkpoint_path` | If enabled | - | Path to SAM2 checkpoint |
| `sam2_initial_point` | If enabled | - | [x, y] point on fabric center |
| `sam2_initial_box` | If enabled | - | [x1, y1, x2, y2] box around fabric |
| `meshgat_device` | No | `cuda` | Device for inference |
| `meshgat_target_num_points` | No | `1024` | Number of pointcloud points |
| `meshgat_input_key` | No | `pcl` | Input key for MeshGAT |
| `meshgat_output_key` | No | `mesh_vertices` | Output key for mesh |

### Camera Parameters:

For MeshGAT, camera MUST have:
- `type: realsense` (only RealSense supports depth)
- `use_depth: true` (enables depth stream)
- `color_mode: RGB` (SAM2 requires RGB)

---

## 🔍 Troubleshooting

### Error: "enable_meshgat=True requires meshgat_checkpoint_path"

**Solution:** Provide all required MeshGAT parameters:
```yaml
enable_meshgat: true
meshgat_checkpoint_path: /path/to/checkpoint.pt
meshgat_config_path: /path/to/config.yaml
sam2_checkpoint_path: external/sam2/checkpoints/sam2.1_hiera_tiny.pt
sam2_initial_point: [320, 240]
meshgat_camera_key: camera_l515
```

### Error: "MeshGAT requires RealSense camera with depth"

**Solution:** Ensure camera config has:
```yaml
cameras:
  camera_l515:
    type: realsense
    use_depth: true  # ← Must be true!
```

### Error: "camera_intrinsics not available"

**Cause:** Camera intrinsics are populated AFTER `camera.connect()`

**Solution:** This is handled automatically by the processor. Intrinsics are read at runtime.

### Performance Issues (< 30 FPS)

**Check:**
1. GPU available: `nvidia-smi`
2. CUDA version: PyTorch 2.7.1 + CUDA 12.6
3. SAM2 checkpoint loaded: `ls external/sam2/checkpoints/`
4. Reduce `meshgat_target_num_points` from 1024 to 512

**See:** `SAM2_MESHGAT_PERFORMANCE_RESULTS.md` for detailed analysis

---

## 📊 Verifying Recording

### Check Dataset Structure:

```bash
lerobot-dataset-viz --repo-id ${HF_USER}/denso_fabric_manipulation --episode-index 0
```

### Check for Mesh Data:

```python
from lerobot.datasets import LeRobotDataset

ds = LeRobotDataset("${HF_USER}/denso_fabric_manipulation")
print(ds.features)  # Should show 'observation.mesh_vertices'

# Get first frame
frame = ds[0]
print(frame.keys())
print(frame["observation.mesh_vertices"].shape)  # Should be (442, 3)
```

---

## 🎓 Advanced Usage

### Multiple Cameras:

```yaml
cameras:
  camera_l515:
    type: realsense
    serial_number: f1181599
    width: 640
    height: 480
    fps: 30
    use_depth: true
    color_mode: RGB
  
  camera_front:
    type: opencv
    index_or_path: 0
    width: 1280
    height: 720
    fps: 30

# Use L515 for MeshGAT
meshgat_camera_key: camera_l515
```

### Different SAM2 Prompts:

**Point prompt** (click on fabric center):
```yaml
sam2_initial_point: [320, 240]
```

**Box prompt** (bounding box around fabric):
```yaml
sam2_initial_box: [100, 100, 540, 380]
```

### Custom MeshGAT Output:

```yaml
meshgat_output_key: fabric_mesh  # Custom key name
```

Then access with: `frame["observation.fabric_mesh"]`

---

## 📝 Summary

**Key Points:**
1. ✅ MeshGAT is **optional** via `enable_meshgat` flag
2. ✅ Use **YAML configs** for cleaner command lines
3. ✅ RealSense camera with `use_depth: true` required for MeshGAT
4. ✅ Works seamlessly with existing `lerobot-record` command
5. ✅ Achieves real-time 30 FPS performance

**Next Steps:**
1. Record dataset with MeshGAT: `lerobot-record --config configs/denso_deltapose_force_meshgat.yaml`
2. Verify mesh data: `lerobot-dataset-viz --repo-id ${HF_USER}/denso_fabric_manipulation`
3. Train policy: `lerobot-train --config lerobot/diffusion_denso --dataset.repo_id=${HF_USER}/denso_fabric_manipulation`

🎉 **You're ready to collect fabric manipulation data with real-time mesh predictions!**
