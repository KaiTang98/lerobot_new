# Standalone Mesh Detection System

Real-time fabric mesh prediction from RealSense RGB-D camera using SAM2 and MeshGAT.

## Features

- ✅ Async RealSense L515 RGB-D capture (30 FPS)
- ✅ SAM2 fabric segmentation
- ✅ PointCloud generation from masked depth
- ✅ MeshGAT mesh prediction
- ✅ Real-time 2D visualization (OpenCV)
- ✅ Real-time 3D visualization (Open3D)
- ✅ FPS monitoring
- ✅ Keyboard controls (pause, reset, save)

## Installation

### Prerequisites

```bash
# Install system dependencies (for OpenCV GUI support)
sudo apt-get install -y libgtk2.0-dev libgtk-3-dev pkg-config

# Activate your conda environment
conda activate lerobot_new
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Requires `opencv-python` (NOT `opencv-python-headless`) for GUI support.

## Quick Start

```bash
# With default config
python main.py

# With custom options
python main.py \
  --camera-serial f1181599 \
  --device cuda \
  --config config.yaml
```

## Configuration

Edit `config.yaml` to customize:

```yaml
camera:
  serial: "f1181599"
  width: 640
  height: 480
  fps: 30

sam2:
  checkpoint: "external/sam2/checkpoints/sam2.1_hiera_tiny.pt"
  initial_point: [320, 240]  # Point on fabric

meshgat:
  checkpoint: "/path/to/meshgat.pt"
  config: "/path/to/config.yaml"
  target_num_points: 1024

processing:
  device: "cuda"  # or "cpu"
  resample_method: "random"  # or "fps"

visualization:
  enable_3d: true
  mesh_color: [0.8, 0.2, 0.2]
  pointcloud_color: [0.2, 0.8, 0.2]
```

## Keyboard Controls

- `Q` or `ESC`: Quit
- `SPACE`: Pause/Resume
- `R`: Reset SAM2 tracking
- `S`: Save current frame

## Module Overview

### `camera_manager.py`
- RealSense camera initialization
- Async RGB-D capture
- Thread-safe frame access

### `mesh_pipeline.py`
- SAM2 segmentation
- Depth → PointCloud projection
- PointCloud centering & resampling
- MeshGAT inference

### `visualizer.py`
- 2D OpenCV overlay (RGB + mask)
- 3D Open3D viewer (mesh + pointcloud)
- FPS counter
- Keyboard input handling

### `main.py`
- Main application loop
- Configuration management
- Module orchestration

## Performance

**Expected Performance:**
- Camera: 30 FPS
- SAM2: ~50-100ms per frame
- MeshGAT: ~30-50ms per frame
- **Target: 10-15 FPS** (66-100ms per iteration)

## Output

Saved frames contain:
- `pointcloud.npy`: (N, 3) float32 centered pointcloud
- `mesh_vertices.npy`: (M, 3) float32 predicted mesh vertices
- `mask.npy`: (H, W) bool segmentation mask

## Troubleshooting

### OpenCV GUI Error
```
cv2.error: The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support.
```

**Solution:**
```bash
pip uninstall opencv-python-headless
pip install opencv-python
```

### Camera Not Found
```
RuntimeError: Failed to initialize camera
```

**Solution:**
- Check camera is connected: `lsusb | grep Intel`
- Check serial number: `rs-enumerate-devices`
- Verify permissions: Add user to `plugdev` group

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce `target_num_points` in config
- Use smaller SAM2 model (hiera_tiny)
- Set `device: "cpu"` in config

## Development

### Test Individual Modules

```bash
# Test camera
python camera_manager.py --serial f1181599 --duration 5

# Test visualizer
python visualizer.py
```

### Project Structure

```
mesh_detection/
├── README.md              # This file
├── config.yaml            # Configuration
├── requirements.txt       # Dependencies
├── main.py               # Main application
├── camera_manager.py     # Camera module
├── mesh_pipeline.py      # Processing pipeline
├── visualizer.py         # Visualization
└── output/               # Saved results
```

## License

Same as parent lerobot project.

## Acknowledgments

- **SAM2**: Meta's Segment Anything 2
- **MeshGAT**: Mesh prediction model
- **RealSense**: Intel RealSense SDK
- **Open3D**: 3D visualization library
