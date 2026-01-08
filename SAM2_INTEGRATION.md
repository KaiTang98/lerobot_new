# SAM2 Integration Guide

## Overview

SAM2 (Segment Anything Model 2) is integrated as a **callable dependency** for FabricPointCloudProcessorStep, not as a standalone processor.

**Architecture:**
```
FabricPointCloudProcessorStep
  └─ sam_runner: Callable[[RGB], Mask]  ← SAM2 wrapped as a simple function
      └─ SAM2 predictor (from external/sam2/api.py)
```

This is different from MeshGAT because:
- MeshGAT is a separate processor step (observation → observation)
- SAM2 is part of FabricPointCloud's internal processing (RGB → mask)

---

## Setup Steps

### 1. Add SAM2 as a Git Submodule

```bash
cd /home/ktang/ws/lerobot_new

# Add SAM2 repository as submodule
git submodule add https://github.com/facebookresearch/segment-anything-2.git external/sam2/sam2

# Initialize and update
git submodule update --init --recursive
```

### 2. Install SAM2 Dependencies

```bash
# Activate your conda environment
conda activate lerobot_new

# Install SAM2
cd external/sam2/sam2
pip install -e .

# Return to repo root
cd ../../..
```

### 3. Download SAM2 Checkpoint

Choose a model size based on your needs:

```bash
# Create checkpoints directory
mkdir -p external/sam2/checkpoints

cd external/sam2/checkpoints

# Option 1: Large model (best accuracy, slower)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# Option 2: Base+ model (good balance)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt

# Option 3: Small model (fastest, lower accuracy)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt

cd ../../..
```

---

## Usage in Your Pipeline

### Basic Example

```python
from external.sam2.api import create_simple_sam2_runner
from lerobot.processor.fabric_pointcloud_processor import FabricPointCloudProcessorStep

# Create SAM2 runner with a fixed bounding box
sam_runner = create_simple_sam2_runner(
    checkpoint_path="external/sam2/checkpoints/sam2_hiera_large.pt",
    model_cfg="sam2_hiera_l.yaml",
    device="cuda",
    box_prompt=[100, 100, 540, 380],  # [x1, y1, x2, y2] - region of interest
)

# Create processor with SAM2 runner
processor = FabricPointCloudProcessorStep(
    rgb_key="rgb",
    depth_key="depth",
    fx=600.974,
    fy=600.81915,
    cx=331.9461,
    cy=248.23386,
    depth_scale=0.00025,
    target_num_points=1024,
    sam_runner=sam_runner,  # ← SAM2 wrapped as callable
    output_key="pcl",
)

# Use it
observation = {"rgb": rgb_image, "depth": depth_image}
observation = processor.observation(observation)
# observation now has "pcl" key with shape (1024, 3)
```

### Advanced: Custom Prompting

```python
from external.sam2.api import load_sam2_predictor, create_sam2_runner

# Load predictor once
predictor = load_sam2_predictor(
    checkpoint_path="external/sam2/checkpoints/sam2_hiera_large.pt",
    model_cfg="sam2_hiera_l.yaml",
    device="cuda",
)

# Option A: Box prompt
sam_runner = create_sam2_runner(
    predictor=predictor,
    prompt_type="box",
    box_prompt=[100, 100, 540, 380],
)

# Option B: Point prompt (click on fabric)
sam_runner = create_sam2_runner(
    predictor=predictor,
    prompt_type="point",
    point_prompt=(
        [[320, 240], [350, 260]],  # Two points on fabric
        [1, 1],                     # Both are foreground
    ),
)

# Option C: Automatic (segments largest object at center)
sam_runner = create_sam2_runner(
    predictor=predictor,
    prompt_type="automatic",
)
```

---

## Integration with Factory

**File: `src/lerobot/processor/factory.py`**

```python
def make_denso_meshgat_robot_observation_processor(
    camera_intrinsics: np.ndarray,
    depth_scale: float,
    meshgat_checkpoint: str,
    meshgat_config: str,
    sam2_checkpoint: str,
    sam2_box_prompt: list[int],  # [x1, y1, x2, y2]
    device: str = "cuda",
) -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    """Create Denso observation processor with SAM2 + MeshGAT."""
    
    # Import here to avoid circular imports
    from external.sam2.api import create_simple_sam2_runner
    from .fabric_pointcloud_processor import FabricPointCloudProcessorStep
    from .mesh_gat_processor import MeshGATObservationProcessorStep
    from .denso_deltapose_strip_remote_action_step import DensoDeltaPoseStripRemoteActionStep
    
    # Create SAM2 runner
    sam_runner = create_simple_sam2_runner(
        checkpoint_path=sam2_checkpoint,
        model_cfg="sam2_hiera_l.yaml",
        device=device,
        box_prompt=sam2_box_prompt,
    )
    
    # Build pipeline
    steps = [
        DensoDeltaPoseStripRemoteActionStep(),
        FabricPointCloudProcessorStep(
            rgb_key="rgb",
            depth_key="depth",
            fx=camera_intrinsics[0, 0],
            fy=camera_intrinsics[1, 1],
            cx=camera_intrinsics[0, 2],
            cy=camera_intrinsics[1, 2],
            depth_scale=depth_scale,
            target_num_points=1024,
            sam_runner=sam_runner,  # ← SAM2 here
        ),
        MeshGATObservationProcessorStep(
            checkpoint_path=meshgat_checkpoint,
            config_path=meshgat_config,
            input_key="pcl",
            output_key="mesh_vertices",
            device=device,
        ),
    ]
    
    return RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=steps,
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
```

---

## Testing SAM2

Create a test script to verify SAM2 works:

**File: `test_sam2_integration.py`**

```python
#!/usr/bin/env python3
"""Test SAM2 integration for fabric segmentation."""

import numpy as np
import matplotlib.pyplot as plt

from external.sam2.api import create_simple_sam2_runner


def test_sam2_basic():
    """Test basic SAM2 functionality."""
    print("Loading SAM2...")
    sam_runner = create_simple_sam2_runner(
        checkpoint_path="external/sam2/checkpoints/sam2_hiera_large.pt",
        model_cfg="sam2_hiera_l.yaml",
        device="cuda",
        box_prompt=[100, 100, 540, 380],
    )
    
    print("Creating test image...")
    # Create a test RGB image
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("Running SAM2...")
    mask = sam_runner(rgb)
    
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Mask sum: {mask.sum()} pixels")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(rgb)
    ax1.set_title("RGB Image")
    ax2.imshow(mask, cmap='gray')
    ax2.set_title("SAM2 Mask")
    plt.savefig("test_sam2_output.png")
    print("Saved visualization to test_sam2_output.png")
    
    print("✓ SAM2 test passed!")


if __name__ == "__main__":
    test_sam2_basic()
```

Run with:
```bash
python test_sam2_integration.py
```

---

## Configuration Storage

Store SAM2 config in robot config or separate config file:

**Option 1: Robot Config (Simple)**
```yaml
# config/robot/denso_meshgat.yaml
robot:
  type: denso_deltapose
  camera:
    serial: f1181599
    use_depth: true
  
sam2:
  checkpoint: external/sam2/checkpoints/sam2_hiera_large.pt
  model_cfg: sam2_hiera_l.yaml
  box_prompt: [100, 100, 540, 380]  # Fabric region of interest

meshgat:
  checkpoint: path/to/meshgat_checkpoint.pt
  config: path/to/meshgat_config.yaml
```

**Option 2: Separate Processor Config**
```yaml
# config/processor/denso_fabric_meshgat.yaml
observation_processors:
  - type: denso_deltapose_strip_remote_action
  
  - type: fabric_pointcloud_processor
    rgb_key: rgb
    depth_key: depth
    target_num_points: 1024
    sam2:
      checkpoint: external/sam2/checkpoints/sam2_hiera_large.pt
      box_prompt: [100, 100, 540, 380]
  
  - type: meshgat_observation_processor
    input_key: pcl
    output_key: mesh_vertices
    checkpoint: path/to/meshgat_checkpoint.pt
    config: path/to/meshgat_config.yaml
```

---

## Performance Optimization

### 1. Model Selection

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| sam2_hiera_small | 184MB | ~20ms | Good |
| sam2_hiera_base_plus | 319MB | ~35ms | Better |
| sam2_hiera_large | 895MB | ~50ms | Best |

For real-time teleoperation (30 FPS), use **sam2_hiera_small** or **base_plus**.

### 2. Caching Strategy

SAM2 can reuse features for same image:

```python
# In your teleop loop, reuse sam_runner instance
sam_runner = create_simple_sam2_runner(...)  # Create once

while teleoperating:
    obs = robot.get_observation()
    obs = processor.observation(obs)  # sam_runner called here
```

### 3. Run at Lower Frequency

Run SAM2 every N frames instead of every frame:

```python
class CachedSAMRunner:
    def __init__(self, sam_runner, cache_frames=5):
        self.sam_runner = sam_runner
        self.cache_frames = cache_frames
        self.frame_count = 0
        self.cached_mask = None
    
    def __call__(self, rgb):
        if self.frame_count % self.cache_frames == 0:
            self.cached_mask = self.sam_runner(rgb)
        self.frame_count += 1
        return self.cached_mask

# Use cached runner
sam_runner = create_simple_sam2_runner(...)
cached_sam_runner = CachedSAMRunner(sam_runner, cache_frames=5)
```

---

## Differences from MeshGAT Integration

| Aspect | MeshGAT | SAM2 |
|--------|---------|------|
| **Role** | Standalone processor step | Callable dependency |
| **Location** | `MeshGATObservationProcessorStep` | Inside `FabricPointCloudProcessorStep` |
| **Interface** | `observation(obs) → obs` | `Callable[[RGB], Mask]` |
| **Registration** | `@ProcessorStepRegistry.register()` | Not registered (just a function) |
| **Serialization** | Saved with pipeline | Not serialized (recreated at load) |
| **Usage** | Add to pipeline steps | Pass as parameter to FabricPointCloud |

---

## Summary

✅ **What's done:**
- Created `external/sam2/api.py` with SAM2 wrapper functions
- Designed for easy integration with FabricPointCloudProcessorStep

⏳ **What you need to do:**
1. Add SAM2 as submodule
2. Install SAM2 dependencies
3. Download checkpoint
4. Test with `test_sam2_integration.py`
5. Update factory function to create sam_runner
6. Determine box_prompt for your fabric setup

**Next step:** Add SAM2 submodule and test the basic integration.
