# Processor Quick Reference: The 5 Things You Need to Know

## 1️⃣ The Class Hierarchy (Simplified)

```
ProcessorStep (ABC)
├─ ObservationProcessorStep (ABC)    ← ⭐ USE THIS for sensor processing
│  ├─ FabricPointCloudProcessorStep  ← Your RGB+depth → pointcloud
│  ├─ MeshGATObservationProcessorStep ← Your pointcloud → mesh
│  ├─ NormalizerProcessor
│  └─ DeviceProcessor
│
├─ ActionProcessorStep (ABC)         ← For action transformations
│  ├─ DeltaActionProcessor
│  └─ TorqueActionProcessor
│
└─ (Other specialized steps)
```

**Rule:** If you're processing **observations** (sensors, cameras, state), inherit from `ObservationProcessorStep`.

---

## 2️⃣ The Two Required Methods

### Method 1: `observation()` - The Processing Logic

```python
def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
    """Process the observation dict and return modified version.
    
    Args:
        observation: Raw robot observation dict, e.g.:
            {
                "rgb": np.array([480, 640, 3]),
                "depth": np.array([480, 640]),
                "joint_pos": np.array([6]),
            }
    
    Returns:
        Modified observation with new keys or transformed values
    """
    # Your processing logic here
    # Example: Add a new computed field
    observation["new_field"] = process(observation["input_field"])
    return observation
```

### Method 2: `transform_features()` - Declare Schema Changes

```python
def transform_features(
    self, 
    features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
    """Tell the system how your processor changes feature shapes/types.
    
    This is used by the dataset system to validate data schemas.
    You MUST declare any new observation keys you add.
    """
    if PipelineFeatureType.OBSERVATION in features:
        # Copy existing features
        obs_features = features[PipelineFeatureType.OBSERVATION].copy()
        
        # Declare your new output
        obs_features["new_field"] = PolicyFeature(
            shape=(1024, 3),  # Example: pointcloud with 1024 points
            dtype="float32",
        )
        
        # Return updated features dict
        features = features.copy()
        features[PipelineFeatureType.OBSERVATION] = obs_features
    
    return features
```

---

## 3️⃣ The Complete Template

```python
from dataclasses import dataclass
from typing import Any
import numpy as np

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("my_sensor_processor")
@dataclass
class MySensorProcessor(ObservationProcessorStep):
    """One-line description of what this processor does.
    
    Args:
        input_key: Name of input field in observation dict
        output_key: Name of output field to create
        param1: Some configuration parameter
    """
    input_key: str = "rgb"
    output_key: str = "processed_rgb"
    param1: float = 1.0
    
    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Process the observation."""
        # Extract input
        input_data = observation[self.input_key]
        
        # Do processing
        output_data = self._my_processing_logic(input_data)
        
        # Add to observation
        observation[self.output_key] = output_data
        
        return observation
    
    def transform_features(
        self, 
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Declare the output feature."""
        if PipelineFeatureType.OBSERVATION in features:
            obs_features = features[PipelineFeatureType.OBSERVATION].copy()
            obs_features[self.output_key] = PolicyFeature(
                shape=(H, W, C),  # Replace with your actual shape
                dtype="float32",
            )
            features = features.copy()
            features[PipelineFeatureType.OBSERVATION] = obs_features
        return features
    
    def _my_processing_logic(self, data):
        """Helper method with your actual algorithm."""
        # Your code here
        return processed_data
```

---

## 4️⃣ How to Chain Processors in a Pipeline

```python
from lerobot.processor.pipeline import RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    transition_to_observation,
)

# Create pipeline with multiple steps
pipeline = RobotProcessorPipeline[RobotObservation, RobotObservation](
    steps=[
        Step1(),  # Runs first
        Step2(),  # Runs second
        Step3(),  # Runs third
    ],
    to_transition=observation_to_transition,  # Convert input to EnvTransition
    to_output=transition_to_observation,       # Convert EnvTransition to output
)

# Use it
raw_obs = robot.get_observation()
processed_obs = pipeline(raw_obs)
```

**Key insight:** Each step processes an `EnvTransition`, but the pipeline handles conversions automatically.

---

## 5️⃣ Real Example: Your MeshGAT Pipeline

### Step A: Define Your Processor (Already Done ✅)

```python
# File: src/lerobot/processor/fabric_pointcloud_processor.py

@ProcessorStepRegistry.register("fabric_pointcloud_processor")
@dataclass
class FabricPointCloudProcessorStep(ObservationProcessorStep):
    rgb_key: str = "rgb"
    depth_key: str = "depth"
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 320.0
    cy: float = 240.0
    depth_scale: float = 0.001
    target_num_points: int = 1024
    sam_runner: Callable = None
    output_key: str = "pcl"
    
    def observation(self, obs: dict) -> dict:
        # 1. Get RGB and depth
        rgb = obs[self.rgb_key]
        depth = obs[self.depth_key]
        
        # 2. Run SAM to get mask
        mask = self.sam_runner(rgb)
        
        # 3. Project depth to 3D
        pcl = self._depth_to_pointcloud(depth, mask)
        
        # 4. Center and resample
        pcl = self._center_pointcloud(pcl)
        pcl = self._resample_pointcloud(pcl, self.target_num_points)
        
        # 5. Add to observation
        obs[self.output_key] = pcl
        return obs
    
    def transform_features(self, features):
        # Declare output shape
        obs_features = features[PipelineFeatureType.OBSERVATION].copy()
        obs_features[self.output_key] = PolicyFeature(
            shape=(self.target_num_points, 3),
            dtype="float32",
        )
        features[PipelineFeatureType.OBSERVATION] = obs_features
        return features
```

### Step B: Chain with Other Processors

```python
# File: src/lerobot/processor/factory.py

from .denso_deltapose_strip_remote_action_step import DensoDeltaPoseStripRemoteActionStep
from .fabric_pointcloud_processor import FabricPointCloudProcessorStep
from .mesh_gat_processor import MeshGATObservationProcessorStep

def make_denso_meshgat_robot_observation_processor(
    camera_config,
    meshgat_checkpoint_path,
    meshgat_config_path,
    sam_runner,
) -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    """Create processor pipeline for Denso robot with MeshGAT fabric perception."""
    
    steps = [
        # Step 1: Clean up internal robot fields
        DensoDeltaPoseStripRemoteActionStep(),
        
        # Step 2: Convert RGB+depth → pointcloud
        FabricPointCloudProcessorStep(
            rgb_key="rgb",
            depth_key="depth",
            fx=camera_config.camera_intrinsics[0][0],
            fy=camera_config.camera_intrinsics[1][1],
            cx=camera_config.camera_intrinsics[0][2],
            cy=camera_config.camera_intrinsics[1][2],
            depth_scale=camera_config.depth_scale,
            target_num_points=1024,
            sam_runner=sam_runner,
            output_key="pcl",
        ),
        
        # Step 3: Run MeshGAT on pointcloud
        MeshGATObservationProcessorStep(
            checkpoint_path=meshgat_checkpoint_path,
            config_path=meshgat_config_path,
            input_key="pcl",
            output_key="mesh_vertices",
        ),
    ]
    
    return RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=steps,
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
```

### Step C: Use in Teleoperation

```python
# In the teleop/record main loop:

# Create processor
robot_obs_processor = make_denso_meshgat_robot_observation_processor(
    camera_config=robot.camera.config,
    meshgat_checkpoint_path="path/to/checkpoint.pt",
    meshgat_config_path="path/to/config.yaml",
    sam_runner=my_sam_function,
)

# In the control loop:
while teleoperating:
    # Get raw observation from robot
    raw_obs = robot.get_observation()
    # raw_obs = {"rgb": ..., "depth": ..., "joint_pos": ...}
    
    # Process it through your pipeline
    processed_obs = robot_obs_processor(raw_obs)
    # processed_obs = {
    #     "rgb": ..., 
    #     "depth": ..., 
    #     "joint_pos": ...,
    #     "pcl": array([1024, 3]),          # Added by FabricPointCloud
    #     "mesh_vertices": array([N, 3]),   # Added by MeshGAT
    # }
    
    # Log to dataset
    episode.add_frame(processed_obs, action)
```

---

## 🔍 Common Questions Answered

### Q: Do I need to modify `__call__()`?
**A:** No! `ObservationProcessorStep.__call__()` is already implemented. It extracts `observation` from the transition, calls your `observation()` method, and puts it back.

### Q: What's the difference between `ProcessorStep` and `ObservationProcessorStep`?
**A:** 
- `ProcessorStep`: Base class, works on full `EnvTransition` (observation + action + reward + done + ...)
- `ObservationProcessorStep`: Specialized for observation-only processing, simpler to use

### Q: Why do I need `transform_features()`?
**A:** The dataset system needs to know feature shapes **before** seeing actual data. This method declares your schema changes so datasets can validate consistency.

### Q: Can I have stateful processors (e.g., running averages)?
**A:** Yes! Implement `state_dict()` and `load_state_dict()` to save/load state. See `NormalizerProcessor` for an example.

### Q: How do I test my processor?
**A:** Create a simple script:
```python
processor = MyProcessor(...)
test_obs = {"input_key": np.random.rand(10, 10)}
result = processor.observation(test_obs)
assert "output_key" in result
assert result["output_key"].shape == (expected_shape)
```

### Q: What if my processing is slow (e.g., neural network)?
**A:** Use lazy initialization:
```python
def __init__(self, ...):
    self._model = None  # Don't load yet

def observation(self, obs):
    if self._model is None:
        self._model = load_heavy_model()  # Load on first use
    ...
```

---

## 📊 Processor Execution Flow

```
┌──────────────────────────────────────────────────────────────┐
│                     RobotProcessorPipeline                    │
│                                                               │
│  Input (RobotObservation)                                    │
│         │                                                     │
│         ▼                                                     │
│  ┌─────────────────┐                                        │
│  │ to_transition() │  Convert to EnvTransition              │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────────────────────────────────┐          │
│  │          Step 1: CleanupStep                  │          │
│  │   transition = step1(transition)              │          │
│  └──────────────────┬───────────────────────────┘          │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────┐          │
│  │    Step 2: FabricPointCloudProcessorStep      │          │
│  │   1. Extract observation from transition      │          │
│  │   2. Call self.observation(obs)               │          │
│  │      obs["pcl"] = process(obs["rgb"], ...)    │          │
│  │   3. Put back into transition                 │          │
│  └──────────────────┬───────────────────────────┘          │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────┐          │
│  │    Step 3: MeshGATObservationProcessorStep    │          │
│  │   obs["mesh_vertices"] = model(obs["pcl"])    │          │
│  └──────────────────┬───────────────────────────┘          │
│                     │                                        │
│                     ▼                                        │
│  ┌─────────────────┐                                        │
│  │  to_output()    │  Convert back to RobotObservation     │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  Output (RobotObservation with new fields)                  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎯 Summary: Your Action Items

1. ✅ **Create processor class** inheriting from `ObservationProcessorStep`
2. ✅ **Implement `observation()`** with your processing logic
3. ✅ **Implement `transform_features()`** to declare output schema
4. ✅ **Register with `@ProcessorStepRegistry.register()`**
5. ✅ **Test with synthetic data** (unit test)
6. ⏳ **Create factory function** to build your pipeline
7. ⏳ **Wire into `make_teleop_robot_processors()`** in `factory.py`
8. ⏳ **Integration test** with real robot

**You've completed steps 1-5. Next: step 6 (factory function).**

---

## 📚 Key Files Reference

| File | Purpose | When to Look |
|------|---------|--------------|
| `processor/pipeline.py` | Core classes: `ProcessorStep`, `ObservationProcessorStep`, `RobotProcessorPipeline` | Understanding base classes |
| `processor/factory.py` | Pipeline factories: `make_teleop_robot_processors()`, etc. | ⭐ Wire your processor here |
| `processor/converters.py` | Type converters: `observation_to_transition()`, etc. | Understanding data flow |
| `processor/core.py` | Type definitions: `EnvTransition`, `RobotObservation`, etc. | Understanding data structures |
| `configs/types.py` | Feature types: `PolicyFeature`, `PipelineFeatureType` | For `transform_features()` |

**Pro tip:** Search for `ObservationProcessorStep` in the codebase to find 20+ examples of real processors.
