# Processor Cheatsheet - One Page Reference

## 🚀 Quick Start (Copy-Paste Template)

```python
from dataclasses import dataclass
from typing import Any
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry

@ProcessorStepRegistry.register("my_processor_name")
@dataclass
class MyProcessor(ObservationProcessorStep):
    input_key: str
    output_key: str
    
    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        obs[self.output_key] = process(obs[self.input_key])
        return obs
    
    def transform_features(self, features):
        obs_features = features[PipelineFeatureType.OBSERVATION].copy()
        obs_features[self.output_key] = PolicyFeature(shape=(N,), dtype="float32")
        features[PipelineFeatureType.OBSERVATION] = obs_features
        return features
```

---

## 📊 Class Hierarchy (Visual)

```
                    ProcessorStep (ABC)
                          │
         ┌────────────────┼────────────────┐
         │                │                │
ObservationProcessorStep  ActionProcessorStep  (other types)
         │                │
         │                │
    ⭐ USE THIS          For actions
         │
         │
    Your Class Here
```

---

## 🔄 Data Flow Diagram

```
Robot Hardware
    │
    ▼
┌─────────────────┐
│ RobotObservation│  = {"rgb": np.array, "joint_pos": np.array}
└────────┬────────┘
         │
         │ to_transition()
         ▼
┌──────────────────┐
│  EnvTransition   │  = {"observation": {...}, "action": {...}, ...}
└────────┬─────────┘
         │
         │ Step 1: processor1(transition)
         ▼
┌──────────────────┐
│  EnvTransition   │  observation["new_field"] added
└────────┬─────────┘
         │
         │ Step 2: processor2(transition)
         ▼
┌──────────────────┐
│  EnvTransition   │  observation["another_field"] added
└────────┬─────────┘
         │
         │ to_output()
         ▼
┌─────────────────┐
│ RobotObservation│  = {"rgb": ..., "joint_pos": ..., "new_field": ..., "another_field": ...}
└─────────────────┘
         │
         ▼
Log to Dataset / Use in Control
```

---

## 📝 The Two Required Methods

### 1. observation() - THE MAIN METHOD

```python
def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
    """Your processing logic.
    
    Input: observation dict with numpy arrays
    Output: modified observation dict
    
    YOU MUST RETURN THE DICT (modified or new)!
    """
    # Extract
    input_data = observation["input_key"]
    
    # Process
    output_data = do_something(input_data)
    
    # Add/modify
    observation["output_key"] = output_data
    
    return observation  # ⭐ DON'T FORGET TO RETURN!
```

### 2. transform_features() - DECLARE OUTPUT SCHEMA

```python
def transform_features(self, features):
    """Tell the system what new fields you're adding and their shapes.
    
    This is for the dataset system to validate data consistency.
    """
    if PipelineFeatureType.OBSERVATION in features:
        obs_features = features[PipelineFeatureType.OBSERVATION].copy()
        obs_features["output_key"] = PolicyFeature(
            shape=(1024, 3),  # Your output shape
            dtype="float32",   # Your output dtype
        )
        features = features.copy()
        features[PipelineFeatureType.OBSERVATION] = obs_features
    return features
```

---

## 🔗 Chaining Processors

```python
from lerobot.processor.pipeline import RobotProcessorPipeline
from lerobot.processor.converters import observation_to_transition, transition_to_observation

pipeline = RobotProcessorPipeline[RobotObservation, RobotObservation](
    steps=[
        Processor1(param1=value1),
        Processor2(param2=value2),
        Processor3(param3=value3),
    ],
    to_transition=observation_to_transition,
    to_output=transition_to_observation,
)

# Use it
output = pipeline(input_obs)
```

---

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forgot to return `observation` | Always `return observation` at end of `observation()` |
| Didn't implement `transform_features()` | Must implement, even if just `return features` |
| Wrong import path | Use `from lerobot.processor.pipeline import` not `.core` |
| Circular import in factory.py | Import from `.my_processor import` not `from lerobot.processor import` |
| Forgot `@ProcessorStepRegistry.register()` | Add decorator above class |
| Used `register_subclass()` | Wrong method! Use `register()` |
| Called `super().__post_init__()` | Parent has no `__post_init__`, remove call |

---

## 🎯 Your Specific Case: MeshGAT Pipeline

### Current Status:
- ✅ FabricPointCloudProcessorStep created
- ✅ MeshGATObservationProcessorStep created
- ✅ Both tested with synthetic data
- ⏳ Need to wire into factory.py

### What You Need to Do Next:

**File: `src/lerobot/processor/factory.py`**

Add this function:
```python
def make_denso_meshgat_robot_observation_processor(
    camera_intrinsics: np.ndarray,
    depth_scale: float,
    meshgat_checkpoint: str,
    meshgat_config: str,
    sam_runner: Callable,
) -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    
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
            sam_runner=sam_runner,
        ),
        MeshGATObservationProcessorStep(
            checkpoint_path=meshgat_checkpoint,
            config_path=meshgat_config,
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

---

## 📚 Key Files

| File | Line Range | What's There |
|------|-----------|--------------|
| `processor/pipeline.py` | 1439-1468 | `ObservationProcessorStep` definition |
| `processor/pipeline.py` | 286-1136 | `RobotProcessorPipeline` implementation |
| `processor/pipeline.py` | 80-141 | `ProcessorStepRegistry` |
| `processor/factory.py` | All | Pipeline factory functions - WIRE HERE |
| `processor/converters.py` | All | Converter functions (observation_to_transition, etc.) |
| `configs/types.py` | All | `PolicyFeature`, `PipelineFeatureType` definitions |

---

## 🔍 Find Examples

Search the codebase for these to see real implementations:
```bash
# Find all observation processors
grep -r "class.*ObservationProcessorStep" src/lerobot/processor/

# Find all registered processors
grep -r "@ProcessorStepRegistry.register" src/lerobot/processor/

# Find pipeline factories
grep -r "RobotProcessorPipeline\[" src/lerobot/processor/factory.py
```

---

## 💡 Pro Tips

1. **Test early, test often**: Create synthetic data and test your processor before integration
2. **Lazy initialization**: For heavy models, load in first call, not `__init__`
3. **Copy features dict**: Always `.copy()` when modifying in `transform_features()`
4. **Type hints**: Add them! Makes code more maintainable
5. **Docstrings**: Future you will thank present you
6. **Error messages**: Add helpful errors with context about what went wrong

---

## 🎓 Learning Path

1. ✅ Read PROCESSOR_QUICK_REFERENCE.md (5 min)
2. ✅ Read PROCESSOR_CLASS_MAP.md (10 min)
3. ✅ Look at one simple example: `device_processor.py` (5 min)
4. ✅ Look at one complex example: `normalize_processor.py` (10 min)
5. ⏳ Implement your processor (30-60 min)
6. ⏳ Test with synthetic data (15 min)
7. ⏳ Wire into factory.py (15 min)
8. ⏳ Integration test with real robot (30 min)

**Total time: ~2-3 hours from zero to working pipeline**

---

## 🆘 When Stuck

1. **Check this cheatsheet** for quick answers
2. **Search for similar processors** in the codebase
3. **Read error messages carefully** - they're usually specific
4. **Add print statements** in `observation()` to debug data flow
5. **Test each processor independently** before chaining

Remember: You're just transforming a dict. It's simpler than it looks!
