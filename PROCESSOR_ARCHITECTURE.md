# LeRobot Processor Architecture Guide

> **TL;DR for the impatient:** Use `ObservationProcessorStep` to process robot observations. Implement `observation()` and `transform_features()`. Register with `@ProcessorStepRegistry.register()`. Chain steps in `RobotProcessorPipeline`. Wire into `factory.py`.

---

## 🎯 Core Concept: Three Representations

LeRobot processors bridge three different data representations:

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Robot Dicts    │ ───▶ │  EnvTransition  │ ───▶ │  Batch Tensors  │
│  (hardware I/O) │ ◀─── │  (typed dict)   │ ◀─── │  (training)     │
└─────────────────┘      └─────────────────┘      └─────────────────┘
      RobotObs                  ▲                      PolicyBatch
      RobotAction               │
                                │
                    All processors work here!
```

### **Robot Representation** (Runtime - Teleoperation/Recording)
```python
RobotObservation = {
    "joint_pos": np.array([0.1, 0.2, ...]),  # (6,) float32
    "rgb": np.array([...]),                   # (480, 640, 3) uint8
    "depth": np.array([...]),                 # (480, 640) uint16
}

RobotAction = {
    "joint_target": np.array([0.15, 0.25, ...]),
}
```

### **Transition Representation** (Universal Format)
```python
EnvTransition = {
    TransitionKey.OBSERVATION: RobotObservation,  # dict[str, Any]
    TransitionKey.ACTION: RobotAction,            # dict[str, Any]
    TransitionKey.REWARD: 0.0,
    TransitionKey.DONE: False,
    # ... other fields
}
```

### **Batch Representation** (Training - Neural Networks)
```python
PolicyBatch = {
    "observation.joint_pos": torch.Tensor(shape=[B, T, 6]),      # normalized, batched
    "observation.image": torch.Tensor(shape=[B, T, C, H, W]),    # normalized, batched
    "action": torch.Tensor(shape=[B, T, action_dim]),            # normalized, batched
}
```

---

## 🔧 Processor Components

### **1. ProcessorStep (Abstract Base Class)**

The fundamental building block. All processors inherit from this:

```python
class ProcessorStep(ABC):
    @abstractmethod
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Transform a transition (the main logic)"""
        pass
    
    @abstractmethod
    def transform_features(
        self, 
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Declare how this step changes feature shapes/types (for dataset schema)"""
        pass
```

**Two main subclasses:**

#### **ObservationProcessorStep** (⭐ Your Focus)
```python
class ObservationProcessorStep(ProcessorStep):
    @abstractmethod
    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Process just the observation dict"""
        pass
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # Automatically extracts observation, calls your method, puts it back
        observation = transition[TransitionKey.OBSERVATION]
        processed_obs = self.observation(observation)
        transition[TransitionKey.OBSERVATION] = processed_obs
        return transition
```

**Examples:**
- `FabricPointCloudProcessorStep`: RGB+depth → pointcloud
- `MeshGATObservationProcessorStep`: pointcloud → mesh vertices
- `DensoDeltaPoseStripRemoteActionStep`: cleanup internal fields
- `NormalizerProcessor`: normalize state values

#### **ActionProcessorStep**
```python
class ActionProcessorStep(ProcessorStep):
    @abstractmethod
    def action(self, action: Any) -> Any:
        """Process just the action"""
        pass
```

**Examples:**
- `DeltaActionProcessor`: absolute → delta actions
- `DeviceProcessor`: move tensors to GPU

---

### **2. RobotProcessorPipeline (Chain of Steps)**

Connects multiple `ProcessorStep`s into a pipeline:

```python
pipeline = RobotProcessorPipeline[InputType, OutputType](
    steps=[
        Step1(),
        Step2(),
        Step3(),
    ],
    to_transition=input_to_transition,    # Converter: Input → EnvTransition
    to_output=transition_to_output,       # Converter: EnvTransition → Output
)

output = pipeline(input_data)
```

**Key insight:** All steps work on `EnvTransition`, but pipeline has custom input/output types via converters.

---

### **3. ProcessorStepRegistry (Registration System)**

Makes steps discoverable and configurable:

```python
@ProcessorStepRegistry.register("my_processor")
@dataclass
class MyProcessorStep(ObservationProcessorStep):
    param1: float = 1.0
    param2: str = "default"
    
    def observation(self, obs: dict) -> dict:
        # Your logic here
        return obs
    
    def transform_features(self, features):
        # Declare output shape changes
        return features
```

**Benefits:**
- Can be instantiated from config: `ProcessorStepRegistry.from_config("my_processor", {...})`
- Can be saved/loaded with checkpoints
- Enables factory functions

---

## 🔄 Processor Flow in Different Contexts

### **Context 1: Teleoperation/Recording** (`lerobot-teleoperate`, `lerobot-record`)

```
Teleoperator          Robot Hardware
    │                      │
    ▼                      ▼
┌────────┐          ┌────────────┐
│ Action │          │ Observation│
└───┬────┘          └──────┬─────┘
    │                      │
    │    ┌─────────────────┘
    │    │
    ▼    ▼
┌──────────────────────────────┐
│  Robot Processor Pipelines   │
│  • teleop_action_processor   │  ← Process raw teleop input
│  • robot_action_processor    │  ← Prepare action for robot
│  • robot_observation_proc    │  ← Process robot sensors ⭐ YOUR WORK
└──────────────┬───────────────┘
               │
               ▼
        Send to Robot
        Log to Dataset
```

**Your MeshGAT pipeline fits here:**
```python
robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
    steps=[
        DensoDeltaPoseStripRemoteActionStep(),    # Cleanup
        FabricPointCloudProcessorStep(...),       # RGB+depth → pcl
        MeshGATObservationProcessorStep(...),     # pcl → mesh_vertices
    ],
    to_transition=observation_to_transition,
    to_output=transition_to_observation,
)
```

### **Context 2: Training** (`lerobot-train`)

```
LeRobot Dataset
    │
    ▼
┌────────────────┐
│  Pre-Processor │  ← Normalize, device transfer, temporal chunking
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Policy Model  │  ← Neural network forward pass
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Post-Processor │  ← Denormalize predictions
└────────────────┘
```

**Training uses different pipelines:**
- `PolicyProcessorPipeline[EnvTransition, dict[str, torch.Tensor]]`: transition → batch tensors
- Pre-processor: normalization, device placement, image transforms
- Post-processor: denormalization

---

## 📦 Key Subsystems You Interact With

### **1. LeRobotDataset** (Training Data)
```python
dataset = make_dataset(cfg)  # Loads from HuggingFace Hub or local

# Dataset provides:
dataset.meta.stats          # Statistics for normalization
dataset.meta.features       # Feature schema (shapes, dtypes)
dataset[i]                  # Get a transition
```

### **2. Converters** (`lerobot.processor.converters`)

Bridge between representations:

```python
from lerobot.processor.converters import (
    observation_to_transition,           # RobotObs → EnvTransition
    transition_to_observation,           # EnvTransition → RobotObs
    robot_action_observation_to_transition,  # (Action, Obs) → EnvTransition
)
```

### **3. Factory Functions** (`lerobot.processor.factory`)

Create processor pipelines for specific robot/teleop combos:

```python
def make_teleop_robot_processors(
    teleopConfig: TeleoperatorConfig,
    robotConfig: RobotConfig,
) -> tuple[
    RobotProcessorPipeline,  # teleop_action_processor
    RobotProcessorPipeline,  # robot_action_processor
    RobotProcessorPipeline,  # robot_observation_processor ⭐
]:
    # Returns appropriate pipelines based on robot type
    ...
```

**Your task:** Add Denso+MeshGAT case to this factory.

### **4. Policy Interface** (`lerobot.policies`)

Policies expose processors they need:

```python
class Policy:
    config: PolicyConfig
    
    # Processors are created from policy config + dataset stats:
    pre_processor: PolicyProcessorPipeline   # For training
    post_processor: PolicyProcessorPipeline  # For inference
```

---

## 🎯 Your Development Workflow

### **Step-by-Step: Adding a New Observation Processor**

1. **Create the Processor Class**
   ```python
   @ProcessorStepRegistry.register("my_processor")
   @dataclass
   class MyObservationProcessor(ObservationProcessorStep):
       input_key: str
       output_key: str
       
       def observation(self, obs: dict) -> dict:
           obs[self.output_key] = process(obs[self.input_key])
           return obs
       
       def transform_features(self, features):
           # Declare new feature
           obs_features = features[PipelineFeatureType.OBSERVATION].copy()
           obs_features[self.output_key] = PolicyFeature(
               shape=(N, 3),
               dtype="float32",
           )
           features[PipelineFeatureType.OBSERVATION] = obs_features
           return features
   ```

2. **Add to Pipeline Factory**
   ```python
   # In factory.py
   def make_my_robot_observation_processor():
       return RobotProcessorPipeline[RobotObservation, RobotObservation](
           steps=[
               CleanupStep(),
               MyObservationProcessor(...),
           ],
           to_transition=observation_to_transition,
           to_output=transition_to_observation,
       )
   ```

3. **Wire into Teleop/Record**
   ```python
   # In factory.py: make_teleop_robot_processors
   if robotConfig.type == "my_robot":
       robot_obs_proc = make_my_robot_observation_processor()
   ```

4. **Test**
   ```python
   # Unit test
   processor = MyObservationProcessor(...)
   obs = {"input_key": np.array(...)}
   result = processor.observation(obs)
   assert "output_key" in result
   ```

---

## 📚 Key Files for Processor Development

### **Must Read:**
1. `src/lerobot/processor/pipeline.py`
   - `ProcessorStep`, `ObservationProcessorStep`, `RobotProcessorPipeline`
   - Core abstractions

2. `src/lerobot/processor/factory.py`
   - `make_teleop_robot_processors()` ⭐ Wire your processor here
   - Examples of robot-specific pipelines

3. `src/lerobot/processor/core.py`
   - `EnvTransition`, `TransitionKey`
   - `RobotObservation`, `RobotAction`
   - Data type definitions

4. `docs/source/introduction_processors.mdx`
   - High-level processor documentation

### **Examples to Study:**
- `src/lerobot/processor/normalize_processor.py` - Complex stateful processor
- `src/lerobot/processor/device_processor.py` - Simple stateless processor
- `src/lerobot/processor/denso_deltapose_strip_remote_action_step.py` - Simple observation cleanup

---

## 🚀 Your MeshGAT Pipeline in Context

```python
# This is where your processors fit:

# 1. Robot returns raw observation
obs = robot.get_observation()  # {"rgb": ..., "depth": ..., "joint_pos": ...}

# 2. Robot observation processor pipeline processes it
robot_observation_processor = RobotProcessorPipeline(
    steps=[
        DensoDeltaPoseStripRemoteActionStep(),    # Cleanup internal fields
        FabricPointCloudProcessorStep(            # ⭐ Your step 2
            rgb_key="rgb",
            depth_key="depth",
            fx=camera.config.camera_intrinsics[0][0],
            fy=camera.config.camera_intrinsics[1][1],
            cx=camera.config.camera_intrinsics[0][2],
            cy=camera.config.camera_intrinsics[1][2],
            depth_scale=camera.config.depth_scale,
            target_num_points=1024,
            sam_runner=my_sam_runner,
        ),
        MeshGATObservationProcessorStep(          # ⭐ Your step 1
            checkpoint_path="path/to/checkpoint.pt",
            config_path="path/to/config.yaml",
            input_key="pcl",
            output_key="mesh_vertices",
        ),
    ],
    to_transition=observation_to_transition,
    to_output=transition_to_observation,
)

# 3. Processed observation has new fields
processed_obs = robot_observation_processor(obs)
# processed_obs = {
#     "rgb": ..., 
#     "depth": ..., 
#     "joint_pos": ...,
#     "pcl": np.array([1024, 3]),           # ⭐ Added by FabricPointCloud
#     "mesh_vertices": np.array([N, 3]),    # ⭐ Added by MeshGAT
# }

# 4. Can be logged to dataset or displayed
log_to_dataset(processed_obs)
```

---

## ⚠️ Common Pitfalls

1. **Forgetting `transform_features()`**
   - Every processor MUST implement this to declare schema changes
   - Used by dataset system to understand feature shapes

2. **Wrong import paths**
   - Use `from lerobot.processor.pipeline import ObservationProcessorStep`
   - NOT `from lerobot.processor.core import ...`

3. **Circular imports**
   - In `factory.py`, import from specific modules: `from .my_processor import MyStep`
   - NOT from package: `from lerobot.processor import MyStep`

4. **Processor state**
   - If your processor has learned parameters, implement `state_dict()` and `load_state_dict()`
   - For heavy resources (models), use lazy initialization in first call

5. **Type conversions**
   - Robot observations are numpy arrays
   - Policy batches are torch tensors
   - Know which representation you're working with!

---

## 📝 Summary: What Matters for You

**Focus on:**
1. **ObservationProcessorStep** - Your base class
2. **RobotProcessorPipeline** - How steps are chained
3. **factory.py: make_teleop_robot_processors()** - Where to wire your pipeline
4. **EnvTransition** - The universal data format
5. **transform_features()** - Always implement this!

**Ignore for now:**
- Training pipelines (`PolicyProcessorPipeline`) - different use case
- Dataset internals - just consume `dataset.meta.stats`
- Policy implementations - processors are policy-agnostic
- Config parsing - Draccus handles this automatically

**Your next step (TODO step 3):**
Update `make_teleop_robot_processors()` in `factory.py` to return your MeshGAT-enabled pipeline when `robot.type == "denso_deltapose"` or similar.
