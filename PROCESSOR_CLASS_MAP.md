# pipeline.py Class Map (Annotated for Understanding)

## The Big Picture

```
pipeline.py has ~1700 lines with many classes, but only a few matter for you:

Lines 143-226:  ProcessorStep (ABC)                    ← Base class (abstract)
Lines 1439-1468: ObservationProcessorStep (ABC)        ← ⭐ YOUR PARENT CLASS
Lines 1470-1499: ActionProcessorStep (ABC)             ← For action processing
Lines 286-1136:  RobotProcessorPipeline                ← ⭐ Pipeline container
Lines 1138-1437: PolicyProcessorPipeline               ← Training pipelines (ignore)
Lines 80-141:    ProcessorStepRegistry                 ← ⭐ Registration system
Lines 33-78:     Various helper types                  ← EnvTransition, etc.
```

---

## Class Hierarchy (Simplified)

```python
# ==============================================================================
# LEVEL 1: Abstract Base Classes (You Never Instantiate These)
# ==============================================================================

class ProcessorStep(ABC):
    """Base class for all processor steps.
    
    Every processor must:
    1. Implement __call__(transition) → transition
    2. Implement transform_features(features) → features
    """
    
    @abstractmethod
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Process a complete transition (obs + action + reward + done + ...)"""
        pass
    
    @abstractmethod
    def transform_features(self, features) -> features:
        """Declare how this step changes feature shapes/types"""
        pass
    
    def get_config(self) -> dict:
        """Optional: return config for serialization"""
        return {}
    
    def state_dict(self) -> dict:
        """Optional: return learnable state (e.g., normalization stats)"""
        return {}
    
    def load_state_dict(self, state: dict):
        """Optional: load state"""
        pass


# ==============================================================================
# LEVEL 2: Specialized Abstract Classes (Choose One of These)
# ==============================================================================

class ObservationProcessorStep(ProcessorStep):
    """⭐ USE THIS: Process only the observation part of a transition.
    
    You only need to implement observation() - the __call__() is handled for you!
    """
    
    @abstractmethod
    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Process the observation dict. THIS IS YOUR MAIN METHOD."""
        pass
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Already implemented! Extracts obs, calls your method, puts it back."""
        observation = transition[TransitionKey.OBSERVATION]
        processed_obs = self.observation(observation.copy())
        transition[TransitionKey.OBSERVATION] = processed_obs
        return transition


class ActionProcessorStep(ProcessorStep):
    """Process only the action part of a transition."""
    
    @abstractmethod
    def action(self, action) -> action:
        """Process the action. THIS IS YOUR MAIN METHOD."""
        pass
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Already implemented! Extracts action, calls your method, puts it back."""
        action = transition[TransitionKey.ACTION]
        processed_action = self.action(action)
        transition[TransitionKey.ACTION] = processed_action
        return transition


# ==============================================================================
# LEVEL 3: Your Concrete Classes (What You Actually Write)
# ==============================================================================

@ProcessorStepRegistry.register("fabric_pointcloud_processor")
@dataclass
class FabricPointCloudProcessorStep(ObservationProcessorStep):
    """⭐ YOUR IMPLEMENTATION
    
    You inherit from ObservationProcessorStep and implement:
    1. observation() - the processing logic
    2. transform_features() - schema declaration
    
    That's it! __call__() is already handled by parent class.
    """
    
    # Config parameters (dataclass fields)
    rgb_key: str = "rgb"
    depth_key: str = "depth"
    target_num_points: int = 1024
    # ... etc
    
    def observation(self, obs: dict) -> dict:
        """YOUR MAIN LOGIC HERE"""
        # Extract inputs
        rgb = obs[self.rgb_key]
        depth = obs[self.depth_key]
        
        # Process
        pcl = self._my_processing(rgb, depth)
        
        # Add output
        obs["pcl"] = pcl
        return obs
    
    def transform_features(self, features):
        """DECLARE OUTPUT SHAPE"""
        obs_features = features[PipelineFeatureType.OBSERVATION].copy()
        obs_features["pcl"] = PolicyFeature(
            shape=(self.target_num_points, 3),
            dtype="float32",
        )
        features[PipelineFeatureType.OBSERVATION] = obs_features
        return features
```

---

## Pipeline Container Classes

### RobotProcessorPipeline (Lines 286-1136)

```python
class RobotProcessorPipeline(Generic[InputType, OutputType]):
    """Chain multiple ProcessorStep instances together.
    
    Key features:
    - Takes a list of ProcessorStep instances
    - Runs them sequentially on an EnvTransition
    - Converts input → transition → output using converter functions
    - Can save/load state from all steps
    
    ⭐ This is what you use in factory.py to chain your processors!
    """
    
    def __init__(
        self,
        steps: list[ProcessorStep],
        to_transition: Callable[[InputType], EnvTransition],
        to_output: Callable[[EnvTransition], OutputType],
    ):
        self.steps = steps
        self.to_transition = to_transition
        self.to_output = to_output
    
    def __call__(self, data: InputType) -> OutputType:
        """Run the pipeline:
        1. Convert input to EnvTransition
        2. Run all steps sequentially
        3. Convert back to output type
        """
        transition = self.to_transition(data)
        
        for step in self.steps:
            transition = step(transition)  # Each step processes the transition
        
        return self.to_output(transition)
```

**Usage Example:**
```python
pipeline = RobotProcessorPipeline[RobotObservation, RobotObservation](
    steps=[
        CleanupStep(),           # Step 1
        PointCloudStep(),        # Step 2
        MeshGATStep(),           # Step 3
    ],
    to_transition=observation_to_transition,  # RobotObs → EnvTransition
    to_output=transition_to_observation,      # EnvTransition → RobotObs
)

raw_obs = {"rgb": ..., "depth": ...}
processed_obs = pipeline(raw_obs)  # Runs all 3 steps
```

---

## ProcessorStepRegistry (Lines 80-141)

```python
class ProcessorStepRegistry:
    """Global registry for processor steps.
    
    Allows steps to be:
    1. Registered by name
    2. Instantiated from config
    3. Saved/loaded with checkpoints
    """
    
    _registry: dict[str, type[ProcessorStep]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a processor class.
        
        Usage:
            @ProcessorStepRegistry.register("my_processor")
            @dataclass
            class MyProcessor(ObservationProcessorStep):
                ...
        """
        def decorator(processor_cls):
            cls._registry[name] = processor_cls
            return processor_cls
        return decorator
    
    @classmethod
    def from_config(cls, name: str, config: dict) -> ProcessorStep:
        """Create a processor instance from config."""
        processor_cls = cls._registry[name]
        return processor_cls(**config)
```

**Why This Matters:**
- Registered processors can be instantiated from YAML configs
- Makes processors discoverable by the framework
- Required for save/load functionality

---

## Data Types (Lines 33-78)

```python
# EnvTransition: The universal data format all processors work with
EnvTransition = dict[str, Any]  # Actually a TypedDict with these keys:

class TransitionKey:
    """Keys in an EnvTransition dict"""
    OBSERVATION = "observation"      # dict[str, np.ndarray | torch.Tensor]
    ACTION = "action"                # dict[str, np.ndarray | torch.Tensor]
    REWARD = "reward"                # float
    DONE = "done"                    # bool
    TRUNCATED = "truncated"          # bool
    NEXT_OBSERVATION = "next.observation"
    # ... etc


# Robot-specific types (from processor/core.py)
RobotObservation = dict[str, np.ndarray]   # e.g., {"rgb": array, "joint_pos": array}
RobotAction = dict[str, np.ndarray]        # e.g., {"joint_target": array}

# Policy types (training)
PolicyAction = dict[str, torch.Tensor]     # Batched, normalized tensors
PolicyBatch = dict[str, torch.Tensor]      # Complete batch for training
```

---

## Converter Functions (from processor/converters.py)

```python
def observation_to_transition(obs: RobotObservation) -> EnvTransition:
    """Convert robot observation dict to EnvTransition format.
    
    Used by RobotProcessorPipeline as to_transition() argument.
    """
    return {
        TransitionKey.OBSERVATION: obs,
        # Other keys initialized to None/default
    }


def transition_to_observation(tr: EnvTransition) -> RobotObservation:
    """Extract observation from EnvTransition.
    
    Used by RobotProcessorPipeline as to_output() argument.
    """
    return tr[TransitionKey.OBSERVATION]
```

---

## What You DON'T Need to Understand (Can Ignore)

### PolicyProcessorPipeline (Lines 1138-1437)
- Used during **training**, not teleoperation
- Converts transitions → batched tensors
- Handles normalization, device placement, temporal chunking
- **You don't use this for robot observation processing**

### Other Specialized Steps
- `RewardProcessorStep` - Process rewards
- `ComplementaryDataProcessorStep` - Process metadata
- These are for specific use cases, not general observation processing

---

## Mental Model: How It All Fits Together

```
┌─────────────────────────────────────────────────────────────────┐
│                    Your Code                                     │
│                                                                  │
│  @ProcessorStepRegistry.register("my_processor")                │
│  @dataclass                                                      │
│  class MyProcessor(ObservationProcessorStep):  ← You write this │
│      def observation(self, obs): ...           ← Implement this │
│      def transform_features(self, f): ...      ← Implement this │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Inherits from
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ObservationProcessorStep (ABC)                      │
│              - Provides __call__() implementation                │
│              - You just implement observation()                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Inherits from
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ProcessorStep (ABC)                            │
│                   - Base interface                               │
│                   - Defines abstract methods                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              │ Used in
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              RobotProcessorPipeline                              │
│              - Chains your processors together                   │
│              - Handles conversions                               │
│                                                                  │
│  pipeline = RobotProcessorPipeline(                             │
│      steps=[MyProcessor(), OtherProcessor()],  ← Your instances │
│      to_transition=...,                                         │
│      to_output=...,                                             │
│  )                                                              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              │ Used in
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   factory.py                                     │
│                                                                  │
│  def make_my_robot_observation_processor():                     │
│      return RobotProcessorPipeline(                             │
│          steps=[MyProcessor(), ...],  ← Build pipeline here     │
│          ...                                                     │
│      )                                                          │
│                                                                  │
│  def make_teleop_robot_processors():                            │
│      if robot.type == "my_robot":                               │
│          return (..., my_pipeline)  ← Wire into teleop here     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              │ Used by
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           lerobot-teleoperate / lerobot-record                   │
│           - Main scripts that run teleop                         │
│           - Use your pipeline to process observations            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Decision Tree: "Which Class Should I Use?"

```
START: What are you processing?
│
├─ Observations (sensors, cameras, state)
│  └─ Use: ObservationProcessorStep
│     - Implement: observation(obs) → obs
│     - Example: FabricPointCloudProcessorStep
│
├─ Actions (motor commands, control signals)
│  └─ Use: ActionProcessorStep
│     - Implement: action(act) → act
│     - Example: DeltaActionProcessor
│
├─ Multiple steps together
│  └─ Use: RobotProcessorPipeline
│     - Pass: list of ProcessorStep instances
│     - Example: [CleanupStep(), PointCloudStep(), MeshGATStep()]
│
└─ Need to register for config/save/load
   └─ Use: @ProcessorStepRegistry.register("name")
      - Decorator for your class
      - Makes it discoverable by framework
```

---

## Common Patterns from Real Processors

### Pattern 1: Simple Passthrough (No Changes)
```python
class IdentityProcessorStep(ProcessorStep):
    def __call__(self, transition):
        return transition  # Do nothing
    
    def transform_features(self, features):
        return features  # No schema changes
```

### Pattern 2: Add New Observation Field
```python
class AddDerivedFieldStep(ObservationProcessorStep):
    def observation(self, obs):
        obs["velocity"] = compute_velocity(obs["position"])
        return obs
    
    def transform_features(self, features):
        obs_features = features[PipelineFeatureType.OBSERVATION].copy()
        obs_features["velocity"] = PolicyFeature(shape=(3,), dtype="float32")
        features[PipelineFeatureType.OBSERVATION] = obs_features
        return features
```

### Pattern 3: Transform Existing Field
```python
class NormalizeStep(ObservationProcessorStep):
    def observation(self, obs):
        obs["joint_pos"] = (obs["joint_pos"] - mean) / std
        return obs
    
    def transform_features(self, features):
        return features  # Shape unchanged, just values transformed
```

### Pattern 4: Lazy Initialization (Heavy Resources)
```python
class HeavyModelStep(ObservationProcessorStep):
    def __init__(self, model_path):
        self.model_path = model_path
        self._model = None  # Don't load yet!
    
    def observation(self, obs):
        if self._model is None:
            self._model = load_model(self.model_path)  # Load on first use
        
        obs["prediction"] = self._model(obs["input"])
        return obs
```

---

## Summary: The 3 Things You Actually Use

1. **ObservationProcessorStep** - Your parent class
2. **RobotProcessorPipeline** - To chain processors
3. **@ProcessorStepRegistry.register()** - To make it discoverable

Everything else in pipeline.py is either:
- Implementation details (ignore)
- Training-related (ignore for now)
- Other specialized use cases (ignore until needed)

**Focus on these 3, and you'll be 90% effective.**
