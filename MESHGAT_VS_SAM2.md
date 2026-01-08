# MeshGAT vs SAM2 Integration: Key Differences

## Quick Answer

**Should you make SAM2 a processor like MeshGAT?**

**No!** Here's why:

```
MeshGAT = Standalone Processor Step
SAM2    = Callable Dependency (wrapped function)
```

---

## Visual Comparison

### MeshGAT Architecture (Standalone Processor)

```
Pipeline: [Step1] → [Step2: MeshGAT] → [Step3]

MeshGATObservationProcessorStep:
  - Registered: @ProcessorStepRegistry.register("meshgat_observation_processor")
  - Input: observation["pcl"]
  - Output: observation["mesh_vertices"]
  - Interface: observation(obs) → obs
  - Used as: pipeline step
```

### SAM2 Architecture (Callable Dependency)

```
Pipeline: [Step1] → [Step2: FabricPointCloud (uses SAM2)] → [Step3]

FabricPointCloudProcessorStep:
  - Registered: @ProcessorStepRegistry.register("fabric_pointcloud_processor")
  - Constructor param: sam_runner (callable)
  - Inside observation():
      mask = self.sam_runner(rgb)  ← SAM2 called here
      pcl = depth_to_pointcloud(depth, mask)
      return obs with "pcl"
```

---

## Code Comparison

### MeshGAT (Processor Approach)

```python
# File: src/lerobot/processor/mesh_gat_processor.py

@ProcessorStepRegistry.register("meshgat_observation_processor")
@dataclass
class MeshGATObservationProcessorStep(ObservationProcessorStep):
    checkpoint_path: str
    config_path: str
    input_key: str = "pcl"
    output_key: str = "mesh_vertices"
    
    def observation(self, obs):
        # Load model (lazy)
        if self._model is None:
            self._model = load_model(...)
        
        # Run inference
        vertices = self._model(obs[self.input_key])
        obs[self.output_key] = vertices
        return obs

# Usage in factory.py:
steps = [
    ...,
    MeshGATObservationProcessorStep(
        checkpoint_path="path/to/checkpoint.pt",
        config_path="path/to/config.yaml",
    ),
]
```

### SAM2 (Callable Approach) ✅ RECOMMENDED

```python
# File: external/sam2/api.py (helper functions, NOT a processor)

def create_simple_sam2_runner(checkpoint, box_prompt):
    predictor = load_sam2_predictor(checkpoint)
    
    def sam_runner(rgb):
        predictor.set_image(rgb)
        mask, _, _ = predictor.predict(box=box_prompt)
        return mask.astype(bool)
    
    return sam_runner

# Usage in factory.py:
from external.sam2.api import create_simple_sam2_runner

# Create the callable
sam_runner = create_simple_sam2_runner(
    checkpoint="path/to/sam2.pt",
    box_prompt=[100, 100, 400, 400],
)

# Pass to FabricPointCloud as parameter
steps = [
    ...,
    FabricPointCloudProcessorStep(
        ...,
        sam_runner=sam_runner,  # ← Just a function, not a processor
    ),
]
```

---

## Why the Difference?

| Aspect | MeshGAT | SAM2 |
|--------|---------|------|
| **Conceptual Role** | Mesh prediction is a separate, independent processing step | Segmentation is part of pointcloud extraction |
| **Input** | Pointcloud (already processed data) | Raw RGB image |
| **Output** | Mesh vertices (new observation field) | Binary mask (intermediate, not saved to observation) |
| **Reusability** | Can be used independently with any pointcloud source | Tightly coupled to FabricPointCloud's workflow |
| **Serialization** | Needs to be saved/loaded with pipeline | Recreated each time from config |
| **Flexibility** | MeshGAT is the only option for mesh prediction | User might want to swap SAM2 for other segmentation |

---

## When to Use Each Pattern

### Use Processor Pattern (like MeshGAT) when:
✅ Processing step is independent and reusable
✅ Input/output are both observation fields
✅ Step can be used in different pipelines
✅ Step needs to be serialized/deserialized
✅ Step is the "main" functionality

**Examples:**
- MeshGAT (pointcloud → mesh)
- Normalization (state → normalized state)
- Device transfer (CPU tensors → GPU tensors)

### Use Callable Pattern (like SAM2) when:
✅ Functionality is a dependency of another processor
✅ Output is intermediate (not saved to observation)
✅ User might want to swap implementations
✅ Keeps main processor flexible
✅ Doesn't need separate serialization

**Examples:**
- SAM2 (RGB → mask) used by FabricPointCloud
- Image augmentation functions used by ImageProcessor
- Noise functions used by DomainRandomization

---

## Real-World Analogy

Think of it like building a car:

### MeshGAT = Complete Car Part (Processor)
```
[Engine] → [Transmission: MeshGAT] → [Wheels]
         (standalone, swappable component)
```

You can remove MeshGAT and replace it with a different mesh predictor. It's a complete, self-contained component.

### SAM2 = Tool Used During Manufacturing (Callable)
```
[Raw Materials] → [Factory: FabricPointCloud] → [Product]
                      ↳ uses SAM2 tool internally
```

SAM2 is a tool the factory uses, not a product itself. The factory (FabricPointCloud) could use a different tool (different segmentation method) without changing the factory's interface.

---

## Implementation Checklist

### ✅ What You Already Have:

**MeshGAT (Processor Pattern):**
- ✅ `external/mesh_gat/api.py` - model loading API
- ✅ `src/lerobot/processor/mesh_gat_processor.py` - processor class
- ✅ Registered with `@ProcessorStepRegistry.register()`
- ✅ Implements `observation()` and `transform_features()`

**FabricPointCloud (Processor with SAM2 dependency):**
- ✅ `src/lerobot/processor/fabric_pointcloud_processor.py` - processor class
- ✅ Accepts `sam_runner: Callable` parameter
- ✅ Registered and tested

### ⏳ What You Need to Add:

**SAM2 (Callable Pattern):**
- ✅ `external/sam2/api.py` - helper functions (created above)
- ⏳ Add SAM2 as git submodule
- ⏳ Download checkpoint
- ⏳ Test with simple script
- ⏳ Create sam_runner in factory function
- ⏳ Wire into pipeline

---

## Example: Complete Pipeline

```python
# File: src/lerobot/processor/factory.py

def make_denso_meshgat_robot_observation_processor(
    camera_intrinsics,
    depth_scale,
    sam2_checkpoint,
    sam2_box_prompt,
    meshgat_checkpoint,
    meshgat_config,
):
    # Import SAM2 helpers (NOT a processor!)
    from external.sam2.api import create_simple_sam2_runner
    
    # Create SAM2 runner as a simple callable
    sam_runner = create_simple_sam2_runner(
        checkpoint_path=sam2_checkpoint,
        box_prompt=sam2_box_prompt,
    )
    
    # Build pipeline with 3 processors
    steps = [
        # Processor 1: Cleanup
        DensoDeltaPoseStripRemoteActionStep(),
        
        # Processor 2: Pointcloud (uses SAM2 internally)
        FabricPointCloudProcessorStep(
            rgb_key="rgb",
            depth_key="depth",
            fx=camera_intrinsics[0, 0],
            fy=camera_intrinsics[1, 1],
            cx=camera_intrinsics[0, 2],
            cy=camera_intrinsics[1, 2],
            depth_scale=depth_scale,
            sam_runner=sam_runner,  # ← SAM2 wrapped as callable
        ),
        
        # Processor 3: Mesh prediction
        MeshGATObservationProcessorStep(
            checkpoint_path=meshgat_checkpoint,
            config_path=meshgat_config,
            input_key="pcl",
            output_key="mesh_vertices",
        ),
    ]
    
    return RobotProcessorPipeline(
        steps=steps,
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
```

---

## Data Flow Visualization

```
Raw Observation from Robot:
{
    "rgb": (480, 640, 3) uint8,
    "depth": (480, 640) uint16,
    "joint_pos": (6,) float32,
}
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Step 1: DensoDeltaPoseStripRemoteActionStep     │
│   Removes internal fields                       │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ Step 2: FabricPointCloudProcessorStep           │
│                                                  │
│   1. Extract rgb and depth                      │
│   2. Call sam_runner(rgb)  ← SAM2 HERE         │
│      → mask: (480, 640) bool                    │
│   3. Project masked depth to 3D                 │
│   4. Center and resample                        │
│   5. Add obs["pcl"] = (1024, 3) float32        │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│ Step 3: MeshGATObservationProcessorStep         │
│                                                  │
│   1. Extract obs["pcl"]                         │
│   2. Run MeshGAT model                          │
│   3. Add obs["mesh_vertices"] = (N, 3) float32 │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
Final Observation:
{
    "rgb": (480, 640, 3) uint8,
    "depth": (480, 640) uint16,
    "joint_pos": (6,) float32,
    "pcl": (1024, 3) float32,           ← Added by FabricPointCloud
    "mesh_vertices": (N, 3) float32,    ← Added by MeshGAT
}
```

Notice: The mask from SAM2 is **not** in the final observation. It's internal to FabricPointCloud.

---

## Common Mistakes to Avoid

❌ **Don't do this:**
```python
# DON'T make SAM2 a separate processor step
@ProcessorStepRegistry.register("sam2_processor")  # ❌ NO!
class SAM2Processor(ObservationProcessorStep):
    def observation(self, obs):
        obs["mask"] = run_sam2(obs["rgb"])
        return obs

steps = [
    SAM2Processor(),           # ❌ Separate step
    FabricPointCloudProcessor(),  # Now needs to read obs["mask"]
    MeshGATProcessor(),
]
```

**Problems:**
- Tight coupling: FabricPointCloud expects mask in observation
- Serialization: mask is intermediate data, shouldn't be saved
- Flexibility: hard to swap segmentation methods
- Extra step: unnecessary indirection

✅ **Do this instead:**
```python
# Create SAM2 as callable
sam_runner = create_simple_sam2_runner(...)

# Pass to FabricPointCloud
steps = [
    FabricPointCloudProcessor(sam_runner=sam_runner),  # ✅ Clean!
    MeshGATProcessor(),
]
```

**Benefits:**
- Loose coupling: sam_runner is just a function
- No serialization: recreated from config
- Flexible: easy to swap `sam_runner` implementation
- Cleaner: one less processor step

---

## Summary

**MeshGAT → Processor Pattern:**
- It's a standalone, independent processing step
- Input and output are both observation fields
- Needs to be serialized and registered

**SAM2 → Callable Pattern:**
- It's a dependency used inside FabricPointCloud
- Output (mask) is intermediate, not saved
- Just a function, no registration needed

**Both are correct patterns for their use cases!**

Your integration is already well-designed. Now just follow the steps in `SAM2_INTEGRATION.md` to add the SAM2 submodule and create the sam_runner in your factory function.
