#!/usr/bin/env python

"""Processor step for running MeshGAT on robot observations.

This step is meant to be plugged into a RobotProcessorPipeline[RobotObservation, RobotObservation]
(as one of the steps in the robot_observation_processor).

It uses the external MeshGAT repository (added as a git submodule under
`external/mesh_gat`) via its public API `load_meshgat_model`.

Contract (current design):
- We support a *single* input key in the observation dict.
- The MeshGAT config's `input_type` field determines whether that input
  is interpreted as a depth image or a point cloud.
- Observations are assumed to be numpy arrays; the processor converts
  numpy -> torch -> numpy for inference.
- The predicted mesh vertices are stored under `output_key` (default:
  "mesh_vertices").

This file only defines the step and does not yet wire it into any
factory; adding it to `make_default_robot_observation_processor` or
other robot-specific factories can be done separately to avoid
impacting existing setups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import os
import sys

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry


# We'll dynamically import load_meshgat_model in _lazy_init_model to handle path setup
load_meshgat_model = None  # type: ignore[assignment]


@dataclass
@ProcessorStepRegistry.register("meshgat_observation_processor")
class MeshGATObservationProcessorStep(ObservationProcessorStep):
    """Run MeshGAT on a single observation and attach predicted mesh vertices.

    Parameters
    ----------
    checkpoint_path: str
        Path to the MeshGAT checkpoint (.pt).
    config_path: str
        Path to the MeshGAT config YAML (compatible with the checkpoint).
    template_path: str, optional
        Optional explicit template pickle path. If not provided, the
        MeshGAT config resolves it as usual.
    device: str, optional
        Device string for the MeshGAT model (e.g. "cuda" or "cpu").
    input_key: str
        Observation key to read as input for MeshGAT. The underlying
        MeshGAT config decides whether this represents depth or
        pointcloud via its `input_type` field.
    output_key: str, optional
        Observation key under which to store the predicted mesh
        vertices. Defaults to "mesh_vertices".
    """

    checkpoint_path: str
    config_path: str
    input_key: str
    template_path: Optional[str] = None
    device: str = "cuda"
    output_key: str = "mesh_vertices"

    # Internal cache (not part of the external config)
    _model: Optional[torch.nn.Module] = field(default=None, init=False, repr=False)
    _cfg: Optional[Any] = field(default=None, init=False, repr=False)

    def _lazy_init_model(self) -> None:
        """Load MeshGAT model on first use.

        Raises
        ------
        RuntimeError
            If the MeshGAT API cannot be imported.
        """

        if self._model is not None:
            return

        # Dynamically import load_meshgat_model with path setup
        try:
            # Add external/mesh_gat to path so its internal imports work
            _mesh_gat_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "external", "mesh_gat")
            _mesh_gat_dir = os.path.abspath(_mesh_gat_dir)
            if os.path.exists(_mesh_gat_dir) and _mesh_gat_dir not in sys.path:
                sys.path.insert(0, _mesh_gat_dir)
            
            from api import load_meshgat_model  # Import from mesh_gat directory
        except ImportError as e:
            raise RuntimeError(
                f"MeshGATObservationProcessorStep requires the `external/mesh_gat` "
                f"submodule to be available and importable. Import error: {e}"
            )

        model, cfg = load_meshgat_model(
            checkpoint_path=self.checkpoint_path,
            config_path=self.config_path,
            device=self.device,
            template_path=self.template_path,
        )
        self._model = model
        self._cfg = cfg

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Adds the mesh vertices output feature to the observation features.

        This step adds a new observation key (output_key) with shape (num_vertices, 3).
        The exact number of vertices is unknown at transform_features time, so we use
        None as a placeholder (variable-length).
        """
        # Add the mesh_vertices as a new observation feature
        if PipelineFeatureType.OBSERVATION in features:
            obs_features = features[PipelineFeatureType.OBSERVATION].copy()
            # Define the mesh_vertices feature (shape is model-dependent, use None)
            obs_features[self.output_key] = PolicyFeature(
                shape=(None, 3),  # Variable number of vertices
                dtype="float32",
            )
            features = features.copy()
            features[PipelineFeatureType.OBSERVATION] = obs_features
        return features

    def _prepare_input_tensor(self, observation: dict[str, Any]) -> Any:
        """Extract and convert the observation field for MeshGAT.

        Returns a torch tensor or dict suitable for passing directly
        to the MeshGAT model, depending on `cfg.input_type`.
        """

        if self._cfg is None:
            raise RuntimeError("MeshGAT model config is not initialized.")

        if self.input_key not in observation:
            raise KeyError(
                f"MeshGATObservationProcessorStep expected key '{self.input_key}' "
                f"in observation, but it is missing."
            )

        raw_value = observation[self.input_key]

        # Ensure numpy array; many robot observations store images/points as numpy.
        if isinstance(raw_value, torch.Tensor):
            arr = raw_value.detach().cpu().numpy()
        else:
            arr = np.asarray(raw_value)

        input_type = getattr(self._cfg, "input_type", "depth")

        if input_type == "pointcloud":
            # Expect shape (N, 3) or (B, N, 3). Convert to torch and add
            # batch dim if necessary.
            if arr.ndim == 2:
                # (N, 3) -> (1, N, 3)
                arr = arr[None, ...]
            elif arr.ndim != 3:
                raise ValueError(
                    "MeshGATObservationProcessorStep expected pointcloud with "
                    "shape (N, 3) or (B, N, 3), got shape " f"{arr.shape}."
                )

            points = torch.from_numpy(arr).float().to(self.device)
            return {"points": points}

        # Default: depth / image mode
        # Expected by MeshGAT as image_batch: (B, C, H, W) or (B, H, W)
        if arr.ndim == 2:
            # (H, W) -> (1, 1, H, W)
            arr = arr[None, None, ...]
        elif arr.ndim == 3:
            # (H, W, C) or (C, H, W)
            if arr.shape[0] in (1, 3):
                # assume (C, H, W) -> (1, C, H, W)
                arr = arr[None, ...]
            else:
                # assume (H, W, C) -> (1, C, H, W)
                arr = np.transpose(arr, (2, 0, 1))[None, ...]
        elif arr.ndim == 4:
            # assume already batched (B, C, H, W)
            pass
        else:
            raise ValueError(
                "MeshGATObservationProcessorStep expected depth/image with "
                "2, 3, or 4 dims, got shape " f"{arr.shape}."
            )

        image = torch.from_numpy(arr).float().to(self.device)
        return image

    def _run_model(self, model_input: Any) -> np.ndarray:
        """Run MeshGAT model and return vertices as a numpy array (V, 3)."""

        if self._model is None:
            raise RuntimeError("MeshGAT model is not initialized.")

        self._model.eval()
        with torch.no_grad():
            pred = self._model(model_input)

        if isinstance(pred, dict):  # in case future versions return dicts
            if "vertices" in pred:
                pred_vertices = pred["vertices"]
            else:
                # Fallback: try common keys or raise
                for key in ("mesh", "mesh_vertices", "pred_mesh"):
                    if key in pred:
                        pred_vertices = pred[key]
                        break
                else:
                    raise KeyError(
                        "MeshGAT model returned a dict without 'vertices' or "
                        "a known mesh key (mesh, mesh_vertices, pred_mesh)."
                    )
        else:
            pred_vertices = pred

        if not isinstance(pred_vertices, torch.Tensor):
            pred_vertices = torch.as_tensor(pred_vertices)

        # Expect (B, V, 3); squeeze batch.
        if pred_vertices.ndim == 3:
            pred_vertices = pred_vertices[0]
        elif pred_vertices.ndim != 2:
            raise ValueError(
                "MeshGAT model output must have shape (B, V, 3) or (V, 3); "
                f"got shape {tuple(pred_vertices.shape)}."
            )

        return pred_vertices.detach().cpu().numpy()

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Process a single robot observation.

        This method:
        - lazily loads the MeshGAT model on first call;
        - extracts the observation at `input_key`;
        - converts it to the appropriate tensor representation;
        - runs MeshGAT;
        - stores the predicted vertices under `output_key` as a numpy
          array with shape (V, 3).
        """

        self._lazy_init_model()

        model_input = self._prepare_input_tensor(observation)
        vertices = self._run_model(model_input)

        observation[self.output_key] = vertices
        return observation

    # Optional: expose config for serialization / hub upload
    def get_config(self) -> dict[str, Any]:  # type: ignore[override]
        return {
            "checkpoint_path": self.checkpoint_path,
            "config_path": self.config_path,
            "template_path": self.template_path,
            "device": self.device,
            "input_key": self.input_key,
            "output_key": self.output_key,
        }
