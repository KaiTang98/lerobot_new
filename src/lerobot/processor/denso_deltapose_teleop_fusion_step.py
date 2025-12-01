#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, Iterable

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep


class DensoDeltaPoseTeleopFusionStep(ProcessorStep):
    """Fuse observation-provided intime fields into the teleop action dict.

    This step copies a whitelist of keys from observation into the action dict so that
    the action_frame written to the dataset includes robot-provided desired signals
    (e.g., desPos/desVel/desVelPure/desFT) alongside teleop-generated deltapose_*.

    Configure with the desired keys via `copy_keys`.
    """

    def __init__(self, copy_keys: Iterable[str] | None = None):
        super().__init__()
        # Default set based on Denso Windows-provided action vectors
        default_keys = [
            # Left arm desired signals
            "desPos_x_A", "desPos_y_A", "desPos_z_A", "desPos_roll_A", "desPos_pitch_A", "desPos_yaw_A",
            "desVel_x_A", "desVel_y_A", "desVel_z_A", "desVel_roll_A", "desVel_pitch_A", "desVel_yaw_A",
            "desVelPure_x_A", "desVelPure_y_A", "desVelPure_z_A", "desVelPure_roll_A", "desVelPure_pitch_A", "desVelPure_yaw_A",
            "desFT_x_A", "desFT_y_A", "desFT_z_A", "desFT_roll_A", "desFT_pitch_A", "desFT_yaw_A",
            # Right arm desired signals
            "desPos_x_B", "desPos_y_B", "desPos_z_B", "desPos_roll_B", "desPos_pitch_B", "desPos_yaw_B",
            "desVel_x_B", "desVel_y_B", "desVel_z_B", "desVel_roll_B", "desVel_pitch_B", "desVel_yaw_B",
            "desVelPure_x_B", "desVelPure_y_B", "desVelPure_z_B", "desVelPure_roll_B", "desVelPure_pitch_B", "desVelPure_yaw_B",
            "desFT_x_B", "desFT_y_B", "desFT_z_B", "desFT_roll_B", "desFT_pitch_B", "desFT_yaw_B",
        ]
        self._copy_keys = list(copy_keys) if copy_keys is not None else default_keys

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        tr = transition.copy()
        action = tr.get(TransitionKey.ACTION) or {}
        obs = tr.get(TransitionKey.OBSERVATION) or {}
        if not isinstance(action, dict) or not isinstance(obs, dict):
            return tr

        cache = None
        if isinstance(obs.get("_last_remote_action"), dict):  # nested cache provided by robot
            cache = obs["_last_remote_action"]

        fused: dict[str, Any] = {**action}
        for k in self._copy_keys:
            # Prefer nested cache; fall back to top-level (in case future robot includes directly)
            if cache and k in cache:
                fused[k] = cache[k]
            elif k in obs:
                fused[k] = obs[k]

        tr[TransitionKey.ACTION] = fused
        return tr

    def transform_features(self, features):
        # # Advertise that ACTION may include additional copied scalar keys.
        # # features comes in as a FeatureSpec mapping; we update the action schema.
        # try:
        #     action_features = features["action"]
        #     for k in self._copy_keys:
        #         # All copied values are scalar floats
        #         action_features[k] = float
        # except Exception:
        #     pass
        return features
