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

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep


class DensoDeltaPoseStripRemoteActionStep(ProcessorStep):
    """Remove the private `_last_remote_action` cache from the observation.

    Raw observation (before this processor) still contains the cache and is passed to
    teleop_action_processor. After this step, the processed observation used for dataset
    frames and logging is clean (state + cameras only).
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        tr = transition.copy()
        obs = tr.get(TransitionKey.OBSERVATION)
        if isinstance(obs, dict) and "_last_remote_action" in obs:
            obs = {k: v for k, v in obs.items() if k != "_last_remote_action"}
            tr[TransitionKey.OBSERVATION] = obs
        return tr

    def transform_features(self, features):
        # No schema change; _last_remote_action was never part of declared observation features.
        return features
