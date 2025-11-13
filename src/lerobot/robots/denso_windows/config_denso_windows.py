#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("denso_windows")
@dataclass(kw_only=True)
class DensoWindowsConfig(RobotConfig):
    """Config for a Denso manipulator proxied via a Windows PC.

    The Linux client connects over TCP to a Windows host that performs the
    low-level robot control and state reporting. This config provides the
    network endpoint and optional cameras attached to the Linux client.
    """

    # Remote Windows server address
    server_ip: str = "192.168.2.100"
    server_port: int = 12345

    # Optional: target loop rate for local timing/teleop assumptions
    fps: int | None = 100

    # Optional local cameras to record alongside robot state
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
