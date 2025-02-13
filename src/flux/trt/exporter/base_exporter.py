#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from tensorrt import __version__ as trt_version

registry = {}


@dataclass
class TRTBaseConfig:
    onnx_dir: str
    engine_dir: str
    precision: str
    trt_verbose: bool
    trt_static_batch: bool
    trt_static_shape: bool
    model_name: str
    onnx_path: str = field(init=False)
    engine_path: str = field(init=False)
    min_batch: int = 1
    max_batch: int = 4
    trt_update_output_names: list[str] | None = None
    trt_enable_all_tactics: bool = False
    trt_timing_cache: str | None = None
    trt_native_instancenorm: bool = True
    trt_builder_optimization_level: int = 3
    trt_precision_constraints: str = "none"

    @classmethod
    @abstractmethod
    def build(cls, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_input_profile(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
        static_batch: bool,
        static_shape: bool,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def check_dims(self, *args, **kwargs) -> None | tuple[int, int]:
        pass

    @abstractmethod
    def get_engine_params(self) -> dict[str, Any]:
        pass

    def __post_init__(self):
        self.onnx_path = self._get_onnx_path()
        self.engine_path = self._get_engine_path()
        assert os.path.isfile(self.onnx_path), "onnx_path do not exists: {}".format(self.onnx_path)

    def _get_onnx_path(self) -> str:
        onnx_model_dir = os.path.join(
            self.onnx_dir,
            self.model_name + ".opt",
        )
        return os.path.join(onnx_model_dir, "model.onnx")

    def _get_engine_path(self) -> str:
        return os.path.join(
            self.engine_dir,
            self.model_name + ".trt" + trt_version + ".plan",
        )

def register_config(model_name: str, tf32=True, bf16=False, fp8=False, fp4=False, t5_fp8=False):
    """Decorator to register a configuration class with specific flag conditions."""

    def decorator(cls):
        if model_name == "t5":
            key = f"model={model_name}_tf32={tf32}_bf16={bf16}_fp8={fp8}_fp4={fp4}_t5-fp8={t5_fp8}"
        else:
            key = f"model={model_name}_tf32={tf32}_bf16={bf16}_fp8={fp8}_fp4={fp4}"
        registry[key] = cls
        return cls

    return decorator


def get_config(model_name: str, tf32=True, bf16=True, fp8=False, fp4=False, t5_fp8=False):
    """Retrieve the appropriate configuration instance based on current flags."""
    if model_name == "t5":
        key = f"model={model_name}_tf32={tf32}_bf16={bf16}_fp8={fp8}_fp4={fp4}_t5-fp8={t5_fp8}"
    else:
        key = f"model={model_name}_tf32={tf32}_bf16={bf16}_fp8={fp8}_fp4={fp4}"
    return registry[key]


class BaseExporter(ABC):
    def __init__(
        self,
        trt_config: TRTBaseConfig,
        max_batch=4,
    ):
        self.min_batch = 1
        self.max_batch = max_batch
        self.trt_config = trt_config

    @abstractmethod
    def get_input_profile(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
        static_batch: bool,
        static_shape: bool,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def check_dims(self, *args, **kwargs) -> None | tuple[int, int]:
        pass
