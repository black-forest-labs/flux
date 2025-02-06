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
from colored import fore, style

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
    trt_update_output_names: list[str] | None = None
    trt_enable_all_tactics: bool = False
    trt_timing_cache: str | None = None
    trt_native_instancenorm: bool = True
    trt_builder_optimization_level: int = 3
    trt_precision_constraints: str = "none"

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

    @staticmethod
    def build(
        engine_path: str,
        onnx_path: str,
        strongly_typed=False,
        tf32=True,
        bf16=False,
        fp8=False,
        fp4=False,
        input_profile: dict[str, Any] | None = None,
        update_output_names: list[str] | None = None,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        native_instancenorm=True,
        builder_optimization_level=3,
        precision_constraints="none",
        verbose=False,
    ):
        print(f"Building TensorRT engine for {onnx_path}: {engine_path}")

        # Base command
        build_command = [f"polygraphy convert {onnx_path} --convert-to trt --output {engine_path}"]

        # Precision flags
        build_args = [
            "--bf16" if bf16 else "",
            "--tf32" if tf32 else "",
            "--fp8" if fp8 else "",
            "--fp4" if fp4 else "",
            "--strongly-typed" if strongly_typed else "",
        ]

        # Additional arguments
        build_args.extend(
            [
                "--refittable" if enable_refit else "",
                "--tactic-sources" if not enable_all_tactics else "",
                "--onnx-flags native_instancenorm" if native_instancenorm else "",
                f"--builder-optimization-level {builder_optimization_level}",
                f"--precision-constraints {precision_constraints}",
            ]
        )

        # Timing cache
        if timing_cache:
            build_args.extend([f"--load-timing-cache {timing_cache}", f"--save-timing-cache {timing_cache}"])

        # Verbosity setting
        verbosity = "extra_verbose" if verbose else "error"
        build_args.append(f"--verbosity {verbosity}")

        # Output names
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            build_args.append(f"--trt-outputs {' '.join(update_output_names)}")

        # Input profiles
        if input_profile:
            profile_args = defaultdict(str)
            for name, dims in input_profile.items():
                assert len(dims) == 3
                profile_args["--trt-min-shapes"] += f"{name}:{str(list(dims[0])).replace(' ', '')} "
                profile_args["--trt-opt-shapes"] += f"{name}:{str(list(dims[1])).replace(' ', '')} "
                profile_args["--trt-max-shapes"] += f"{name}:{str(list(dims[2])).replace(' ', '')} "

            build_args.extend(f"{k} {v}" for k, v in profile_args.items())

        # Filter out empty strings and join command
        build_args = [arg for arg in build_args if arg]
        final_command = " \\\n".join(build_command + build_args)

        # Execute command with improved error handling
        try:
            print(f"Engine build command:{fore('yellow')}\n{final_command}\n{style('reset')}")
            subprocess.run(final_command, check=True, shell=True)
        except subprocess.CalledProcessError as exc:
            error_msg = f"Failed to build TensorRT engine. Error details:\nCommand: {exc.cmd}\n"
            raise RuntimeError(error_msg) from exc
