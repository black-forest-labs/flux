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
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from colored import fore, style
from huggingface_hub import snapshot_download
from tensorrt import __version__ as trt_version


class ModuleName(Enum):
    CLIP = "clip"
    T5 = "t5"
    TRANSFORMER = "transformer"
    VAE = "vae"
    VAE_ENCODER = "vae_encoder"


registry = {}


@dataclass
class TRTBaseConfig:
    engine_dir: str
    precision: str
    trt_verbose: bool
    trt_static_batch: bool
    trt_static_shape: bool
    model_name: str
    module_name: ModuleName
    onnx_path: str = field(init=False)
    engine_path: str = field(init=False)
    trt_tf32: bool
    trt_bf16: bool
    trt_fp8: bool
    trt_fp4: bool
    trt_build_strongly_typed: bool
    custom_onnx_path: str | None = None
    trt_update_output_names: list[str] | None = None
    trt_enable_all_tactics: bool = False
    trt_timing_cache: str | None = None
    trt_native_instancenorm: bool = True
    trt_builder_optimization_level: int = 3
    trt_precision_constraints: str = "none"

    min_batch: int = 1
    max_batch: int = 4

    @staticmethod
    def build_trt_engine(
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
        timing_cache: str | None = None,
        native_instancenorm=True,
        builder_optimization_level=3,
        precision_constraints="none",
        verbose=False,
    ):
        """
        Metod used to build a TRT engine from a given set of flags or configurations using polygraphy.

        Args:
            engine_path (str): Output path used to store the build engine.
            onnx_path (str): Path containing an onnx model used to generated the engine.
            strongly_typed (bool): Flag indicating if the engine should be strongly typed.
            tf32 (bool): Whether to build the engine with TF32 precision enabled.
            bf16 (bool): Whether to build the engine with BF16 precision enabled.
            fp8 (bool): Whether to build the engine with FP8 precision enabled. Refer to plain dataype and do not interfer with quantization introduced by modelopt.
            fp4 (bool): Whether to build the engine with FP4 precision enabled. Refer to plain dataype and do not interfer with quantization introduced by modelopt.
            input_profile (dict[str, Any]): A set of optimization profiles to add to the configuration. Only needed for networks with dynamic input shapes.
            update_output_names (list[str]): List of output names to use in the trt engines.
            enable_refit (bool): Enables the engine to be refitted with new weights after it is built.
            enable_all_tactics (bool): Enables TRT to leverage all tactics or not.
            timing_cache (str): A path or file-like object from which to load a tactic timing cache.
            native_instancenorm (bool): support of instancenorm plugin.
            builder_optimization_level (int): The builder optimization level.
            precision_constraints (str):  If set to "obey", require that layers execute in specified precisions. If set to "prefer", prefer that layers execute in specified precisions but allow TRT to fall back to other precisions if no implementation exists for the requested precision. Otherwise, precision constraints are ignored.
            verbose (bool): Weather to support verbose output

        Returns:
            dict[str, Any]: A dictionary representing the input profile configuration.
        """
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

    @classmethod
    @abstractmethod
    def from_args(cls, model_name: str, *args, **kwargs) -> Any:
        raise NotImplementedError("Factory method is missing")

    @abstractmethod
    def get_input_profile(
        self,
        batch_size: int,
        image_height: int | None,
        image_width: int | None,
    ) -> dict[str, Any]:
        """
        Generate max and min shape that each input of a TRT engine can have.

        Subclasses must implement this method to return a dictionary that defines
        the input profile based on the provided parameters. The input profile typically
        includes details such as the expected shape of input tensors, whether the batch size
        or image dimensions are fixed, and any additional configuration required by the
        data processing or model inference pipeline.

        Args:
            batch_size (int): The number of images per batch.
            image_height (int): Default height of each image in pixels.
            image_width (int): Defailt width of each image in pixels.
            static_batch (bool): Flag indicating if the batch size is fixed (static).
            static_shape (bool): Flag indicating if the image dimensions are fixed (static).

        Returns:
            dict[str, Any]: A dictionary representing the input profile configuration.

        Raises:
            NotImplementedError: If the subclass does not override this abstract method.
        """
        pass

    @abstractmethod
    def check_dims(self, *args, **kwargs) -> None | tuple[int, int] | int:
        """helper function that check the dimentions associated to each input of a TRT engine"""
        pass

    def _check_batch(self, batch_size):
        assert (
            self.min_batch <= batch_size <= self.max_batch
        ), f"Batch size {batch_size} must be between {self.min_batch} and {self.max_batch}"

    def __post_init__(self):
        self.onnx_path = self._get_onnx_path()
        self.engine_path = self._get_engine_path()
        assert os.path.isfile(self.onnx_path), "onnx_path do not exists: {}".format(self.onnx_path)

    def _get_onnx_path(self) -> str:
        if self.custom_onnx_path:
            return self.custom_onnx_path

        repo_id = self._get_repo_id(self.model_name)
        snapshot_path = snapshot_download(repo_id, allow_patterns=[f"{self.module_name.value}.opt/*"])
        onnx_model_path = os.path.join(snapshot_path, f"{self.module_name.value}.opt/model.onnx")
        return onnx_model_path

    def _get_engine_path(self) -> str:
        return os.path.join(
            self.engine_dir,
            self.model_name,
            f"{self.module_name.value}_{self.precision}.trt_{trt_version}.plan",
        )

    @staticmethod
    def _get_repo_id(model_name: str) -> str:
        if model_name == "flux-dev":
            return "black-forest-labs/FLUX.1-dev-onnx"
        elif model_name == "flux-schnell":
            return "black-forest-labs/FLUX.1-schnell-onnx"
        elif model_name == "flux-dev-canny":
            return "black-forest-labs/FLUX.1-Canny-dev-onnx"
        elif model_name == "flux-dev-depth":
            return "black-forest-labs/FLUX.1-Depth-dev-onnx"
        elif model_name == "flux-dev-kontext":
            return "black-forest-labs/FLUX.1-Kontext-dev-onnx"
        else:
            raise ValueError(f"Unknown model name: {model_name}")


def register_config(module_name: ModuleName, precision: str):
    """Decorator to register a configuration class with specific flag conditions."""

    def decorator(cls):
        key = f"module={module_name.value}_dtype={precision}"
        registry[key] = cls
        return cls

    return decorator


def get_config(module_name: ModuleName, precision: str) -> TRTBaseConfig:
    """Retrieve the appropriate configuration instance based on current flags."""
    key = f"module={module_name.value}_dtype={precision}"
    return registry[key]
