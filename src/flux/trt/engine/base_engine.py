#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gc
import subprocess
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Any

import tensorrt as trt
import torch
from colored import fore, style
from cuda import cudart
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


class BaseEngine(ABC):
    @property
    def trt_to_torch_dtype_dict(self):
        return {
            trt.DataType.BOOL: torch.bool,
            trt.DataType.UINT8: torch.uint8,
            trt.DataType.INT8: torch.int8,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.HALF: torch.float16,
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.BF16: torch.bfloat16,
        }

    @abstractmethod
    def cpu(self) -> "BaseEngine":
        pass

    @abstractmethod
    def to(self, device: str) -> "BaseEngine":
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def activate(
        self,
        device: str | torch.device,
        device_memory: int | None = None,
    ):
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


class Engine(BaseEngine):
    def __init__(
        self,
        engine_path: str,
        stream: cudart.cudaStream_t,
    ):
        self.engine_path = engine_path
        self.stream = stream
        self.engine: trt.ICudaEngine | None = None
        self.context = None
        self.tensors = OrderedDict()
        self.shared_device_memory: int | None = None

    @abstractmethod
    def __call__(self, *args, **Kwargs) -> torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor]:
        pass

    @abstractmethod
    def get_shape_dict(
        self,
        *args,
        **kwargs,
    ) -> dict[str, tuple]:
        pass

    def cpu(self) -> "Engine":
        self.deallocate_buffers()
        self.deactivate()
        self.unload()
        if self.shared_device_memory is not None:
            cudart.cudaFree(self.shared_device_memory)
            self.shared_device_memory = None
        return self

    def to(self, device: str) -> "Engine":
        self.load()
        self.activate(device=device)
        return self

    def load(self):
        if self.engine is not None:
            print(f"[W]: Engine {self.engine_path} already loaded, skip reloading")
            return

        if not hasattr(self, "engine_bytes_cpu") or self.engine_bytes_cpu is None:
            # keep a cpu copy of the engine to reduce reloading time.
            print(f"Loading TensorRT engine to cpu bytes: {self.engine_path}")
            self.engine_bytes_cpu = bytes_from_path(self.engine_path)

        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(self.engine_bytes_cpu)

    def unload(self):
        if self.engine is not None:
            print(f"Unloading TensorRT engine: {self.engine_path}")
            del self.engine
            self.engine = None
            gc.collect()
        else:
            print(f"[W]: Unload an unloaded engine {self.engine_path}, skip unloading")

    def activate(
        self,
        device: str | torch.device,
        device_memory: int | None = None,
    ):
        self.device = device
        if device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = device_memory
            self.shared_device_memory = device_memory
        else:
            self.context = self.engine.create_execution_context()
            self.shared_device_memory = self.engine.device_memory_size

    def reactivate(self, device_memory: int):
        assert self.context
        self.context.device_memory = device_memory

    def deactivate(self):
        del self.context
        self.context = None

    def allocate_buffers(
        self,
        shape_dict: dict[str, tuple],
        device: str | torch.device ="cuda",
    ):
        for binding in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(binding)
            tensor_shape = shape_dict[tensor_name]

            if tensor_name in self.tensors and self.tensors[tensor_name].shape == tensor_shape:
                continue

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, tensor_shape)
            tensor_dtype = self.trt_to_torch_dtype_dict[self.engine.get_tensor_dtype(tensor_name)]
            tensor = torch.empty(
                size=tensor_shape,
                dtype=tensor_dtype,
            ).to(device=device)
            self.tensors[tensor_name] = tensor

    def deallocate_buffers(self):
        if self.engine is None:
            return

        for binding in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(binding)
            if tensor_name in self.tensors:
                del self.tensors[tensor_name]

        torch.cuda.empty_cache()
        self.tensors = OrderedDict()

    def infer(
        self,
        feed_dict: dict[str, torch.Tensor],
    ):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        noerror = self.context.execute_async_v3(self.stream)
        if not noerror:
            raise ValueError("ERROR: inference failed.")

        return self.tensors
