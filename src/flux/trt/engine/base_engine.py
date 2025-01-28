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
from abc import ABC, abstractmethod
from collections import OrderedDict

import tensorrt as trt
import torch
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
    def set_stream(self, stream):
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


class Engine(BaseEngine):
    def __init__(
        self,
        engine_path: str,
    ):
        self.engine_path = engine_path
        self.stream = None
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

    def set_stream(self, stream):
        self.stream = stream

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
        device: str,
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
        device="cuda",
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
