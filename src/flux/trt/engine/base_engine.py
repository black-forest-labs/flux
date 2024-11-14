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

from collections import OrderedDict
import gc

from cuda import cudart
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes
import tensorrt as trt
import torch
from abc import ABC, abstractmethod

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

    def __init__(
        self,
        engine_path: str,
    ):
        self.engine_path = engine_path
        self.stream: cudart.cudaStream_t | None = None
        self.engine: trt.ICudaEngine | None = None
        self.context = None
        self.tensors = OrderedDict()

    @abstractmethod
    def __call__(self, *args, **Kwargs) -> torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor]:
        pass

    @abstractmethod
    def get_shape_dict(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ) -> dict[str, tuple]:
        pass

    def load(
        self,
        stream: cudart.cudaStream_t,
    ):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))
        self.stream = stream

    def activate(
        self,
        device_memory: int | None = None,
    ):
        if device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = device_memory
        else:
            self.context = self.engine.create_execution_context()

    def reactivate(self, device_memory: int):
        assert self.context
        self.context.device_memory = device_memory

    def deactivate(self):
        del self.context
        self.context = None
        gc.collect()

    def allocate_buffers(
        self,
        shape_dict: dict[str, tuple],
        device="cuda",
    ):
        for binding in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(binding)
            tensor_shape = shape_dict[tensor_name]

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, tensor_shape)
            tensor_dtype = self.trt_to_torch_dtype_dict[self.engine.get_tensor_dtype(tensor_name)]
            tensor = torch.empty(
                size=tensor_shape,
                dtype=tensor_dtype,
            ).to(device=device)
            self.tensors[tensor_name] = tensor

    def deallocate_buffers(self):
        for idx in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(binding)
            del self.tensors[tensor_name]

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
            raise ValueError(f"ERROR: inference failed.")

        return self.tensors
