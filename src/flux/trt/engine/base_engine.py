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
from typing import Dict

import tensorrt as trt
import torch
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes

from flux.trt.trt_config import TRTBaseConfig

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


class SharedMemory(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super(SharedMemory, cls).__new__(cls)
            cls.instance.__init__(*args, **kwargs)
        return cls.instance

    def __init__(self, size: int, device=torch.device("cuda")):
        self.allocations = {}
        self._buffer = torch.empty(
            size,
            dtype=torch.uint8,
            device=device,
            memory_format=torch.contiguous_format,
        )

    def resize(self, name: str, size: int):
        self.allocations[name] = size
        if max(self.allocations.values()) > self._buffer.numel():
            self.buffer = self._buffer.resize_(size)
            torch.cuda.empty_cache()

    def reset(self, name: str):
        self.allocations.pop(name)
        new_max = max(self.allocations.values())
        if new_max < self._buffer.numel():
            self.buffer = self._buffer.resize_(new_max)
            torch.cuda.empty_cache()

    def deallocate(self):
        del self._buffer
        torch.cuda.empty_cache()
        self._buffer = torch.empty(
            1024,
            dtype=torch.uint8,
            device="cuda",
            memory_format=torch.contiguous_format,
        )

    @property
    def shared_device_memory(self):
        return self._buffer.data_ptr()

    def __str__(self):
        def human_readable_size(size):
            for unit in ["B", "KiB", "MiB", "GiB"]:
                if size < 1024.0:
                    return size, unit
                size /= 1024.0
            return size, unit

        allocations_str = []

        for name, size_bytes in self.allocations.items():
            size, unit = human_readable_size(size_bytes)
            allocations_str.append(f"\t{name}: {size:.2f} {unit}\n")
        allocations_output = "".join(allocations_str)

        size, unit = human_readable_size(self._buffer.numel())
        allocations_buffer = f"{size:.2f} {unit}"
        return f"Shared Memory Allocations: \n{allocations_output} \n\tCurrent: {allocations_buffer}"


TRT_ALLOCATION_POLICY = {"global", "dynamic"}
TRT_OFFLOAD_POLICY = "cpu_buffer"


class BaseEngine(ABC):
    @staticmethod
    def trt_datatype_to_torch(datatype):
        datatype_mapping = {
            trt.DataType.BOOL: torch.bool,
            trt.DataType.UINT8: torch.uint8,
            trt.DataType.INT8: torch.int8,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.HALF: torch.float16,
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.BF16: torch.bfloat16,
        }
        if datatype not in datatype_mapping:
            raise ValueError(f"No PyTorch equivalent for TensorRT data type: {datatype}")

        return datatype_mapping[datatype]

    @abstractmethod
    def cpu(self) -> "BaseEngine":
        pass

    @abstractmethod
    def cuda(self) -> "BaseEngine":
        pass

    @abstractmethod
    def to(self, device: str | torch.device) -> "BaseEngine":
        pass


class Engine(BaseEngine):
    def __init__(
        self,
        trt_config: TRTBaseConfig,
        stream: torch.cuda.Stream,
        context_memory: SharedMemory | None = None,
        allocation_policy: str = "global",
    ):
        self.trt_config = trt_config
        self.stream = stream
        self.context = None
        self.tensors = OrderedDict()
        self.context_memory = context_memory
        self.device: torch.device = torch.device("cpu")

        if TRT_OFFLOAD_POLICY == "cpu_buffer":
            self.engine: trt.ICudaEngine | bytes = None
            self.cpu_engine_buffer: bytes = bytes_from_path(self.trt_config.engine_path)
        else:
            self.engine: trt.ICudaEngine | bytes = bytes_from_path(self.trt_config.engine_path)

        assert allocation_policy in TRT_ALLOCATION_POLICY
        self.allocation_policy = allocation_policy
        self.current_input_hash = None
        self.cuda_graph = None

    @abstractmethod
    def __call__(self, *args, **Kwargs) -> torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor]:
        pass

    def cpu(self) -> "Engine":
        if self.device == torch.device("cpu"):
            return self
        self.deactivate()
        if TRT_OFFLOAD_POLICY == "cpu_buffer":
            del self.engine
            return self
        self.engine = memoryview(self.engine.serialize())

        return self

    def cuda(self) -> "Engine":
        if self.device == torch.device("cuda"):
            return self
        buffer = self.cpu_engine_buffer if TRT_OFFLOAD_POLICY == "cpu_buffer" else self.engine
        self.engine = engine_from_bytes(buffer)
        gc.collect()
        self.context = self.engine.create_execution_context_without_device_memory()
        self.context_memory.resize(self.__class__.__name__, self.device_memory_size)
        self.context.device_memory = self.context_memory.shared_device_memory
        return self

    def to(self, device: str | torch.device) -> "Engine":
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if self.device == device:
            return self
        if device == torch.device("cpu"):
            self.cpu()
        else:
            self.cuda()
        self.device = device
        return self

    def deactivate(self):
        del self.context
        self.context = None

    def allocate_buffers(
        self,
        shape_dict: dict[str, tuple],
        device: str | torch.device = "cuda",
    ):
        for binding in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(binding)
            tensor_shape = shape_dict[tensor_name]

            if tensor_name in self.tensors and self.tensors[tensor_name].shape == tensor_shape:
                continue

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, tensor_shape)
            tensor_dtype = self.trt_datatype_to_torch(self.engine.get_tensor_dtype(tensor_name))
            tensor = torch.empty(
                size=tensor_shape,
                dtype=tensor_dtype,
                memory_format=torch.contiguous_format,
            ).to(device=device)
            self.tensors[tensor_name] = tensor

    def get_dtype(self, tensor_name: str):
        return self.trt_datatype_to_torch(self.engine.get_tensor_dtype(tensor_name))

    def override_shapes(self, feed_dict: Dict[str, torch.Tensor]):
        for name, tensor in feed_dict.items():
            shape = tensor.shape
            assert tensor.dtype == self.trt_datatype_to_torch(self.engine.get_tensor_dtype(name)), (
                f"Debug: Mismatched data types for tensor '{name}'. "
                f"Expected: {self.trt_datatype_to_torch(self.engine.get_tensor_dtype(name))}, "
                f"Found: {tensor.dtype} "
                f"in {self.__class__.__name__}"
            )
            self.context.set_input_shape(name, shape)

        assert self.context.all_binding_shapes_specified
        self.context.infer_shapes()
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            dtype = self.trt_datatype_to_torch(self.engine.get_tensor_dtype(name))
            shape = self.context.get_tensor_shape(name)
            if -1 in shape:
                raise Exception("Unspecified shape identified for tensor {}: {} ".format(name, shape))
            self.tensors[name] = torch.zeros(tuple(shape), dtype=dtype, device=self.device).contiguous()
            self.context.set_tensor_address(name, self.tensors[name].data_ptr())

        if self.allocation_policy == "dynamic":
            self.context_memory.resize(self.__class__.__name__, self.device_memory_size)
        self.current_input_hash = self.calculate_input_hash(feed_dict)

    def deallocate_buffers(self):
        if len(self.tensors) == 0:
            return

        del self.tensors
        self.tensors = OrderedDict()
        torch.cuda.empty_cache()

    @property
    def device_memory_size(self):
        if self.allocation_policy == "global":
            return self.engine.device_memory_size
        else:
            if not self.context.all_binding_shapes_specified:
                return 0
            return self.context.update_device_memory_size_for_shapes()

    @staticmethod
    def calculate_input_hash(feed_dict: Dict[str, torch.Tensor]):
        return hash(tuple(feed_dict[key].shape for key in sorted(feed_dict.keys())))

    def _capture_cuda_graph(self):
        self.cuda_graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.graph(self.cuda_graph, stream=s):
            noerror = self.context.execute_async_v3(s.cuda_stream)
            if not noerror:
                raise ValueError("ERROR: inference failed.")

        # self.cuda_graph.replay()

    def infer(
        self,
        feed_dict: dict[str, torch.Tensor],
    ):
        if self.current_input_hash != self.calculate_input_hash(feed_dict):
            self.override_shapes(feed_dict)

        self.context.device_memory = self.context_memory.shared_device_memory
        for name, tensor in feed_dict.items():
            self.tensors[name].copy_(tensor, non_blocking=True)

        noerror = self.context.execute_async_v3(self.stream.cuda_stream)
        if not noerror:
            raise ValueError("ERROR: inference failed.")

        return self.tensors

    def __str__(self):
        if self.engine is None:
            return "Engine has not been initialized"
        out = ""
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.trt_datatype_to_torch(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            out += f"\t{mode.name}: {name}={shape} {dtype.__str__()}\n"
        return out
