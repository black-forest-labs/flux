from typing import Any
from collections import OrderedDict
import gc

import numpy as np

from cuda import cudart
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import (
    CreateConfig,
    ModifyNetworkOutputs,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.logger import G_LOGGER
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


# Map of TensorRT dtype -> torch dtype
trt_to_torch_dtype_dict = {
    trt.DataType.BOOL: torch.bool,
    trt.DataType.UINT8: torch.uint8,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.HALF: torch.float16,
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.BF16: torch.bfloat16,
}


class BaseEngine(ABC):
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

    def build(
        self,
        onnx_path: str,
        strongly_typed=False,
        fp16=True,
        bf16=False,
        tf32=False,
        int8=False,
        fp8=False,
        input_profile: dict[str, Any] | None = None,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names: list[str] | None = None,
        native_instancenorm=True,
        verbose=False,
        **extra_build_args,
    ):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        if not enable_all_tactics:
            extra_build_args["tactic_sources"] = []

        flags = []
        if native_instancenorm:
            flags.append(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
        network = network_from_onnx_path(onnx_path, flags=flags, strongly_typed=strongly_typed)
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)
        with G_LOGGER.verbosity(G_LOGGER.EXTRA_VERBOSE if verbose else G_LOGGER.ERROR):
            engine = engine_from_network(
                network,
                config=CreateConfig(
                    fp16=fp16,
                    bf16=bf16,
                    tf32=tf32,
                    int8=int8,
                    fp8=fp8,
                    refittable=enable_refit,
                    profiles=[p],
                    load_timing_cache=timing_cache,
                    **extra_build_args,
                ),
                save_timing_cache=timing_cache,
            )
            save_engine(engine, path=self.engine_path)

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

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
            tensor_dtype = trt_to_torch_dtype_dict[self.engine.get_tensor_dtype(tensor_name)]
            tensor = torch.empty(
                tensor_shape,
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
