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
import gc
import os
import sys
import warnings
from typing import Union

import tensorrt as trt
import torch
from cuda import cudart

from flux.trt.engine import (
    BaseEngine,
    CLIPEngine,
    T5Engine,
    TransformerEngine,
    VAEDecoder,
    VAEEncoder,
    VAEEngine,
)
from flux.trt.exporter import (
    BaseExporter,
    CLIPExporter,
    T5Exporter,
    TransformerExporter,
    VAEDecoderExporter,
    VAEEncoderExporter,
    get_config,
)
from flux.trt.mixin import BaseMixin

TRT_LOGGER = trt.Logger()


class TRTManager:
    @property
    def model_to_engine_class(self) -> dict[str, type[Union[BaseMixin, BaseEngine]]]:
        return {
            "clip": CLIPEngine,
            "transformer": TransformerEngine,
            "t5": T5Engine,
            "vae": VAEDecoder,
            "vae_encoder": VAEEncoder,
        }

    @property
    def model_to_exporter_dict(self) -> dict[str, type[Union[BaseMixin, BaseExporter]]]:
        return {
            "clip": CLIPExporter,
            "transformer": TransformerExporter,
            "t5": T5Exporter,
            "vae": VAEDecoderExporter,
            "vae_encoder": VAEEncoderExporter,
        }

    def __init__(
        self,
        device: str | torch.device,
        max_batch=2,
        tf32=True,
        bf16=False,
        fp8=False,
        fp4=False,
        t5_fp8=False,
        verbose=False,
    ):
        assert bf16 + fp8 + fp4 == 1, "only one model type can be active"
        self.device = device
        self.max_batch = max_batch
        self.tf32 = tf32
        self.bf16 = bf16
        self.fp8 = fp8
        self.fp4 = fp4
        self.t5_fp8 = t5_fp8
        self.verbose = verbose
        self.runtime: trt.Runtime = None

        assert torch.cuda.is_available(), "No cuda device available"

    @staticmethod
    def _create_directories(engine_dir: str, onnx_dir: str):
        # Create directories if missing
        for directory in [engine_dir, onnx_dir]:
            print(f"[I] Create directory: {directory} if not existing")
            os.makedirs(directory, exist_ok=True)

    def _get_exporters(
        self,
        models: dict[str, torch.nn.Module],
        engine_dir: str,
        onnx_dir: str,
        trt_static_batch: bool,
        trt_static_shape: bool,
        trt_enable_all_tactics: bool,
        trt_timing_cache: str | None,
        trt_native_instancenorm: bool,
        trt_builder_optimization_level: int,
        trt_precision_constraints: str,
    ) -> dict[str, Union[BaseMixin, BaseExporter]]:
        exporters = {}
        for model_name, model in models.items():
            exporter_class = self.model_to_exporter_dict[model_name]
            config_cls = get_config(
                model_name=model_name,
                tf32=self.tf32,
                bf16=self.bf16,
                fp8=self.fp8,
                fp4=self.fp4,
                t5_fp8=self.t5_fp8,
            )
            trt_config = config_cls(
                onnx_dir=onnx_dir,
                engine_dir=engine_dir,
                trt_verbose=self.verbose,
                precision="fp4" if self.fp4 else "fp8" if self.fp8 else "bf16",
                trt_static_batch=trt_static_batch,
                trt_static_shape=trt_static_shape,
                trt_enable_all_tactics=trt_enable_all_tactics,
                trt_timing_cache=trt_timing_cache,
                trt_native_instancenorm=trt_native_instancenorm,
                trt_builder_optimization_level=trt_builder_optimization_level,
                trt_precision_constraints=trt_precision_constraints,
            )
            exporters[model_name] = exporter_class(
                trt_config=trt_config,
                max_batch=self.max_batch,
                model=model,
            )

        if "transformer" in exporters and "t5" in exporters:
            exporters["transformer"].text_maxlen = exporters["t5"].text_maxlen
        else:
            warnings.warn("`text_maxlen` attribute of flux-trasformer is not update. Default value is used.")

        return exporters

    @staticmethod
    def _build_engine(
        model_exporter: BaseExporter,
        opt_batch_size: int,
        opt_image_height: int,
        opt_image_width: int,
    ):
        already_build = os.path.exists(model_exporter.trt_config.engine_path)
        if already_build:
            return

        model_exporter.build(
            engine_path=model_exporter.trt_config.engine_path,
            onnx_path=model_exporter.trt_config.onnx_path,
            strongly_typed=model_exporter.trt_config.trt_build_strongly_typed,
            tf32=model_exporter.trt_config.trt_tf32,
            bf16=model_exporter.trt_config.trt_bf16,
            fp8=model_exporter.trt_config.trt_fp8,
            fp4=model_exporter.trt_config.trt_fp4,
            input_profile=model_exporter.get_input_profile(
                batch_size=opt_batch_size,
                image_height=opt_image_height,
                image_width=opt_image_width,
                static_batch=model_exporter.trt_config.trt_static_batch,
                static_shape=model_exporter.trt_config.trt_static_shape,
            ),
            enable_all_tactics=model_exporter.trt_config.trt_enable_all_tactics,
            timing_cache=model_exporter.trt_config.trt_timing_cache,
            update_output_names=model_exporter.trt_config.trt_update_output_names,
            builder_optimization_level=model_exporter.trt_config.trt_builder_optimization_level,
            verbose=model_exporter.trt_config.trt_verbose,
        )

        # Reclaim GPU memory from torch cache
        gc.collect()
        torch.cuda.empty_cache()

    def load_engines(
        self,
        models: dict[str, torch.nn.Module],
        engine_dir: str,
        onnx_dir: str,
        opt_image_height: int,
        opt_image_width: int,
        opt_batch_size=1,
        trt_static_batch=True,
        trt_static_shape=True,
        trt_enable_all_tactics=False,
        trt_timing_cache: str | None = None,
        trt_native_instancenorm=True,
        trt_builder_optimization_level=3,
        trt_precision_constraints="none",
    ) -> dict[str, BaseEngine]:
        self._create_directories(
            engine_dir=engine_dir,
            onnx_dir=onnx_dir,
        )

        exporters = self._get_exporters(
            models,
            engine_dir=engine_dir,
            onnx_dir=onnx_dir,
            trt_static_batch=trt_static_batch,
            trt_static_shape=trt_static_shape,
            trt_enable_all_tactics=trt_enable_all_tactics,
            trt_timing_cache=trt_timing_cache,
            trt_native_instancenorm=trt_native_instancenorm,
            trt_builder_optimization_level=trt_builder_optimization_level,
            trt_precision_constraints=trt_precision_constraints,
        )

        # Build TRT engines
        for model_name, model_exporter in exporters.items():
            self._build_engine(
                model_exporter=model_exporter,
                opt_batch_size=opt_batch_size,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
            )

        gc.collect()
        torch.cuda.empty_cache()
        self.init_runtime()
        # load TRT engines
        engines = {}
        for model_name, model_exporter in exporters.items():
            engine_class = self.model_to_engine_class[model_name]
            engine = engine_class(
                engine_path=model_exporter.trt_config.engine_path,
                stream=self.stream,
                **model_exporter.get_mixin_params(),
            )
            engines[model_name] = engine

        if "vae" in engines:
            engines["vae"] = VAEEngine(
                decoder=engines.pop("vae"),
                encoder=engines.pop("vae_encoder", None),
            )
        gc.collect()
        torch.cuda.empty_cache()
        return engines

    @staticmethod
    def calculate_max_device_memory(engines: dict[str, BaseEngine]) -> int:
        max_device_memory = 0
        for model_name, engine in engines.items():
            if model_name == "vae":
                # TODO: refactor VAEengine by adding a engine.device_memory_size that return the device memory for decoder + encoder
                max_device_memory = max(max_device_memory, engine.get_device_memory())
            else:
                max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
        return max_device_memory

    def init_runtime(self):
        print("init TRT runtime")
        self.runtime = trt.Runtime(TRT_LOGGER)
        enter_fn = type(self.runtime).__enter__
        enter_fn(self.runtime)
        self.stream = cudart.cudaStreamCreate()[1]

    def stop_runtime(self):
        exit_fn = type(self.runtime).__exit__
        exit_fn(self.runtime, *sys.exc_info())
        cudart.cudaStreamDestroy(self.stream)
        del self.stream
