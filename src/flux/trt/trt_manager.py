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
import os
import sys
import warnings

import tensorrt as trt
import torch
from cuda import cudart

from flux.trt.engine import (
    BaseEngine,
    CLIPEngine,
    Engine,
    T5Engine,
    TransformerEngine,
    VAEDecoder,
    VAEEncoder,
    VAEEngine,
)
from flux.trt.trt_config import (
    TRTBaseConfig,
    get_config,
)

TRT_LOGGER = trt.Logger()


class TRTManager:
    @property
    def model_to_engine_class(self) -> dict[str, type[Engine]]:
        return {
            "clip": CLIPEngine,
            "transformer": TransformerEngine,
            "t5": T5Engine,
            "vae": VAEDecoder,
            "vae_encoder": VAEEncoder,
        }

    def __init__(
        self,
        max_batch=2,
        tf32=True,
        bf16=False,
        fp8=False,
        fp4=False,
        verbose=False,
    ):
        assert bf16 + fp8 + fp4 == 1, "only one model type can be active"
        self.max_batch = max_batch
        self.tf32 = tf32
        self.bf16 = bf16
        self.fp8 = fp8
        self.fp4 = fp4
        self.verbose = verbose
        self.runtime: trt.Runtime = None

        assert torch.cuda.is_available(), "No cuda device available"

    @staticmethod
    def _create_directories(engine_dir: str):
        print(f"[I] Create directory: {engine_dir} if not existing")
        os.makedirs(engine_dir, exist_ok=True)

    def _get_trt_configs(
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
    ) -> dict[str, TRTBaseConfig]:
        trt_configs = {}
        for model_name, model in models.items():
            config_cls = get_config(
                model_name=model_name,
                tf32=self.tf32,
                bf16=self.bf16,
                fp8=self.fp8,
                fp4=self.fp4,
            )

            trt_config = config_cls.from_model(
                model=model,
                max_batch=self.max_batch,
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

            trt_configs[model_name] = trt_config

        if "transformer" in trt_configs and "t5" in trt_configs:
            trt_configs["transformer"].text_maxlen = trt_configs["t5"].text_maxlen
        else:
            warnings.warn("`text_maxlen` attribute of flux-trasformer is not update. Default value is used.")

        return trt_configs

    @staticmethod
    def _build_engine(
        trt_config: TRTBaseConfig,
        batch_size: int,
        image_height: int,
        image_width: int,
    ):
        already_build = os.path.exists(trt_config.engine_path)
        if already_build:
            return

        trt_config.build_trt_engine(
            engine_path=trt_config.engine_path,
            onnx_path=trt_config.onnx_path,
            strongly_typed=trt_config.trt_build_strongly_typed,
            tf32=trt_config.trt_tf32,
            bf16=trt_config.trt_bf16,
            fp8=trt_config.trt_fp8,
            fp4=trt_config.trt_fp4,
            input_profile=trt_config.get_input_profile(
                batch_size=batch_size,
                image_height=image_height,
                image_width=image_width,
                static_batch=trt_config.trt_static_batch,
                static_shape=trt_config.trt_static_shape,
            ),
            enable_all_tactics=trt_config.trt_enable_all_tactics,
            timing_cache=trt_config.trt_timing_cache,
            update_output_names=trt_config.trt_update_output_names,
            builder_optimization_level=trt_config.trt_builder_optimization_level,
            verbose=trt_config.trt_verbose,
        )

        # Reclaim GPU memory from torch cache
        gc.collect()
        torch.cuda.empty_cache()

    def load_engines(
        self,
        models: dict[str, torch.nn.Module],
        engine_dir: str,
        onnx_dir: str,
        trt_image_height: int,
        trt_image_width: int,
        trt_batch_size=1,
        trt_static_batch=True,
        trt_static_shape=True,
        trt_enable_all_tactics=False,
        trt_timing_cache: str | None = None,
        trt_native_instancenorm=True,
        trt_builder_optimization_level=3,
        trt_precision_constraints="none",
    ) -> dict[str, BaseEngine]:
        self._create_directories(engine_dir=engine_dir)

        trt_configs = self._get_trt_configs(
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
        for model_name, trt_config in trt_configs.items():
            self._build_engine(
                trt_config=trt_config,
                batch_size=trt_batch_size,
                image_height=trt_image_height,
                image_width=trt_image_width,
            )

        gc.collect()
        torch.cuda.empty_cache()
        self.init_runtime()
        # load TRT engines
        engines = {}
        for model_name, trt_config in trt_configs.items():
            engine_class = self.model_to_engine_class[model_name]
            engine = engine_class(
                trt_config=trt_config,
                stream=self.stream,
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
        print("[I] Init TRT runtime")
        self.runtime = trt.Runtime(TRT_LOGGER)
        enter_fn = type(self.runtime).__enter__
        enter_fn(self.runtime)
        self.stream = cudart.cudaStreamCreate()[1]

    def stop_runtime(self):
        exit_fn = type(self.runtime).__exit__
        exit_fn(self.runtime, *sys.exc_info())
        cudart.cudaStreamDestroy(self.stream)
        del self.stream
        print("[I] Stop TRT runtime")
