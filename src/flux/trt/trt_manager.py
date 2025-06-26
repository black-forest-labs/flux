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

from flux.trt.engine import (
    BaseEngine,
    CLIPEngine,
    Engine,
    SharedMemory,
    T5Engine,
    TransformerEngine,
    VAEDecoder,
    VAEEncoder,
    VAEEngine,
)
from flux.trt.trt_config import (
    ModuleName,
    TRTBaseConfig,
    get_config,
)

TRT_LOGGER = trt.Logger()
VALID_TRANSFORMER_PRECISIONS = {"bf16", "fp8", "fp4", "fp4_svd32"}
VALID_T5_PRECISIONS = {"bf16", "fp8"}


class TRTManager:
    @property
    def module_to_engine_class(self) -> dict[ModuleName, type[Engine]]:
        return {
            ModuleName.CLIP: CLIPEngine,
            ModuleName.TRANSFORMER: TransformerEngine,
            ModuleName.T5: T5Engine,
            ModuleName.VAE: VAEDecoder,
            ModuleName.VAE_ENCODER: VAEEncoder,
        }

    def __init__(
        self,
        trt_transformer_precision: str,
        trt_t5_precision: str,
        max_batch=2,
        verbose=False,
    ):
        self.max_batch = max_batch
        self.precisions = self._parse_models_precisions(
            trt_transformer_precision=trt_transformer_precision,
            trt_t5_precision=trt_t5_precision,
        )
        self.verbose = verbose
        self.runtime: trt.Runtime = None
        self.device_memory = SharedMemory(1024)

        assert torch.cuda.is_available(), "No cuda device available"

    @staticmethod
    def _parse_models_precisions(
        trt_transformer_precision: str, trt_t5_precision: str
    ) -> dict[ModuleName, str]:
        precisions = {
            ModuleName.CLIP: "bf16",
            ModuleName.VAE: "bf16",
            ModuleName.VAE_ENCODER: "bf16",
        }

        assert (
            trt_transformer_precision in VALID_TRANSFORMER_PRECISIONS
        ), f"Invalid precision for flux-transformer `{trt_transformer_precision}`. Possible value are {VALID_TRANSFORMER_PRECISIONS}"
        precisions[ModuleName.TRANSFORMER] = (
            trt_transformer_precision if trt_transformer_precision != "fp4_svd32" else "fp4"
        )

        assert (
            trt_t5_precision in VALID_T5_PRECISIONS
        ), f"Invalid precision for T5 `{trt_t5_precision}`. Possible value are {VALID_T5_PRECISIONS}"
        precisions[ModuleName.T5] = trt_t5_precision
        return precisions

    @staticmethod
    def _parse_custom_onnx_path(custom_onnx_paths: str) -> dict[ModuleName, str]:
        """Parse a string of comma-separated key-value pairs into a dictionary.

        Args:
            string (str): A string of comma-separated key-value pairs.

        Returns:
            Dict[str, str]: Parsed dictionary of key-value pairs.

        Example:
            >>> parse_key_value_pairs("key1:value1,key2:value2")
            {"key1": "value1", "key2": "value2"}
        """
        parsed = {}

        for key_value_pair in custom_onnx_paths.split(","):
            if not key_value_pair:
                continue

            key_value_pair = key_value_pair.split(":")
            if len(key_value_pair) != 2:
                raise ValueError(f"Invalid key-value pair: {key_value_pair}. Must have length 2.")
            key, value = key_value_pair
            key = ModuleName(key)
            parsed[key] = value

        return parsed

    @staticmethod
    def _create_directories(engine_dir: str):
        print(f"[I] Create directory: {engine_dir} if not existing")
        os.makedirs(engine_dir, exist_ok=True)

    def _get_trt_configs(
        self,
        model_name: str,
        module_names: set[ModuleName],
        engine_dir: str,
        custom_onnx_paths: dict[ModuleName, str],
        trt_static_batch: bool,
        trt_static_shape: bool,
        trt_enable_all_tactics: bool,
        trt_timing_cache: str | None,
        trt_native_instancenorm: bool,
        trt_builder_optimization_level: int,
        trt_precision_constraints: str,
        **kwargs,
    ) -> dict[ModuleName, TRTBaseConfig]:
        trt_configs = {}
        for module_name in module_names:
            config_cls = get_config(module_name=module_name, precision=self.precisions[module_name])
            custom_onnx_path = custom_onnx_paths.get(module_name, None)

            trt_config = config_cls.from_args(
                model_name=model_name,
                max_batch=self.max_batch,
                custom_onnx_path=custom_onnx_path,
                engine_dir=engine_dir,
                trt_verbose=self.verbose,
                precision=self.precisions[module_name],
                trt_static_batch=trt_static_batch,
                trt_static_shape=trt_static_shape,
                trt_enable_all_tactics=trt_enable_all_tactics,
                trt_timing_cache=trt_timing_cache,
                trt_native_instancenorm=trt_native_instancenorm,
                trt_builder_optimization_level=trt_builder_optimization_level,
                trt_precision_constraints=trt_precision_constraints,
                **kwargs,
            )

            trt_configs[module_name] = trt_config

        if ModuleName.TRANSFORMER in trt_configs and ModuleName.T5 in trt_configs:
            trt_configs[ModuleName.TRANSFORMER].text_maxlen = trt_configs[ModuleName.T5].text_maxlen
        else:
            warnings.warn("`text_maxlen` attribute of flux-trasformer is not update. Default value is used.")

        return trt_configs

    @staticmethod
    def _build_engine(
        trt_config: TRTBaseConfig,
        batch_size: int,
        image_height: int | None,
        image_width: int | None,
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
            ),
            enable_all_tactics=trt_config.trt_enable_all_tactics,
            timing_cache=trt_config.trt_timing_cache,
            update_output_names=trt_config.trt_update_output_names,
            builder_optimization_level=trt_config.trt_builder_optimization_level,
            verbose=trt_config.trt_verbose,
        )

        TRTManager._clean_memory()

    def load_engines(
        self,
        model_name: str,
        module_names: set[ModuleName],
        engine_dir: str,
        trt_image_height: int | None,
        trt_image_width: int | None,
        trt_batch_size=1,
        trt_static_batch=True,
        trt_static_shape=True,
        trt_enable_all_tactics=False,
        trt_timing_cache: str | None = None,
        trt_native_instancenorm=True,
        trt_builder_optimization_level=3,
        trt_precision_constraints="none",
        custom_onnx_paths="",
        **kwargs,
    ) -> dict[ModuleName, BaseEngine]:
        TRTManager._clean_memory()
        TRTManager._create_directories(engine_dir)
        custom_onnx_paths = TRTManager._parse_custom_onnx_path(custom_onnx_paths)

        trt_configs = self._get_trt_configs(
            model_name,
            module_names,
            engine_dir=engine_dir,
            custom_onnx_paths=custom_onnx_paths,
            trt_static_batch=trt_static_batch,
            trt_static_shape=trt_static_shape,
            trt_enable_all_tactics=trt_enable_all_tactics,
            trt_timing_cache=trt_timing_cache,
            trt_native_instancenorm=trt_native_instancenorm,
            trt_builder_optimization_level=trt_builder_optimization_level,
            trt_precision_constraints=trt_precision_constraints,
            **kwargs,
        )

        # Build TRT engines
        for module_name, trt_config in trt_configs.items():
            self._build_engine(
                trt_config=trt_config,
                batch_size=trt_batch_size,
                image_height=trt_image_height,
                image_width=trt_image_width,
            )

        self.init_runtime()
        # load TRT engines
        engines = {}
        for module_name, trt_config in trt_configs.items():
            engine_class = self.module_to_engine_class[module_name]
            engine = engine_class(
                trt_config=trt_config,
                stream=self.stream,
                context_memory=self.device_memory,
                allocation_policy=os.getenv("TRT_ALLOCATION_POLICY", "global"),
            )
            engines[module_name] = engine

        if ModuleName.VAE in engines:
            engines[ModuleName.VAE] = VAEEngine(
                decoder=engines.pop(ModuleName.VAE),
                encoder=engines.pop(ModuleName.VAE_ENCODER, None),
            )
        self._clean_memory()
        return engines

    @staticmethod
    def _clean_memory():
        gc.collect()
        torch.cuda.empty_cache()

    def init_runtime(self):
        print("[I] Init TRT runtime")
        self.runtime = trt.Runtime(TRT_LOGGER)
        enter_fn = type(self.runtime).__enter__
        enter_fn(self.runtime)
        self.stream = torch.cuda.current_stream()

    def stop_runtime(self):
        exit_fn = type(self.runtime).__exit__
        exit_fn(self.runtime, *sys.exc_info())
        del self.stream
        del self.device_memory
        print("[I] Stop TRT runtime")
