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
from typing import Any, Union

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
        max_batch=1,
        fp16=False,
        bf16=False,
        tf32=True,
        static_batch=True,
        static_shape=True,
        verbose=True,
        **kwargs,
    ):
        self.device = device
        self.max_batch = max_batch
        self.fp16 = fp16
        self.bf16 = bf16
        self.tf32 = tf32
        self.static_batch = static_batch
        self.static_shape = static_shape
        self.verbose = verbose
        self.runtime: trt.Runtime = None

        assert torch.cuda.is_available(), "No cuda device available"

    @staticmethod
    def _create_directories(engine_dir: str, onnx_dir: str):
        # Create directories if missing
        for directory in [engine_dir, onnx_dir]:
            print(f"[I] Create directory: {directory} if not existing")
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def _get_onnx_path(
        model_name: str,
        onnx_dir: str,
        opt: bool = True,
        suffix: str = "",
        transformer_precision: str = "bf16",
    ) -> str:
        onnx_model_dir = os.path.join(
            onnx_dir,
            model_name + suffix + (".opt" if opt else ""),
        )
        if model_name == "transformer":
            onnx_model_dir = os.path.join(onnx_model_dir, transformer_precision)
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, "model.onnx")

    @staticmethod
    def _get_engine_path(
        model_name: str,
        engine_dir: str,
        suffix: str = "",
        transformer_precision: str = "bf16",
    ) -> str:
        return os.path.join(
            engine_dir,
            model_name
            + suffix
            + (f"_{transformer_precision}" if model_name == "transformer" else "")
            + ".trt"
            + trt.__version__
            + ".plan",
        )

    @staticmethod
    def _get_weights_map_path(
        model_name: str,
        onnx_dir: str,
    ) -> str:
        onnx_model_dir = os.path.join(onnx_dir, model_name + ".opt")
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, "weights_map.json")

    @staticmethod
    def _get_refit_nodes_path(
        model_name: str,
        onnx_dir: str,
        suffix: str = "",
    ) -> str:
        onnx_model_dir = os.path.join(onnx_dir, model_name + ".opt")
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, "refit" + suffix + ".json")

    @staticmethod
    def _prepare_model_configs(
        models: dict[str, torch.nn.Module],
        engine_dir: str,
        onnx_dir: str,
        transformer_precision: str,
    ) -> dict[str, dict[str, Any]]:
        model_names = models.keys()
        configs = {}
        for model_name in model_names:
            config: dict[str, Any] = {}
            config["model_suffix"] = ""

            config["onnx_path"] = TRTManager._get_onnx_path(
                model_name=model_name,
                onnx_dir=onnx_dir,
                opt=False,
                suffix=config["model_suffix"],
                transformer_precision=transformer_precision,
            )
            config["onnx_opt_path"] = TRTManager._get_onnx_path(
                model_name=model_name,
                onnx_dir=onnx_dir,
                suffix=config["model_suffix"],
                transformer_precision=transformer_precision,
            )
            config["engine_path"] = TRTManager._get_engine_path(
                model_name=model_name,
                engine_dir=engine_dir,
                suffix=config["model_suffix"],
                transformer_precision=transformer_precision,
            )

            configs[model_name] = config

        return configs

    def _get_exporters(
        self,
        models: dict[str, torch.nn.Module],
    ) -> dict[str, Union[BaseMixin, BaseExporter]]:
        exporters = {}
        for model_name, model in models.items():
            exporter_class = self.model_to_exporter_dict[model_name]

            if model_name == "t5":
                # traced in tf32 for numerical stability when on fp16
                exporter = exporter_class(
                    model=model,
                    fp16=False,
                    bf16=self.bf16,
                    tf32=self.tf32,
                    max_batch=self.max_batch,
                    verbose=self.verbose,
                )
                exporters[model_name] = exporter

            elif model_name.startswith("vae"):
                # Accuracy issues with FP16 and BF16
                # fallback to FP32
                exporter = exporter_class(
                    model=model,
                    fp16=False,
                    bf16=False,
                    tf32=self.tf32,
                    max_batch=self.max_batch,
                    verbose=self.verbose,
                )
                exporters[model_name] = exporter

            else:
                onnx_exporter = exporter_class(
                    model=model,
                    fp16=self.fp16,
                    bf16=self.bf16,
                    tf32=self.tf32,
                    max_batch=self.max_batch,
                    verbose=self.verbose,
                )
                exporters[model_name] = onnx_exporter

        if "transformer" in exporters and "t5" in exporters:
            exporters["transformer"].text_maxlen = exporters["t5"].text_maxlen
        else:
            warnings.warn("`text_maxlen` attribute of flux-trasformer is not update. Default value is used.")

        return exporters

    def _export_onnx(
        self,
        model_exporter: Union[BaseMixin, BaseExporter],
        model_config: dict[str, Any],
        opt_image_height: int,
        opt_image_width: int,
        onnx_opset: int,
    ):
        do_export_onnx = not os.path.exists(model_config["engine_path"]) and not os.path.exists(
            model_config["onnx_opt_path"]
        )

        model_exporter.model = model_exporter.model.to(self.device)

        if do_export_onnx:
            model_exporter.export_onnx(
                onnx_path=model_config["onnx_path"],
                onnx_opt_path=model_config["onnx_opt_path"],
                onnx_opset=onnx_opset,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
            )

        model_exporter.model = model_exporter.model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _build_engine(
        model_exporter: BaseExporter,
        model_config: dict[str, Any],
        opt_batch_size: int,
        opt_image_height: int,
        opt_image_width: int,
        static_batch: bool,
        static_shape: bool,
        optimization_level: int,
        enable_all_tactics: bool,
        timing_cache,
        verbose: bool,
    ):
        already_build = os.path.exists(model_config["engine_path"])
        if already_build:
            return

        update_output_names = (
            model_exporter.get_output_names() + model_exporter.extra_output_names
            if model_exporter.extra_output_names
            else None
        )
        fp16amp = False if getattr(model_exporter, "build_strongly_typed", False) else model_exporter.fp16
        tf32amp = model_exporter.tf32
        bf16amp = False if getattr(model_exporter, "build_strongly_typed", False) else model_exporter.bf16
        strongly_typed = True if getattr(model_exporter, "build_strongly_typed", False) else False

        extra_build_args = {
            "verbose": verbose,
            "builder_optimization_level": optimization_level,
        }

        model_exporter.build(
            engine_path=model_config["engine_path"],
            onnx_path=model_config["onnx_opt_path"],
            strongly_typed=strongly_typed,
            fp16=fp16amp,
            tf32=tf32amp,
            bf16=bf16amp,
            input_profile=model_exporter.get_input_profile(
                batch_size=opt_batch_size,
                image_height=opt_image_height,
                image_width=opt_image_width,
                static_batch=static_batch,
                static_shape=static_shape,
            ),
            enable_all_tactics=enable_all_tactics,
            timing_cache=timing_cache,
            update_output_names=update_output_names,
            **extra_build_args,
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
        transformer_precision: str,
        opt_batch_size=1,
        onnx_opset=19,
        optimization_level=3,
        enable_all_tactics=False,
        timing_cache=None,
    ) -> dict[str, BaseEngine]:
        assert transformer_precision in ["bf16", "fp8", "fp4"], "Invalid transformer precision"

        self._create_directories(
            engine_dir=engine_dir,
            onnx_dir=onnx_dir,
        )

        model_configs = self._prepare_model_configs(
            models=models,
            engine_dir=engine_dir,
            onnx_dir=onnx_dir,
            transformer_precision=transformer_precision,
        )

        exporters = self._get_exporters(models)

        # Export models to ONNX
        for model_name, model_exporter in exporters.items():
            self._export_onnx(
                model_exporter=model_exporter,
                model_config=model_configs[model_name],
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                onnx_opset=onnx_opset,
            )

        # Build TRT engines
        for model_name, model_exporter in exporters.items():
            model_config = model_configs[model_name]
            self._build_engine(
                model_exporter=model_exporter,
                model_config=model_config,
                opt_batch_size=opt_batch_size,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                static_batch=self.static_batch,
                static_shape=self.static_shape,
                optimization_level=optimization_level,
                enable_all_tactics=enable_all_tactics,
                timing_cache=timing_cache,
                verbose=self.verbose,
            )

        # load TRT engines
        engines = {}
        for model_name, model_exporter in exporters.items():
            model_config = model_configs[model_name]

            engine_class = self.model_to_engine_class[model_name]
            engine = engine_class(
                engine_path=model_config["engine_path"],
                **model_exporter.get_mixin_params(),
            )
            engines[model_name] = engine

        if "vae" in engines:
            engines["vae"] = VAEEngine(
                decoder=engines.pop("vae"),
                encoder=engines.pop("vae_encoder", None),
            )
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
        self.runtime = trt.Runtime(TRT_LOGGER)
        enter_fn = type(self.runtime).__enter__
        enter_fn(self.runtime)
        self.stream = cudart.cudaStreamCreate()[1]

    def stop_runtime(self):
        exit_fn = type(self.runtime).__exit__
        exit_fn(self.runtime, *sys.exc_info())
        cudart.cudaStreamDestroy(self.stream)
        del self.stream
