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

import os
import sys
import gc
import torch
import tensorrt as trt
from typing import Any, Union

from flux.trt.onnx_export import BaseExporter, CLIPExporter, FluxExporter, T5Exporter, VAEExporter
from flux.trt.mixin import BaseMixin
from flux.trt.engine import BaseEngine, CLIPEngine, FluxEngine, T5Engine, VAEEngine

TRT_LOGGER = trt.Logger()


class TRTManager:
    __stages__ = ["clip", "t5", "flux_transformer", "ae"]

    @property
    def stages(self) -> list[str]:
        return self.__stages__

    @property
    def model_to_engine_class(self) -> dict[str, type[Union[BaseMixin, BaseEngine]]]:
        return {
            "clip": CLIPEngine,
            "flux_transformer": FluxEngine,
            "t5": T5Engine,
            "vae": VAEEngine,
        }

    @property
    def model_to_exporter_dict(self) -> dict[str, type[Union[BaseMixin, BaseExporter]]]:
        return {
            "clip": CLIPExporter,
            "flux_transformer": FluxExporter,
            "t5": T5Exporter,
            "vae": VAEExporter,
        }

    def __init__(
        self,
        device: str | torch.device,
        max_batch=16,
        fp16=False,
        tf32=False,
        bf16=False,
        static_batch=True,
        verbose=True,
        **kwargs,
    ):
        self.device = device
        self.max_batch = max_batch
        self.fp16 = fp16
        self.tf32 = tf32
        self.bf16 = bf16
        self.static_batch = static_batch
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
    ) -> str:
        onnx_model_dir = os.path.join(
            onnx_dir,
            model_name + suffix + (".opt" if opt else ""),
        )
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, "model.onnx")

    @staticmethod
    def _get_engine_path(
        model_name: str,
        engine_dir: str,
        suffix: str = "",
    ) -> str:
        return os.path.join(
            engine_dir,
            model_name + suffix + ".trt" + trt.__version__ + ".plan",
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
    def _get_state_dict_path(
        model_name: str,
        onnx_dir: str,
        suffix: str = "",
    ) -> str:
        onnx_model_dir = os.path.join(onnx_dir, model_name + suffix)
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, "state_dict.pt")

    @staticmethod
    def _prepare_model_configs(
        models: dict[str, torch.nn.Module],
        engine_dir: str,
        onnx_dir: str,
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
            )
            config["onnx_opt_path"] = TRTManager._get_onnx_path(
                model_name=model_name,
                onnx_dir=onnx_dir,
                suffix=config["model_suffix"],
            )
            config["engine_path"] = TRTManager._get_engine_path(
                model_name=model_name,
                engine_dir=engine_dir,
                suffix=config["model_suffix"],
            )
            config["state_dict_path"] = TRTManager._get_state_dict_path(
                model_name=model_name,
                onnx_dir=onnx_dir,
                suffix=config["model_suffix"],
            )

            configs[model_name] = config

        return configs

    def _get_onnx_exporters(
        self,
        models: dict[str, torch.nn.Module],
    ) -> dict[str, Union[BaseMixin, BaseExporter]]:
        onnx_exporters = {}
        for model_name, model in models.items():
            onnx_exporter_class = self.model_to_exporter_dict[model_name]

            if model_name in {"ae", "t5"}:
                # traced in tf32 for numerical stability
                # pass expected dtype for cast the final output/input
                onnx_exporter = onnx_exporter_class(
                    model=model,
                    tf32=True,
                    max_batch=self.max_batch,
                    verbose=self.verbose,
                )
                onnx_exporters[model_name] = onnx_exporter

            else:
                onnx_exporter = onnx_exporter_class(
                    model=model,
                    fp16=self.fp16,
                    bf16=self.bf16,
                    tf32=self.tf32,
                    max_batch=self.max_batch,
                    verbose=self.verbose,
                )
                onnx_exporters[model_name] = onnx_exporter

        return onnx_exporters

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
        obj: BaseExporter,
        engine: BaseEngine,
        model_config: dict[str, Any],
        opt_batch_size: int,
        opt_image_height: int,
        opt_image_width: int,
        static_batch: bool,
        optimization_level: int,
        enable_all_tactics: bool,
        timing_cache,
        verbose: bool,
    ):
        update_output_names = obj.get_output_names() + obj.extra_output_names if obj.extra_output_names else None
        fp16amp = False if getattr(obj, "build_strongly_typed", False) else obj.fp16
        tf32amp = obj.tf32
        bf16amp = False if getattr(obj, "build_strongly_typed", False) else obj.bf16
        strongly_typed = True if getattr(obj, "build_strongly_typed", False) else False

        extra_build_args = {"verbose": verbose}
        extra_build_args["builder_optimization_level"] = optimization_level

        engine.build(
            model_config["onnx_opt_path"],
            strongly_typed=strongly_typed,
            fp16=fp16amp,
            tf32=tf32amp,
            bf16=bf16amp,
            input_profile=obj.get_input_profile(
                batch_size=opt_batch_size,
                image_height=opt_image_height,
                image_width=opt_image_width,
                static_batch=static_batch,
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
        onnx_opset: int,
        opt_batch_size: int,
        opt_image_height: int,
        opt_image_width: int,
        optimization_level=3,
        enable_all_tactics=False,
        timing_cache=None,
    ):
        # assert all(
        #     stage in models for stage in self.stages
        # ), f"some stage is missing\n\tstages: {models.keys()}\n\tneeded stages: {self.stages}"

        self._create_directories(
            engine_dir=engine_dir,
            onnx_dir=onnx_dir,
        )

        model_configs = self._prepare_model_configs(
            models=models,
            engine_dir=engine_dir,
            onnx_dir=onnx_dir,
        )

        onnx_exporters = self._get_onnx_exporters(models)

        # Export models to ONNX
        for model_name, model_exporter in onnx_exporters.items():
            self._export_onnx(
                model_exporter=model_exporter,
                model_config=model_configs[model_name],
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                onnx_opset=onnx_opset,
            )

        engines = {}

        # Build TensorRT engines
        for model_name, obj in onnx_exporters.items():
            model_config = model_configs[model_name]

            # TODO per model the proper class engine needs to be used
            engine_class = self.model_to_engine_class[model_name]
            engine = engine_class(
                engine_path=model_config["engine_path"],
                **obj.get_mixin_params(),
            )

            if not os.path.exists(model_config["engine_path"]):
                self._build_engine(
                    obj=obj,
                    engine=engine,
                    model_config=model_config,
                    opt_batch_size=opt_batch_size,
                    opt_image_height=opt_image_height,
                    opt_image_width=opt_image_width,
                    static_batch=self.static_batch,
                    optimization_level=optimization_level,
                    enable_all_tactics=enable_all_tactics,
                    timing_cache=timing_cache,
                    verbose=self.verbose,
                )

            engines[model_name] = engine

        return engines

    @staticmethod
    def calculate_max_device_memory(engines: dict[str, BaseEngine]) -> int:
        max_device_memory = 0
        for model_name, engine in engines.items():
            max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
        return max_device_memory

    def init_runtime(self):
        self.runtime = trt.Runtime(TRT_LOGGER)
        enter_fn = type(self.runtime).__enter__
        enter_fn(self.runtime)

    def stop_runtime(self):
        exit_fn = type(self.runtime).__exit__
        exit_fn(self.runtime, *sys.exc_info())
