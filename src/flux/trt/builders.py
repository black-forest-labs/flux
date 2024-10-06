import torch
import os
import tensorrt as trt
from typing import Any

from flux.modules.autoencoder import AutoEncoder
from flux.modules.conditioner import HFEmbedder
from flux.model import Flux
from flux.trt.wrappers import BaseWrapper, AEWrapper, CLIPWrapper, FluxWrapper, T5Wrapper, Engine


class TRTBuilder:
    __stages__ = ["clip", "t5", "transformer", "ae"]

    @property
    def stages(self) -> list[str]:
        return self.__stages__

    def __init__(
        self,
        flux_model: Flux,
        t5_model: HFEmbedder,
        clip_model: HFEmbedder,
        ae_model: AutoEncoder,
        max_batch=16,
        fp16=True,
        tf32=False,
        bf16=False,
        verbose=True,
        **kwargs,
    ):
        self.models = {
            "clip": CLIPWrapper(
                clip_model,
                max_batch=max_batch,
                fp16=fp16,
                tf32=tf32,
                bf16=bf16,
                verbose=verbose,
            ),
            "transformer": FluxWrapper(
                flux_model,
                max_batch=max_batch,
                fp16=fp16,
                tf32=tf32,
                bf16=bf16,
                verbose=verbose,
                compression_factor=kwargs.get("compression_factor", 8),
            ),
            "t5": T5Wrapper(
                t5_model,
                max_batch=max_batch,
                fp16=fp16,
                tf32=tf32,
                bf16=bf16,
                verbose=verbose,
            ),
            "ae": AEWrapper(
                ae_model,
                max_batch=max_batch,
                fp16=fp16,
                tf32=tf32,
                bf16=bf16,
                verbose=verbose,
                compression_factor=kwargs.get("compression_factor", 8),
            ),
        }
        self.verbose = verbose

        assert all(
            stage in self.models for stage in self.stages
        ), f"some stage is missing\n\tstages: {self.models.keys()}\n\tneeded stages: {self.stages}"

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

    def _prepare_model_configs(
        self,
        engine_dir: str,
        onnx_dir: str,
    ) -> dict[str, dict[str, Any]]:
        model_names = self.models.keys()
        configs = {}
        for model_name in model_names:
            config: dict[str, Any] = {
                "use_int8": False,
                "use_fp8": False,
            }
            config["model_suffix"] = ""

            config["onnx_path"] = self._get_onnx_path(
                model_name=model_name,
                onnx_dir=onnx_dir,
                opt=False,
                suffix=config["model_suffix"],
            )
            config["onnx_opt_path"] = self._get_onnx_path(
                model_name=model_name,
                onnx_dir=onnx_dir,
                suffix=config["model_suffix"],
            )
            config["engine_path"] = self._get_engine_path(
                model_name=model_name,
                engine_dir=engine_dir,
                suffix=config["model_suffix"],
            )
            config["state_dict_path"] = self._get_state_dict_path(
                model_name=model_name,
                onnx_dir=onnx_dir,
                suffix=config["model_suffix"],
            )

            configs[model_name] = config

        return configs

    def _export_onnx(
        self,
        obj: BaseWrapper,
        model_config: dict[str, Any],
        opt_image_height: int,
        opt_image_width: int,
        static_shape: bool,
        onnx_opset: int,
        quantization_level: float,
        quantization_percentile: float,
        quantization_alpha: float,
        calibration_size: int,
        calib_batch_size: int,
    ):
        do_export_onnx = not os.path.exists(model_config["engine_path"]) and not os.path.exists(
            model_config["onnx_opt_path"]
        )

        if do_export_onnx:
            obj.export_onnx(
                model_config["onnx_path"],
                model_config["onnx_opt_path"],
                onnx_opset,
                opt_image_height,
                opt_image_width,
                static_shape=static_shape,
            )

    def _build_engine(
        self,
        obj: BaseWrapper,
        engine: Engine,
        model_config: dict[str, Any],
        opt_batch_size: int,
        opt_image_height: int,
        opt_image_width: int,
        optimization_level: int,
        static_batch: bool,
        static_shape: bool,
        enable_all_tactics: bool,
        timing_cache,
    ):
        update_output_names = obj.get_output_names() + obj.extra_output_names if obj.extra_output_names else None
        fp16amp = False if getattr(obj, "build_strongly_typed", False) else obj.fp16
        tf32amp = obj.tf32
        bf16amp = False if getattr(obj, "build_strongly_typed", False) else obj.bf16
        strongly_typed = True if getattr(obj, "build_strongly_typed", False) else False

        fp16amp = obj.fp16
        tf32amp = obj.tf32
        bf16amp = obj.bf16
        strongly_typed = False
        extra_build_args = {"verbose": self.verbose}
        extra_build_args["builder_optimization_level"] = optimization_level

        engine.build(
            model_config["onnx_opt_path"],
            strongly_typed=strongly_typed,
            fp16=fp16amp,
            tf32=tf32amp,
            bf16=bf16amp,
            input_profile=obj.get_input_profile(
                opt_batch_size,
                opt_image_height,
                opt_image_width,
                static_batch=static_batch,
                static_shape=static_shape,
            ),
            enable_all_tactics=enable_all_tactics,
            timing_cache=timing_cache,
            update_output_names=update_output_names,
            **extra_build_args,
        )

    def load_engines(
        self,
        engine_dir: str,
        onnx_dir: str,
        onnx_opset: int,
        opt_batch_size: int,
        opt_image_height: int,
        opt_image_width: int,
        optimization_level=3,
        static_batch=False,
        static_shape=True,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        int8=False,
        fp8=False,
        quantization_level=2.5,
        quantization_percentile=1.0,
        quantization_alpha=0.8,
        calibration_size=32,
        calib_batch_size=2,
    ):
        self._create_directories(
            engine_dir=engine_dir,
            onnx_dir=onnx_dir,
        )

        model_configs = self._prepare_model_configs(
            engine_dir=engine_dir,
            onnx_dir=onnx_dir,
        )

        # Export models to ONNX
        for model_name, obj in self.models.items():
            self._export_onnx(
                obj,
                model_config=model_configs[model_name],
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                static_shape=static_shape,
                onnx_opset=onnx_opset,
                quantization_level=quantization_level,
                quantization_percentile=quantization_percentile,
                quantization_alpha=quantization_alpha,
                calibration_size=calibration_size,
                calib_batch_size=calib_batch_size,
            )

        # Build TensorRT engines
        for model_name, obj in self.models.items():

            model_config = model_configs[model_name]
            engine = Engine(model_config["engine_path"])
            if not os.path.exists(model_config["engine_path"]):
                self._build_engine(
                    obj,
                    engine,
                    model_config,
                    opt_batch_size,
                    opt_image_height,
                    opt_image_width,
                    optimization_level,
                    static_batch,
                    static_shape,
                    enable_all_tactics,
                    timing_cache,
                )
            self.engine[model_name] = engine
