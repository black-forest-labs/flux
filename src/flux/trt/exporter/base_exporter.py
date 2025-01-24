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
import tempfile
from abc import ABC, abstractmethod
from typing import Any

import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
import torch
from onnx import shape_inference
from polygraphy.backend.onnx.loader import fold_constants
from polygraphy.backend.trt import (
    CreateConfig,
    ModifyNetworkOutputs,
    Profile,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.logger import G_LOGGER
from torch import Tensor, nn
from transformers import PreTrainedModel

from flux.model import Flux
from flux.modules.conditioner import HFEmbedder


class TransformersModelWrapper(torch.nn.Module):
    def __init__(self, model: HFEmbedder, output_name: str):
        super().__init__()
        self.model: PreTrainedModel = model.hf_module
        self.output_name = output_name

    def forward(self, input_ids: Tensor, *args):
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=None,
            output_hidden_states=False,
        )
        text_embeddings = outputs[self.output_name]
        return text_embeddings


class FluxModelWrapper(torch.nn.Module):
    def __init__(self, model: Flux):
        super().__init__()
        self.base_model = model

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        pooled_projections: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        # add batch dim to img_ids and txt_ids
        img_ids = torch.unsqueeze(img_ids, 0)
        txt_ids = torch.unsqueeze(txt_ids, 0)

        return self.base_model(
            img=hidden_states,
            img_ids=img_ids,
            txt=encoder_hidden_states,
            txt_ids=txt_ids,
            timesteps=timestep,
            y=pooled_projections,
            guidance=guidance,
        )


class Optimizer:
    def __init__(self, onnx_graph, verbose=False):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        return gs.export_onnx(self.graph) if return_onnx else self.graph

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(
            gs.export_onnx(self.graph),
            allow_onnxruntime_shape_inference=True,
            error_ok=False,
        )
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            temp_dir = tempfile.TemporaryDirectory().name
            os.makedirs(temp_dir, exist_ok=True)
            onnx_orig_path = os.path.join(temp_dir, "model.onnx")
            onnx_inferred_path = os.path.join(temp_dir, "inferred.onnx")
            onnx.save_model(
                onnx_graph,
                onnx_orig_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False,
            )
            onnx.shape_inference.infer_shapes_path(
                model_path=onnx_orig_path,
                output_path=onnx_inferred_path,
            )
            onnx_graph = onnx.load(onnx_inferred_path)
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def clip_add_hidden_states(self, hidden_layer_offset, return_onnx=False):
        hidden_layers = -1
        onnx_graph = gs.export_onnx(self.graph)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                name = onnx_graph.graph.node[i].output[j]
                if "layers" in name:
                    hidden_layers = max(int(name.split(".")[1].split("/")[0]), hidden_layers)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                if onnx_graph.graph.node[i].output[
                    j
                ] == "/text_model/encoder/layers.{}/Add_1_output_0".format(
                    hidden_layers + hidden_layer_offset
                ):
                    onnx_graph.graph.node[i].output[j] = "hidden_states"
            for j in range(len(onnx_graph.graph.node[i].input)):
                if onnx_graph.graph.node[i].input[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(
                    hidden_layers + hidden_layer_offset
                ):
                    onnx_graph.graph.node[i].input[j] = "hidden_states"
        if return_onnx:
            return onnx_graph


class BaseExporter(ABC):
    def __init__(
        self,
        model: nn.Module,
        fp16=False,
        tf32=False,
        bf16=False,
        max_batch=4,
        verbose=True,
        do_constant_folding=True,
    ):
        self.model = model
        self.name = model.__class__.__name__
        self.verbose = verbose
        self.do_constant_folding = do_constant_folding

        self.fp16 = fp16
        self.tf32 = tf32
        self.bf16 = bf16

        self.min_batch = 1
        self.max_batch = max_batch
        self.extra_output_names = []

    @property
    def device(self):
        return next(self.model.parameters()).device

    def prepare_model(self):
        if self.fp16:
            self.model = self.model.to(dtype=torch.float16)
        elif self.bf16:
            self.model = self.model.to(dtype=torch.bfloat16)
        else:
            self.model = self.model.to(dtype=torch.float32)

        self.model = self.model.eval().requires_grad_(False)

    def get_model(self) -> torch.nn.Module:
        return self.model

    @abstractmethod
    def get_sample_input(
        self,
        batch_size: int,
        opt_image_height: int,
        opt_image_width: int,
    ) -> tuple | torch.Tensor:
        pass

    @abstractmethod
    def get_input_names(self) -> list[str]:
        pass

    @abstractmethod
    def get_input_profile(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
        static_batch: bool,
        static_shape: bool,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_output_names(self) -> list[str]:
        pass

    @abstractmethod
    def get_dynamic_axes(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def check_dims(self, *args, **kwargs) -> None | tuple[int, int]:
        pass

    # Helper utility for ONNX export
    def export_onnx(
        self,
        onnx_path: str,
        onnx_opt_path: str,
        onnx_opset: int,
        opt_image_height: int,
        opt_image_width: int,
    ):
        onnx_opt_graph = None
        # Export optimized ONNX model (if missing)
        if not os.path.exists(onnx_opt_path):
            if not os.path.exists(onnx_path):
                print(f"[I] Exporting ONNX model: {onnx_path}")
                print("[I] model dtype: {}".format(next(self.model.parameters()).dtype))
                print(f"[I] fp16 = {self.fp16} | tf32 = {self.tf32} | bf16 = {self.bf16}")

                def export_onnx(model: torch.nn.Module):
                    inputs = self.get_sample_input(
                        self.min_batch,
                        opt_image_height,
                        opt_image_width,
                    )
                    torch.onnx.export(
                        model,
                        inputs,
                        onnx_path,
                        export_params=True,
                        opset_version=onnx_opset,
                        do_constant_folding=self.do_constant_folding,
                        input_names=self.get_input_names(),
                        output_names=self.get_output_names(),
                        dynamic_axes=self.get_dynamic_axes(),
                    )

                # WAR: Enable autocast for BF16 Stable Cascade pipeline
                with torch.inference_mode():
                    export_onnx(self.get_model())
            else:
                print(f"[I] Found cached ONNX model: {onnx_path}")

            print(f"[I] Optimizing ONNX model: {onnx_opt_path}")
            print("debuglog - loading the model to optimize")
            onnx_opt_graph = self.optimize(onnx.load(onnx_path))
            if onnx_opt_graph.ByteSize() > 2147483648:
                onnx.save_model(
                    onnx_opt_graph,
                    onnx_opt_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    convert_attribute=False,
                )
            else:
                onnx.save(onnx_opt_graph, onnx_opt_path)
        else:
            print(f"[I] Found cached optimized ONNX model: {onnx_opt_path} ")

    def optimize(self, onnx_graph, return_onnx=True, *args, **kwargs):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ": original")
        opt.cleanup()
        opt.info(self.name + ": cleanup")

        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")

        onnx_opt_graph = opt.cleanup(return_onnx=return_onnx)
        opt.info(self.name + ": finished")
        return onnx_opt_graph

    @staticmethod
    def build(
        engine_path: str,
        onnx_path: str,
        strongly_typed=False,
        fp16=False,
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
        print(f"Building TensorRT engine for {onnx_path}: {engine_path}")
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

        print(f"Strongly typed mode is {strongly_typed} for {onnx_path}")
        network = network_from_onnx_path(
            onnx_path,
            flags=flags,
            strongly_typed=strongly_typed,
        )
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
            save_engine(engine, path=engine_path)
