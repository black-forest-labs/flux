import os
import re
import tempfile
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference

from onnxconverter_common.float16 import convert_float_to_float16
from polygraphy.backend.onnx.loader import fold_constants


from .utils_modelopt import (
    convert_zp_fp8,
    cast_resize_io,
    convert_fp16_io,
    cast_fp8_mha_io,
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
                if onnx_graph.graph.node[i].output[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(
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

    def fuse_mha_qkv_int8_sq(self):
        tensors = self.graph.tensors()
        keys = tensors.keys()

        # mha  : fuse QKV QDQ nodes
        # mhca : fuse KV QDQ nodes
        q_pat = (
            "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
            ".\\d+/attn\\d+/to_q/input_quantizer/DequantizeLinear_output_0"
        )
        k_pat = (
            "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
            ".\\d+/attn\\d+/to_k/input_quantizer/DequantizeLinear_output_0"
        )
        v_pat = (
            "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
            ".\\d+/attn\\d+/to_v/input_quantizer/DequantizeLinear_output_0"
        )

        qs = sorted(
            (
                x.group(0)
                for x in filter(
                    lambda x: x is not None,
                    [re.match(q_pat, key) for key in keys],
                )
            )
        )
        ks = sorted(
            (
                x.group(0)
                for x in filter(
                    lambda x: x is not None,
                    [re.match(k_pat, key) for key in keys],
                )
            )
        )
        vs = sorted(
            (
                x.group(0)
                for x in filter(
                    lambda x: x is not None,
                    [re.match(v_pat, key) for key in keys],
                )
            )
        )

        removed = 0
        assert len(qs) == len(ks) == len(vs), "Failed to collect tensors"
        for q, k, v in zip(qs, ks, vs):
            is_mha = all("attn1" in tensor for tensor in [q, k, v])
            is_mhca = all("attn2" in tensor for tensor in [q, k, v])
            assert (is_mha or is_mhca) and (not (is_mha and is_mhca))

            if is_mha:
                tensors[k].outputs[0].inputs[0] = tensors[q]
                tensors[v].outputs[0].inputs[0] = tensors[q]
                del tensors[k]
                del tensors[v]
                removed += 2
            else:  # is_mhca
                tensors[k].outputs[0].inputs[0] = tensors[v]
                del tensors[k]
                removed += 1
        print(f"Removed {removed} QDQ nodes")
        return removed  # expected 72 for L2.5

    def modify_fp8_graph(self):
        onnx_graph = gs.export_onnx(self.graph)
        # Convert INT8 Zero to FP8.
        onnx_graph = convert_zp_fp8(onnx_graph)
        # Convert weights and activations to FP16 and insert Cast nodes in FP8 MHA.
        onnx_graph = convert_float_to_float16(onnx_graph, keep_io_types=True, disable_shape_infer=True)
        self.graph = gs.import_onnx(onnx_graph)
        # Add cast nodes to Resize I/O.
        cast_resize_io(self.graph)
        # Convert model inputs and outputs to fp16 I/O.
        convert_fp16_io(self.graph)
        # Add cast nodes to MHA's BMM1 and BMM2's I/O.
        cast_fp8_mha_io(self.graph)


class BaseWrapper(ABC):
    def __init__(
        self,
        model: nn.Module,
        fp16=False,
        tf32=False,
        bf16=False,
        max_batch=16,
        verbose=True,
        do_constant_folding=True,
    ):
        self.model = model
        self.name = model.__class__.__name__
        self.device = next(model.parameters()).device
        self.verbose = verbose
        self.do_constant_folding = do_constant_folding

        self.fp16 = fp16
        self.tf32 = tf32
        self.bf16 = bf16

        self.min_batch = 1
        self.max_batch = max_batch
        self.extra_output_names = []

        assert sum([self.fp16, self.bf16, self.tf32]) <= 1, "too many dtype specified. only one is allowed"

    def prepare_model(self):
        if self.fp16:
            self.model = self.model.to(dtype=torch.float16)
        elif self.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.model = self.model.to(dtype=torch.float32)
        elif self.bf16:
            self.model = self.model.to(dtype=torch.bfloat16)
        else:
            self.model = self.model.to(dtype=torch.float32)

        self.model = self.model.eval().requires_grad_(False)

    @abstractmethod
    def get_sample_input(
        self,
        batch_size: int,
        opt_image_height: int,
        opt_image_width: int,
        static_shape: bool,
    ) -> tuple | torch.Tensor:
        pass

    @abstractmethod
    def get_input_names(self) -> list[str]:
        pass

    @abstractmethod
    def get_output_names(self) -> list[str]:
        pass

    @abstractmethod
    def get_dynamic_axes(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def check_dims(self, *args) -> None | tuple[int, int]:
        pass

    @abstractmethod
    def get_model(self) -> torch.nn.Module:
        pass

    # Helper utility for ONNX export
    def export_onnx(
        self,
        onnx_path: str,
        onnx_opt_path: str,
        onnx_opset: int,
        opt_image_height: int,
        opt_image_width: int,
        static_shape: bool = False,
    ):
        onnx_opt_graph = None
        # Export optimized ONNX model (if missing)
        if not os.path.exists(onnx_opt_path):
            if not os.path.exists(onnx_path):
                print(f"[I] Exporting ONNX model: {onnx_path}")

                def export_onnx(model: torch.nn.Module):
                    inputs = self.get_sample_input(1, opt_image_height, opt_image_width, static_shape)
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

                with torch.inference_mode(), torch.autocast("cuda"):
                    export_onnx(self.get_model())
            else:
                print(f"[I] Found cached ONNX model: {onnx_path}")

            print(f"[I] Optimizing ONNX model: {onnx_opt_path}")
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
        if kwargs.get("modify_fp8_graph", False):
            opt.modify_fp8_graph()
            opt.info(self.name + ": modify fp8 graph")
        else:
            opt.fold_constants()
            opt.info(self.name + ": fold constants")
            opt.infer_shapes()
            opt.info(self.name + ": shape inference")
            if kwargs.get("fuse_mha_qkv_int8", False):
                opt.fuse_mha_qkv_int8_sq()
                opt.info(self.name + ": fuse QKV nodes")
        onnx_opt_graph = opt.cleanup(return_onnx=return_onnx)
        opt.info(self.name + ": finished")
        return onnx_opt_graph
