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
from dataclasses import dataclass, field

import tensorrt as trt

registry = {}


@dataclass
class BaseConfig:
    onnx_dir: str
    engine_dir: str
    verbose: bool
    precision: str
    onnx_path: str = field(init=False)
    engine_path: str = field(init=False)

    def __post_init__(self):
        self.onnx_path = self._get_onnx_path()
        self.engine_path = self._get_engine_path()
        assert os.path.isfile(self.onnx_path), "onnx_path do not exists: {}".format(self.onnx_path)

    def _get_onnx_path(self) -> str:
        onnx_model_dir = os.path.join(
            self.onnx_dir,
            self.model_name + ".opt",
        )
        return os.path.join(onnx_model_dir, "model.onnx")

    def _get_engine_path(self) -> str:
        return os.path.join(
            self.engine_dir,
            self.model_name + ".trt" + trt.__version__ + ".plan",
        )


def register_config(model_name: str, tf32=True, bf16=False, fp8=False, fp4=False, t5_fp8=False):
    """Decorator to register a configuration class with specific flag conditions."""

    def decorator(cls):
        if model_name == "t5":
            key = f"model={model_name}_tf32={tf32}_bf16={bf16}_fp8={fp8}_fp4={fp4}_t5-fp8={t5_fp8}"
        else:
            key = f"model={model_name}_tf32={tf32}_bf16={bf16}_fp8={fp8}_fp4={fp4}"
        registry[key] = cls
        return cls

    return decorator


def get_config(model_name: str, tf32=True, bf16=True, fp8=False, fp4=False, t5_fp8=False):
    """Retrieve the appropriate configuration instance based on current flags."""
    if model_name == "t5":
        key = f"model={model_name}_tf32={tf32}_bf16={bf16}_fp8={fp8}_fp4={fp4}_t5-fp8={t5_fp8}"
    else:
        key = f"model={model_name}_tf32={tf32}_bf16={bf16}_fp8={fp8}_fp4={fp4}"
    return registry[key]


@register_config(model_name="clip", tf32=True, bf16=True, fp8=False, fp4=False)
@register_config(model_name="clip", tf32=True, bf16=False, fp8=True, fp4=False)
@register_config(model_name="clip", tf32=True, bf16=False, fp8=False, fp4=True)
@dataclass
class ClipConfig(BaseConfig):
    model_name: str = "clip"
    tf32: bool = True
    bf16: bool = True
    fp8: bool = False
    fp4: bool = False
    build_strongly_typed: bool = False


@register_config(model_name="vae", tf32=True, bf16=True, fp8=False, fp4=False)
@register_config(model_name="vae", tf32=True, bf16=False, fp8=True, fp4=False)
@register_config(model_name="vae", tf32=True, bf16=False, fp8=False, fp4=True)
@dataclass
class VAEConfig(BaseConfig):
    model_name: str = "vae"
    tf32: bool = True
    bf16: bool = True
    fp8: bool = False
    fp4: bool = False
    build_strongly_typed: bool = False


@register_config(model_name="vae_encoder", tf32=True, bf16=True, fp8=False, fp4=False)
@register_config(model_name="vae_encoder", tf32=True, bf16=False, fp8=True, fp4=False)
@register_config(model_name="vae_encoder", tf32=True, bf16=False, fp8=False, fp4=True)
@dataclass
class VAEEncoderConfig(BaseConfig):
    model_name: str = "vae_encoder"
    tf32: bool = True
    bf16: bool = False
    fp8: bool = False
    fp4: bool = False
    build_strongly_typed: bool = False


@register_config(model_name="transformer", tf32=True, bf16=True, fp8=False, fp4=False)
@register_config(model_name="transformer", tf32=True, bf16=False, fp8=True, fp4=False)
@register_config(model_name="transformer", tf32=True, bf16=False, fp8=False, fp4=True)
@dataclass
class TransformerConfig(BaseConfig):
    model_name: str = "transformer"
    tf32: bool = True
    bf16: bool = False
    fp8: bool = False
    fp4: bool = False
    build_strongly_typed: bool = True

    def _get_onnx_path(self) -> str:
        return os.path.join(
            self.onnx_dir,
            self.model_name + ".opt",
            self.precision,
            "model.onnx",
        )

    def _get_engine_path(self) -> str:
        return os.path.join(
            self.engine_dir,
            f"{self.model_name}_{self.precision}.trt{trt.__version__}.plan",
        )


@register_config(model_name="t5", tf32=True, bf16=True, fp8=False, fp4=False)
@register_config(model_name="t5", tf32=True, bf16=True, fp8=True, fp4=False)
@register_config(model_name="t5", tf32=True, bf16=False, fp8=False, fp4=True)
@dataclass
class T5Config(BaseConfig):
    model_name: str = "t5"
    tf32: bool = True
    bf16: bool = False
    fp8: bool = False
    fp4: bool = False
    build_strongly_typed: bool = True


@register_config(model_name="t5", tf32=True, bf16=True, fp8=False, fp4=False, t5_fp8=True)
@register_config(model_name="t5", tf32=True, bf16=False, fp8=True, fp4=False, t5_fp8=True)
@register_config(model_name="t5", tf32=True, bf16=False, fp8=False, fp4=True, t5_fp8=True)
@dataclass
class T5Fp8Config(BaseConfig):
    model_name: str = "t5"
    tf32: bool = True
    bf16: bool = True
    fp8: bool = True
    fp4: bool = False
    build_strongly_typed: bool = False
