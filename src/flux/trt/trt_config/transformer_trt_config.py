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

import os
from dataclasses import dataclass, field
from math import ceil
from typing import Any

from tensorrt import __version__ as trt_version
from flux.model import Flux
from flux.trt.trt_config.base_trt_config import TRTBaseConfig, register_config


@register_config(model_name="transformer", tf32=True, bf16=True, fp8=False, fp4=False)
@register_config(model_name="transformer", tf32=True, bf16=False, fp8=True, fp4=False)
@register_config(model_name="transformer", tf32=True, bf16=False, fp8=False, fp4=True)
@dataclass
class TransformerConfig(TRTBaseConfig):
    guidance_embed: int | None = None
    vec_in_dim: int | None = None
    context_in_dim: int | None = None
    in_channels: int | None = None
    out_channels: int | None = None

    compression_factor: int = 8
    text_maxlen: int = 512
    min_image_shape: int = 768
    max_image_shape: int = 1344
    min_latent_shape: int = field(init=False)
    max_latent_shape: int = field(init=False)

    model_name: str = "transformer"
    trt_tf32: bool = True
    trt_bf16: bool = False
    trt_fp8: bool = False
    trt_fp4: bool = False
    trt_build_strongly_typed: bool = True

    @classmethod
    def from_model(
        cls,
        model: Flux,
        **kwargs,
    ):
        return cls(
            guidance_embed=model.params.guidance_embed,
            vec_in_dim=model.params.vec_in_dim,
            context_in_dim=model.params.context_in_dim,
            in_channels=model.params.in_channels,
            out_channels=model.out_channels,
            **kwargs,
        )

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
            f"{self.model_name}_{self.precision}.trt{trt_version}.plan",
        )

    def _get_latent_dim_(self, image_dim: int) -> int:
        return 2 * ceil(image_dim / (2 * self.compression_factor))

    def __post_init__(self):
        self.min_latent_shape = self._get_latent_dim_(self.min_image_shape)
        self.max_latent_shape = self._get_latent_dim_(self.max_image_shape)
        super().__post_init__()

    def get_minmax_dims(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
        static_batch: bool,
        static_shape: bool,
    ):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch

        latent_height = self._get_latent_dim_(image_height)
        latent_width = self._get_latent_dim_(image_width)
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape

        return (
            min_batch,
            max_batch,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )

    def get_latent_dim(
        self,
        image_height: int,
        image_width: int,
    ) -> tuple[int, int]:
        latent_height = 2 * ceil(image_height / (2 * self.compression_factor))
        latent_width = 2 * ceil(image_width / (2 * self.compression_factor))

        return (latent_height, latent_width)

    def check_dims(self, batch_size: int, image_height: int, image_width: int) -> tuple[int, int]:
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % self.compression_factor == 0 or image_width % self.compression_factor == 0

        latent_height = self._get_latent_dim_(image_height)
        latent_width = self._get_latent_dim_(image_width)

        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_input_profile(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
        static_batch: bool,
        static_shape: bool,
    ) -> dict[str, list[tuple]]:
        latent_height, latent_width = self.check_dims(
            batch_size=batch_size,
            image_height=image_height,
            image_width=image_width,
        )

        (
            min_batch,
            max_batch,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(
            batch_size=batch_size,
            image_height=image_height,
            image_width=image_width,
            static_batch=static_batch,
            static_shape=static_shape,
        )

        input_profile = {
            "hidden_states": [
                (min_batch, (min_latent_height // 2) * (min_latent_width // 2), self.in_channels),
                (batch_size, (latent_height // 2) * (latent_width // 2), self.in_channels),
                (max_batch, (max_latent_height // 2) * (max_latent_width // 2), self.in_channels),
            ],
            "encoder_hidden_states": [
                (min_batch, self.text_maxlen, self.context_in_dim),
                (batch_size, self.text_maxlen, self.context_in_dim),
                (max_batch, self.text_maxlen, self.context_in_dim),
            ],
            "pooled_projections": [
                (min_batch, self.vec_in_dim),
                (batch_size, self.vec_in_dim),
                (max_batch, self.vec_in_dim),
            ],
            "img_ids": [
                ((min_latent_height // 2) * (min_latent_width // 2), 3),
                ((latent_height // 2) * (latent_width // 2), 3),
                ((max_latent_height // 2) * (max_latent_width // 2), 3),
            ],
            "txt_ids": [
                (self.text_maxlen, 3),
                (self.text_maxlen, 3),
                (self.text_maxlen, 3),
            ],
            "timestep": [(min_batch,), (batch_size,), (max_batch,)],
        }

        if self.guidance_embed:
            input_profile["guidance"] = [(min_batch,), (batch_size,), (max_batch,)]

        return input_profile

    def get_engine_params(self) -> dict[str, Any]:
        """helper class that return the parameters used for construction"""
        return {
            "guidance_embed": self.guidance_embed,
            "vec_in_dim": self.vec_in_dim,
            "context_in_dim": self.context_in_dim,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "compression_factor": self.compression_factor,
            "text_maxlen": self.text_maxlen,
            "engine_path": self.engine_path,
        }
