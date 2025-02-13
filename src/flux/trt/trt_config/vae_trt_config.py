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
from dataclasses import dataclass, field
from math import ceil
from typing import Any

from flux.modules.autoencoder import Decoder, Encoder
from flux.trt.trt_config.base_trt_config import TRTBaseConfig, register_config


@register_config(model_name="vae", tf32=True, bf16=True, fp8=False, fp4=False)
@register_config(model_name="vae", tf32=True, bf16=False, fp8=True, fp4=False)
@register_config(model_name="vae", tf32=True, bf16=False, fp8=False, fp4=True)
@dataclass
class VAEDecoderConfig(TRTBaseConfig):
    z_channels: int | None = None
    scale_factor: float | None = None
    shift_factor: float | None = None

    compression_factor: int = 8
    min_image_shape: int = 768
    max_image_shape: int = 1344
    min_latent_shape: int = field(init=False)
    max_latent_shape: int = field(init=False)
    model_name: str = "vae"
    trt_tf32: bool = True
    trt_bf16: bool = True
    trt_fp8: bool = False
    trt_fp4: bool = False
    trt_build_strongly_typed: bool = False

    @classmethod
    def from_model(
        cls,
        model: Decoder,
        **kwargs,
    ):
        return cls(
            z_channels=model.params.z_channels,
            scale_factor=model.params.scale_factor,
            shift_factor=model.params.shift_factor,
            **kwargs,
        )

    def _get_latent_dim_(self, image_dim: int) -> int:
        return 2 * ceil(image_dim / (2 * self.compression_factor))

    def _get_img_dim_(self, latent_dim: int) -> int:
        return latent_dim * self.compression_factor

    def __post_init__(self):
        self.min_latent_shape = self._get_latent_dim_(self.min_image_shape)
        self.max_latent_shape = self._get_latent_dim_(self.max_image_shape)
        super().__post_init__()

    def check_dims(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ) -> tuple[int, int]:
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % self.compression_factor == 0 or image_width % self.compression_factor == 0

        latent_height = self._get_latent_dim_(image_height)
        latent_width = self._get_latent_dim_(image_width)

        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

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

    def get_input_profile(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
        static_batch: bool,
        static_shape: bool,
    ):
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

        return {
            "latent": [
                (min_batch, self.z_channels, min_latent_height, min_latent_width),
                (batch_size, self.z_channels, latent_height, latent_width),
                (max_batch, self.z_channels, max_latent_height, max_latent_width),
            ]
        }

    def get_engine_params(self) -> dict[str, Any]:
        return {
            "z_channels": self.z_channels,
            "compression_factor": self.compression_factor,
            "scale_factor": self.scale_factor,
            "shift_factor": self.shift_factor,
        }


@register_config(model_name="vae_encoder", tf32=True, bf16=True, fp8=False, fp4=False)
@register_config(model_name="vae_encoder", tf32=True, bf16=False, fp8=True, fp4=False)
@register_config(model_name="vae_encoder", tf32=True, bf16=False, fp8=False, fp4=True)
@dataclass
class VAEEncoderConfig(TRTBaseConfig):
    z_channels: int | None = None
    scale_factor: float | None = None
    shift_factor: float | None = None

    compression_factor: int = 8
    min_image_shape: int = 768
    max_image_shape: int = 1344
    min_latent_shape: int = field(init=False)
    max_latent_shape: int = field(init=False)

    model_name: str = "vae_encoder"
    trt_tf32: bool = True
    trt_bf16: bool = True
    trt_fp8: bool = False
    trt_fp4: bool = False
    trt_build_strongly_typed: bool = False

    @classmethod
    def from_model(
        cls,
        model: Encoder,
        **kwargs,
    ):
        return cls(
            z_channels=model.params.z_channels,
            scale_factor=model.params.scale_factor,
            shift_factor=model.params.shift_factor,
            **kwargs,
        )

    def _get_latent_dim_(self, image_dim: int) -> int:
        return 2 * ceil(image_dim / (2 * self.compression_factor))

    def _get_img_dim_(self, latent_dim: int) -> int:
        return latent_dim * self.compression_factor

    def __post_init__(self):
        self.min_latent_shape = self._get_latent_dim_(self.min_image_shape)
        self.max_latent_shape = self._get_latent_dim_(self.max_image_shape)
        super().__post_init__()

    def check_dims(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ) -> tuple[int, int]:
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % self.compression_factor == 0 or image_width % self.compression_factor == 0

        latent_height = self._get_latent_dim_(image_height)
        latent_width = self._get_latent_dim_(image_width)

        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

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

        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape

        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
        )

    def get_input_profile(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
        static_batch: bool,
        static_shape: bool,
    ):
        self.check_dims(
            batch_size=batch_size,
            image_height=image_height,
            image_width=image_width,
        )

        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
        ) = self.get_minmax_dims(
            batch_size=batch_size,
            image_height=image_height,
            image_width=image_width,
            static_batch=static_batch,
            static_shape=static_shape,
        )

        return {
            "images": [
                (min_batch, 3, min_image_height, min_image_width),
                (batch_size, 3, image_height, image_width),
                (max_batch, 3, max_image_height, max_image_width),
            ],
        }
