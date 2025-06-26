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
import warnings
from dataclasses import dataclass, field
from math import ceil

from flux.trt.trt_config.base_trt_config import ModuleName, TRTBaseConfig, register_config
from flux.util import configs


@dataclass
class VAEBaseConfig(TRTBaseConfig):
    z_channels: int | None = None
    scale_factor: float | None = None
    shift_factor: float | None = None

    default_image_shape: int = 1024
    compression_factor: int = 8
    min_image_shape: int | None = None
    max_image_shape: int | None = None

    min_latent_shape: int = field(init=False)
    max_latent_shape: int = field(init=False)

    def _get_latent_dim(self, image_dim: int) -> int:
        return 2 * ceil(image_dim / (2 * self.compression_factor))

    def __post_init__(self):
        self.min_latent_shape = self._get_latent_dim(self.min_image_shape)
        self.max_latent_shape = self._get_latent_dim(self.max_image_shape)
        super().__post_init__()

    def check_dims(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ) -> tuple[int, int]:
        self._check_batch(batch_size)
        assert (
            image_height % self.compression_factor == 0 or image_width % self.compression_factor == 0
        ), f"Image dimensions must be divisible by compression factor {self.compression_factor}"

        latent_height = self._get_latent_dim(image_height)
        latent_width = self._get_latent_dim(image_width)

        assert (
            self.min_latent_shape <= latent_height <= self.max_latent_shape
        ), f"Latent height {latent_height} must be between {self.min_latent_shape} and {self.max_latent_shape}"
        assert (
            self.min_latent_shape <= latent_width <= self.max_latent_shape
        ), f"Latent width {latent_width} must be between {self.min_latent_shape} and {self.max_latent_shape}"
        return latent_height, latent_width


@register_config(module_name=ModuleName.VAE, precision="bf16")
@dataclass
class VAEDecoderConfig(VAEBaseConfig):
    trt_tf32: bool = True
    trt_bf16: bool = True
    trt_fp8: bool = False
    trt_fp4: bool = False
    trt_build_strongly_typed: bool = False

    @classmethod
    def from_args(
        cls,
        model_name: str,
        **kwargs,
    ):
        if model_name == "flux-dev-kontext":
            min_image_shape = 672
            max_image_shape = 1568
        else:
            min_image_shape = 768
            max_image_shape = 1360

        return cls(
            model_name=model_name,
            module_name=ModuleName.VAE,
            z_channels=configs[model_name].ae_params.z_channels,
            scale_factor=configs[model_name].ae_params.scale_factor,
            shift_factor=configs[model_name].ae_params.shift_factor,
            min_image_shape=min_image_shape,
            max_image_shape=max_image_shape,
            **kwargs,
        )

    def get_minmax_dims(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ):
        min_batch = batch_size if self.trt_static_batch else self.min_batch
        max_batch = batch_size if self.trt_static_batch else self.max_batch

        latent_height = self._get_latent_dim(image_height)
        latent_width = self._get_latent_dim(image_width)

        min_latent_height = latent_height if self.trt_static_shape else self.min_latent_shape
        max_latent_height = latent_height if self.trt_static_shape else self.max_latent_shape
        min_latent_width = latent_width if self.trt_static_shape else self.min_latent_shape
        max_latent_width = latent_width if self.trt_static_shape else self.max_latent_shape

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
        image_height: int | None,
        image_width: int | None,
    ):
        assert self.model_name == "flux-dev-kontext" or (
            image_height is not None and image_width is not None
        ), "Only Flux-dev-kontext allows None image shape"

        assert not self.trt_static_shape or (
            image_height is not None and image_width is not None
        ), "If static_shape is True, image_height and image_width must be not None"

        image_height = self.default_image_shape if image_height is None else image_height
        image_width = self.default_image_shape if image_width is None else image_width

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
        )

        return {
            "latent": [
                (min_batch, self.z_channels, min_latent_height, min_latent_width),
                (batch_size, self.z_channels, latent_height, latent_width),
                (max_batch, self.z_channels, max_latent_height, max_latent_width),
            ]
        }


@register_config(module_name=ModuleName.VAE_ENCODER, precision="bf16")
@dataclass
class VAEEncoderConfig(VAEBaseConfig):
    trt_tf32: bool = True
    trt_bf16: bool = True
    trt_fp8: bool = False
    trt_fp4: bool = False
    trt_build_strongly_typed: bool = False

    @classmethod
    def from_args(cls, model_name: str, **kwargs):
        if model_name == "flux-dev-kontext" and kwargs["trt_static_shape"]:
            warnings.warn("Flux-dev-Kontext does not support static shapes for the encoder.")
            kwargs["trt_static_shape"] = False

        if model_name == "flux-dev-kontext":
            min_image_shape = 672
            max_image_shape = 1568
        else:
            min_image_shape = 768
            max_image_shape = 1360

        return cls(
            model_name=model_name,
            module_name=ModuleName.VAE_ENCODER,
            z_channels=configs[model_name].ae_params.z_channels,
            scale_factor=configs[model_name].ae_params.scale_factor,
            shift_factor=configs[model_name].ae_params.shift_factor,
            min_image_shape=min_image_shape,
            max_image_shape=max_image_shape,
            **kwargs,
        )

    def get_minmax_dims(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ):
        min_batch = batch_size if self.trt_static_batch else self.min_batch
        max_batch = batch_size if self.trt_static_batch else self.max_batch

        min_image_height = image_height if self.trt_static_shape else self.min_image_shape
        max_image_height = image_height if self.trt_static_shape else self.max_image_shape
        min_image_width = image_width if self.trt_static_shape else self.min_image_shape
        max_image_width = image_width if self.trt_static_shape else self.max_image_shape

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
        image_height: int | None,
        image_width: int | None,
    ):
        if self.model_name == "flux-dev-kontext":
            assert (
                not self.trt_static_shape
            ), "Flux-dev-kontext does not support dynamic shapes for the encoder."
        else:
            assert isinstance(image_height, int) and isinstance(
                image_width, int
            ), "Only Flux-dev-kontext allows None image shape"

        image_height = self.default_image_shape if image_height is None else image_height
        image_width = self.default_image_shape if image_width is None else image_width

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
        )

        return {
            "images": [
                (min_batch, 3, min_image_height, min_image_width),
                (batch_size, 3, image_height, image_width),
                (max_batch, 3, max_image_height, max_image_width),
            ],
        }
