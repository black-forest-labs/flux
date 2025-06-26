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
import warnings
from dataclasses import dataclass, field
from math import ceil

from huggingface_hub import snapshot_download

from flux.trt.trt_config.base_trt_config import ModuleName, TRTBaseConfig, register_config
from flux.util import PREFERED_KONTEXT_RESOLUTIONS, configs


@register_config(module_name=ModuleName.TRANSFORMER, precision="bf16")
@register_config(module_name=ModuleName.TRANSFORMER, precision="fp8")
@register_config(module_name=ModuleName.TRANSFORMER, precision="fp4")
@dataclass
class TransformerConfig(TRTBaseConfig):
    guidance_embed: bool | None = None
    vec_in_dim: int | None = None
    context_in_dim: int | None = None
    in_channels: int | None = None
    out_channels: int | None = None

    min_image_shape: int | None = None
    max_image_shape: int | None = None
    default_image_shape: int = 1024
    compression_factor: int = 8
    text_maxlen: int | None = None

    min_latent_dim: int = field(init=False)
    max_latent_dim: int = field(init=False)

    min_context_latent_dim: int = field(init=False)
    max_context_latent_dim: int = field(init=False)

    trt_tf32: bool = True
    trt_bf16: bool = False
    trt_fp8: bool = False
    trt_fp4: bool = False
    trt_build_strongly_typed: bool = True

    @classmethod
    def from_args(
        cls,
        model_name,
        **kwargs,
    ):
        if model_name == "flux-dev-kontext" and kwargs["trt_static_shape"]:
            warnings.warn("Flux-dev-Kontext does not support static shapes for the encoder.")
            kwargs["trt_static_shape"] = False

        if model_name == "flux-dev-kontext":
            min_image_shape = 1008
            max_image_shape = 1040
        else:
            min_image_shape = 768
            max_image_shape = 1360

        return cls(
            model_name=model_name,
            module_name=ModuleName.TRANSFORMER,
            guidance_embed=configs[model_name].params.guidance_embed,
            vec_in_dim=configs[model_name].params.vec_in_dim,
            context_in_dim=configs[model_name].params.context_in_dim,
            in_channels=configs[model_name].params.in_channels,
            out_channels=configs[model_name].params.out_channels,
            text_maxlen=256 if model_name == "flux-schnell" else 512,
            min_image_shape=min_image_shape,
            max_image_shape=max_image_shape,
            **kwargs,
        )

    def _get_onnx_path(self) -> str:
        if self.custom_onnx_path:
            return self.custom_onnx_path

        repo_id = self._get_repo_id(self.model_name)
        typed_model_path = os.path.join(f"{self.module_name.value}.opt", self.precision)

        snapshot_path = snapshot_download(repo_id, allow_patterns=[f"{typed_model_path}/*"])
        onnx_model_path = os.path.join(snapshot_path, typed_model_path, "model.onnx")
        return onnx_model_path

    @staticmethod
    def _get_latent(image_dim: int, compression_factor: int) -> int:
        return ceil(image_dim / (2 * compression_factor))

    @staticmethod
    def _get_context_dim(
        image_height: int,
        image_width: int,
        compression_factor: int,
    ) -> int:
        seq_len = TransformerConfig._get_latent(
            image_dim=image_height,
            compression_factor=compression_factor,
        ) * TransformerConfig._get_latent(
            image_dim=image_width,
            compression_factor=compression_factor,
        )

        return seq_len

    def __post_init__(self):
        min_latent_dim = TransformerConfig._get_context_dim(
            image_height=self.min_image_shape,
            image_width=self.min_image_shape,
            compression_factor=self.compression_factor,
        )

        max_latent_dim = TransformerConfig._get_context_dim(
            image_height=self.max_image_shape,
            image_width=self.max_image_shape,
            compression_factor=self.compression_factor,
        )

        if self.model_name == "flux-dev-kontext":
            # get min context size
            _, min_context_height, min_context_width = min(
                (w * h, w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS
            )
            self.min_context_latent_dim = TransformerConfig._get_context_dim(
                image_height=min_context_height,
                image_width=min_context_width,
                compression_factor=self.compression_factor,
            )

            # get max context size
            _, max_context_height, max_context_width = max(
                (w * h, w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS
            )
            self.max_context_latent_dim = TransformerConfig._get_context_dim(
                image_height=max_context_height,
                image_width=max_context_width,
                compression_factor=self.compression_factor,
            )
        else:
            self.min_context_latent_dim = 0
            self.max_context_latent_dim = 0

        self.min_latent_dim = min_latent_dim + self.min_context_latent_dim
        self.max_latent_dim = max_latent_dim + self.max_context_latent_dim

        super().__post_init__()

    def get_minmax_dims(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ):
        min_batch = batch_size if self.trt_static_batch else self.min_batch
        max_batch = batch_size if self.trt_static_batch else self.max_batch

        # if a model has context: it is always dynamic. target image can be static
        # or dynamic for every-model
        min_latent_dim = (
            self._get_context_dim(
                image_height=image_height,
                image_width=image_width,
                compression_factor=self.compression_factor,
            )
            + self.min_context_latent_dim
        )
        max_latent_dim = (
            self._get_context_dim(
                image_height=image_height,
                image_width=image_width,
                compression_factor=self.compression_factor,
            )
            + self.max_context_latent_dim
        )

        # static-shape affects only the target image size
        min_latent_dim = min_latent_dim if self.trt_static_shape else self.min_latent_dim
        max_latent_dim = max_latent_dim if self.trt_static_shape else self.max_latent_dim

        return (min_batch, max_batch, min_latent_dim, max_latent_dim)

    def check_dims(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ) -> int:
        self._check_batch(batch_size)
        assert (
            image_height % self.compression_factor == 0 or image_width % self.compression_factor == 0
        ), f"Image dimensions must be divisible by compression factor {self.compression_factor}"

        latent_dim = self._get_context_dim(
            image_height=image_height,
            image_width=image_width,
            compression_factor=self.compression_factor,
        )

        if self.model_name == "flux-dev-kontext":
            # for context models, it is assumed that the optimal context image shape is the same
            # as target image shape
            latent_dim = 2 * latent_dim

        assert self.min_latent_dim <= latent_dim <= self.max_latent_dim, "Image resolution out of boundaries."
        return latent_dim

    def get_input_profile(
        self,
        batch_size: int,
        image_height: int | None,
        image_width: int | None,
    ) -> dict[str, list[tuple]]:
        if self.model_name == "flux-dev-kontext":
            assert not self.trt_static_shape, "If Flux-dev-kontext then static_shape must be False."
        else:
            assert isinstance(image_height, int) and isinstance(
                image_width, int
            ), "Only Flux-dev-kontext allows None image shape"

        image_height = self.default_image_shape if image_height is None else image_height
        image_width = self.default_image_shape if image_width is None else image_width

        opt_latent_dim = self.check_dims(
            batch_size=batch_size,
            image_height=image_height,
            image_width=image_width,
        )

        (
            min_batch,
            max_batch,
            min_latent_dim,
            max_latent_dim,
        ) = self.get_minmax_dims(
            batch_size=batch_size,
            image_height=image_height,
            image_width=image_width,
        )

        input_profile = {
            "hidden_states": [
                (min_batch, min_latent_dim, self.in_channels),
                (batch_size, opt_latent_dim, self.in_channels),
                (max_batch, max_latent_dim, self.in_channels),
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
                (min_latent_dim, 3),
                (opt_latent_dim, 3),
                (max_latent_dim, 3),
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
