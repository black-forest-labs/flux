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

from math import ceil

import torch

from flux.model import Flux
from flux.trt.exporter.base_exporter import BaseExporter, FluxModelWrapper
from flux.trt.mixin import TransformerMixin


class TransformerExporter(TransformerMixin, BaseExporter):
    def __init__(
        self,
        model: Flux,
        fp16=False,
        tf32=True,
        bf16=False,
        max_batch=8,
        verbose=True,
        compression_factor=8,
        build_strongly_typed=True,
    ):
        super().__init__(
            guidance_embed=model.params.guidance_embed,
            vec_in_dim=model.params.vec_in_dim,
            context_in_dim=model.params.context_in_dim,
            in_channels=model.params.in_channels,
            out_channels=model.out_channels,
            compression_factor=compression_factor,
            model=FluxModelWrapper(model),
            fp16=fp16,
            tf32=tf32,
            bf16=bf16,
            max_batch=max_batch,
            verbose=verbose,
        )

        self.min_image_shape = 768
        self.max_image_shape = 1344
        self.min_latent_shape = 2 * ceil(self.min_image_shape / (self.compression_factor * 2))
        self.max_latent_shape = 2 * ceil(self.max_image_shape / (self.compression_factor * 2))
        self.build_strongly_typed = build_strongly_typed

        # set proper dtype
        self.prepare_model()

    def get_input_names(self):
        inputs = [
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
            "img_ids",
            "txt_ids",
        ]
        if self.guidance_embed:
            inputs.append("guidance")

        return inputs

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        dynamic_axes = {
            "hidden_states": {0: "B", 1: "latent_dim"},
            "encoder_hidden_states": {0: "B"},
            "pooled_projections": {0: "B"},
            "timestep": {0: "B"},
            "img_ids": {0: "latent_dim"},
        }
        if self.guidance_embed:
            dynamic_axes["guidance"] = {0: "B"}

        # dynamic_axes["latent"] = {0: "B", 1: "latent_dim"}
        return dynamic_axes

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

        latent_height = image_height // self.compression_factor
        latent_width = image_width // self.compression_factor
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

    def check_dims(self, batch_size: int, image_height: int, image_width: int) -> tuple[int, int]:
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % self.compression_factor == 0 or image_width % self.compression_factor == 0

        latent_height, latent_width = self.get_latent_dim(
            image_height=image_height,
            image_width=image_width,
        )

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

    def get_sample_input(
        self,
        batch_size: int,
        opt_image_height: int,
        opt_image_width: int,
    ) -> tuple:
        latent_height, latent_width = self.check_dims(
            batch_size=batch_size,
            image_height=opt_image_height,
            image_width=opt_image_width,
        )
        if self.fp16:
            dtype = torch.float16
        elif self.bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        inputs = (
            torch.randn(
                batch_size,
                (latent_height // 2) * (latent_width // 2),
                self.in_channels,
                dtype=dtype,
                device=self.device,
            ),
            torch.randn(batch_size, self.text_maxlen, self.context_in_dim, dtype=dtype, device=self.device)
            * 0.5,
            torch.randn(batch_size, self.vec_in_dim, dtype=dtype, device=self.device),
            torch.tensor(data=[1.0] * batch_size, dtype=dtype, device=self.device),
            torch.zeros(
                (latent_height // 2) * (latent_width // 2), 3, dtype=torch.float32, device=self.device
            ),
            torch.zeros(self.text_maxlen, 3, dtype=torch.float32, device=self.device),
        )

        if self.guidance_embed:
            inputs = inputs + (torch.full((batch_size,), 3.5, dtype=dtype, device=self.device),)
        return inputs
