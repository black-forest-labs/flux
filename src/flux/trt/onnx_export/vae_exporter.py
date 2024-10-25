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

import torch
from math import ceil
from flux.modules.autoencoder import AutoEncoder
from flux.trt.onnx_export.base_exporter import BaseExporter
from flux.trt.mixin import VAEMixin


class VAEExporter(VAEMixin, BaseExporter):
    def __init__(
        self,
        model: AutoEncoder,
        fp16=False,
        tf32=True,
        bf16=False,
        max_batch=8,
        verbose=True,
        compression_factor=8,
    ):
        super().__init__(
            z_channels=model.params.z_channels,
            compression_factor=compression_factor,
            scale_factor=model.params.scale_factor,
            shift_factor=model.params.shift_factor,
            model=model,
            fp16=fp16,
            tf32=tf32,
            bf16=bf16,
            max_batch=max_batch,
            verbose=verbose,
        )

        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1360  # max image resolution: 1344x1344
        self.min_latent_shape = 2 * ceil(self.min_image_shape / (self.compression_factor * 2))
        self.max_latent_shape = 2 * ceil(self.max_image_shape / (self.compression_factor * 2))

        # set proper dtype
        self.prepare_model()

    def get_model(self) -> torch.nn.Module:
        self.model.forward = self.model.decode
        return self.model

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {
            "latent": {0: "B", 2: "H", 3: "W"},
            "images": {0: "B", 2: f"{self.compression_factor}H", 3: f"{self.compression_factor}W"},
        }

    def check_dims(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ) -> None | tuple[int, int]:
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
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
    ):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch

        latent_height, latent_width = self.check_dims(
            batch_size=batch_size,
            image_height=image_height,
            image_width=image_width,
        )

        return {
            "latent": [
                (min_batch, self.z_channels, latent_height, latent_width),
                (batch_size, self.z_channels, latent_height, latent_width),
                (max_batch, self.z_channels, latent_height, latent_width),
            ]
        }

    def get_sample_input(
        self,
        batch_size: int,
        opt_image_height: int,
        opt_image_width: int,
    ) -> torch.Tensor:
        latent_height, latent_width = self.check_dims(
            batch_size=batch_size,
            image_height=opt_image_height,
            image_width=opt_image_width,
        )

        return torch.randn(
            batch_size,
            self.z_channels,
            latent_height,
            latent_width,
            dtype=torch.float32,
            device=self.device,
        )
