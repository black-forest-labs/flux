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

from flux.trt.engine.base_engine import BaseEngine
from flux.trt.mixin import VAEMixin


class VAEEngine(VAEMixin, BaseEngine):
    def __init__(
        self,
        z_channels: int,
        compression_factor: int,
        scale_factor: float,
        shift_factor: float,
        engine_path: str,
    ):
        super().__init__(
            z_channels=z_channels,
            compression_factor=compression_factor,
            scale_factor=scale_factor,
            shift_factor=shift_factor,
            engine_path=engine_path,
        )

    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode(x)


    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert z.device == self.tensors["latent"].device, "device mismatch | expected {}; actual {}".format(
            self.tensors["latent"].device,
            z.device,
        )

        assert z.dtype == self.tensors["latent"].dtype, "dtype mismatch | expected {}; actual {}".format(
            self.tensors["latent"].dtype,
            z.dtype,
        )
        z = z / self.scale_factor + self.shift_factor

        feed_dict = {"latent": z}
        images = self.infer(feed_dict=feed_dict)["images"].clone()
        return images


    def get_shape_dict(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ) -> dict[str, tuple]:
        latent_height, latent_width = self.get_latent_dim(
            image_height=image_height,
            image_width=image_width,
        )
        return {
            "latent": (batch_size, self.z_channels, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }
