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
from cuda import cudart

from flux.trt.engine.base_engine import BaseEngine, Engine
from flux.trt.mixin import VAEMixin


class VAEDecoder(VAEMixin, Engine):
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
        z: torch.Tensor,
    ) -> torch.Tensor:
        shape_dict = self.get_shape_dict(
            batch_size=z.size(0),
            latent_height=z.size(2),
            latent_width=z.size(3),
        )
        self.allocate_buffers(shape_dict=shape_dict, device=self.device)

        z = z.to(dtype=self.tensors["latent"].dtype)
        z = (z / self.scale_factor) + self.shift_factor
        feed_dict = {"latent": z}
        images = self.infer(feed_dict=feed_dict)["images"].clone()
        return images

    def get_shape_dict(self, batch_size: int, latent_height: int, latent_width: int) -> dict[str, tuple]:
        image_height, image_width = self.get_img_dim(
            latent_height=latent_height,
            latent_width=latent_width,
        )
        return {
            "latent": (batch_size, self.z_channels, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }


class VAEEncoder(VAEMixin, Engine):
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
        shape_dict = self.get_shape_dict(
            batch_size=x.size(0),
            image_height=x.size(2),
            image_width=x.size(3),
        )
        self.allocate_buffers(shape_dict=shape_dict, device=self.device)

        feed_dict = {"images": x}
        latent = self.infer(feed_dict=feed_dict)["latent"].clone()
        latent = self.scale_factor * (latent - self.shift_factor)
        return latent

    def get_shape_dict(self, batch_size: int, image_height: int, image_width: int) -> dict[str, tuple]:
        latent_height, latent_width = self.get_latent_dim(
            image_height=image_height,
            image_width=image_width,
        )
        return {
            "images": (batch_size, 3, image_height, image_width),
            "latent": (batch_size, self.z_channels, latent_height, latent_width),
        }


class VAEEngine(BaseEngine):
    def __init__(
        self,
        decoder: VAEDecoder,
        encoder: VAEEncoder | None = None,
    ):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        assert self.encoder is not None, "An encoder is needed to encode an image"
        return self.encoder(x)

    def cpu(self):
        self.decoder = self.decoder.cpu()
        if self.encoder is not None:
            self.encoder = self.encoder.cpu()
        return self

    def get_device_memory(self):
        if self.encoder:
            device_memory = max(
                self.decoder.engine.device_memory_size,
                self.encoder.engine.device_memory_size,
            )
        else:
            device_memory = self.decoder.engine.device_memory_size
        return device_memory

    def to(self, device):
        self.load()

        device_memory = self.get_device_memory()
        _, self.decoder.shared_device_memory = cudart.cudaMalloc(device_memory)

        self.activate(device=device, device_memory=self.decoder.shared_device_memory)
        return self

    def set_stream(self, stream):
        self.decoder.set_stream(stream)
        if self.encoder is not None:
            self.encoder.set_stream(stream)

    def load(self):
        self.decoder.load()
        if self.encoder is not None:
            self.encoder.load()

    def activate(
        self,
        device: str,
        device_memory: int | None = None,
    ):
        self.decoder.activate(device=device, device_memory=device_memory)
        if self.encoder is not None:
            self.encoder.activate(device=device, device_memory=device_memory)
