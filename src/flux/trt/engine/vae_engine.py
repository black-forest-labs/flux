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

from flux.trt.engine.base_engine import BaseEngine, Engine
from flux.trt.trt_config import VAEDecoderConfig, VAEEncoderConfig


class VAEDecoder(Engine):
    def __init__(self, trt_config: VAEDecoderConfig, stream: torch.cuda.Stream, **kwargs):
        super().__init__(trt_config=trt_config, stream=stream, **kwargs)

    @torch.inference_mode()
    def __call__(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        z = z.to(dtype=self.get_dtype("latent"))
        z = (z / self.trt_config.scale_factor) + self.trt_config.shift_factor
        feed_dict = {"latent": z}
        images = self.infer(feed_dict=feed_dict)["images"]
        return images


class VAEEncoder(Engine):
    def __init__(self, trt_config: VAEEncoderConfig, stream: torch.cuda.Stream, **kwargs):
        super().__init__(trt_config=trt_config, stream=stream, **kwargs)

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        feed_dict = {"images": x.to(dtype=self.get_dtype("images"))}
        latent = self.infer(feed_dict=feed_dict)["latent"]
        latent = self.trt_config.scale_factor * (latent - self.trt_config.shift_factor)
        return latent


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

    def cuda(self):
        self.decoder = self.decoder.cuda()
        if self.encoder is not None:
            self.encoder = self.encoder.cuda()
        return self

    def to(self, device):
        self.decoder = self.decoder.to(device)
        if self.encoder is not None:
            self.encoder = self.encoder.to(device)
        return self

    @property
    def device_memory_size(self):
        device_memory = self.decoder.device_memory_size
        if self.encoder is not None:
            device_memory = max(device_memory, self.encoder.device_memory_size)
        return device_memory
