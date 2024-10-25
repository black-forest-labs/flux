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
from flux.trt.mixin import TransformerMixin


class TransformerEngine(TransformerMixin, BaseEngine):
    def __init__(
        self,
        guidance_embed: bool,
        vec_in_dim: int,
        context_in_dim: int,
        in_channels: int,
        out_channels: int,
        compression_factor: int,
        text_maxlen: int,
        engine_path: str,
    ):
        super().__init__(
            guidance_embed=guidance_embed,
            vec_in_dim=vec_in_dim,
            context_in_dim=context_in_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            compression_factor=compression_factor,
            text_maxlen=text_maxlen,
            engine_path=engine_path,
        )

    def __call__(
        self,
        **kwargs,
    ) -> torch.Tensor:
        feed_dict = {
            tensor_name: kwargs[tensor_name].to(dtype=tensor_value.dtype)
            for tensor_name, tensor_value in self.tensors.items()
            if tensor_name != "latent"
        }

        latent = self.infer(feed_dict=feed_dict)["latent"].clone()
        return latent

    def get_shape_dict(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ) -> dict[str, tuple]:
        latent_height, latent_width = self.get_latent_dims(
            image_height=image_height,
            image_width=image_width,
        )

        shape_dict = {
            "hidden_states": (batch_size, (latent_height // 2) * (latent_width // 2), self.in_channels),
            "encoder_hidden_states": (batch_size, self.text_maxlen, self.context_in_dim),
            "pooled_projections": (batch_size, self.vec_in_dim),
            "timestep": (batch_size,),
            "img_ids": ((latent_height // 2) * (latent_width // 2), 3),
            "txt_ids": (self.text_maxlen, 3),
            "latent": (batch_size, (latent_height // 2) * (latent_width // 2), self.out_channels),
        }

        if self.guidance_embed:
            shape_dict["guidance"] = (batch_size,)

        return shape_dict
