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
from cuda.cudart import cudaStream_t

from flux.trt.engine import Engine
from flux.trt.trt_config import TransformerConfig


class TransformerEngine(Engine):
    __dd_to_flux__ = {
        "hidden_states": "img",
        "img_ids": "img_ids",
        "encoder_hidden_states": "txt",
        "pooled_projections": "y",
        "txt_ids": "txt_ids",
        "timestep": "timesteps",
        "guidance": "guidance",
        "latent": "latent",
    }

    __flux_to_dd__ = {
        "img": "hidden_states",
        "img_ids": "img_ids",
        "txt": "encoder_hidden_states",
        "y": "pooled_projections",
        "txt_ids": "txt_ids",
        "timesteps": "timestep",
        "guidance": "guidance",
        "latent": "latent",
    }

    def __init__(
        self,
        trt_config: TransformerConfig,
        stream: cudaStream_t,
    ):
        super().__init__(
            trt_config=trt_config,
            stream=stream,
        )

    @property
    def dd_to_flux(self):
        return TransformerEngine.__dd_to_flux__

    @property
    def flux_to_dd(self):
        return TransformerEngine.__flux_to_dd__

    def __call__(
        self,
        **kwargs,
    ) -> torch.Tensor:
        img = kwargs["img"]
        shape_dict = self.get_shape_dict(
            batch_size=img.size(0),
            hidden_size=img.size(1),
        )
        self.allocate_buffers(shape_dict=shape_dict, device=self.device)

        with torch.inference_mode():
            feed_dict = {
                tensor_name: kwargs[self.dd_to_flux[tensor_name]].to(dtype=tensor_value.dtype)
                for tensor_name, tensor_value in self.tensors.items()
                if tensor_name != "latent"
            }

            # remove batch dim to match demo-diffusion
            feed_dict["img_ids"] = feed_dict["img_ids"][0]
            feed_dict["txt_ids"] = feed_dict["txt_ids"][0]

            latent = self.infer(feed_dict=feed_dict)["latent"].clone()

        return latent

    def get_shape_dict(
        self,
        batch_size: int,
        hidden_size: int,
    ) -> dict[str, tuple]:
        shape_dict = {
            "hidden_states": (batch_size, hidden_size, self.trt_config.in_channels),
            "encoder_hidden_states": (
                batch_size,
                self.trt_config.text_maxlen,
                self.trt_config.context_in_dim,
            ),
            "pooled_projections": (batch_size, self.trt_config.vec_in_dim),
            "timestep": (batch_size,),
            "img_ids": (hidden_size, 3),
            "txt_ids": (self.trt_config.text_maxlen, 3),
            "latent": (batch_size, hidden_size, self.trt_config.out_channels),
        }

        if self.trt_config.guidance_embed:
            shape_dict["guidance"] = (batch_size,)

        return shape_dict
