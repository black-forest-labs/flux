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

    def __init__(self, trt_config: TransformerConfig, stream: torch.cuda.Stream, **kwargs):
        super().__init__(trt_config=trt_config, stream=stream, **kwargs)

    @property
    def dd_to_flux(self):
        return TransformerEngine.__dd_to_flux__

    @property
    def flux_to_dd(self):
        return TransformerEngine.__flux_to_dd__

    @torch.inference_mode()
    def __call__(
        self,
        **kwargs,
    ) -> torch.Tensor:
        feed_dict = {}

        if self.trt_config.model_name == "flux-schnell":
            # remove guidance
            kwargs.pop("guidance")

        for tensor_name, tensor_value in kwargs.items():
            if tensor_name == "latent":
                continue
            dd_name = self.flux_to_dd[tensor_name]
            feed_dict[dd_name] = tensor_value.to(dtype=self.get_dtype(dd_name))

        # remove batch dim to match demo-diffusion
        feed_dict["img_ids"] = feed_dict["img_ids"][0]
        feed_dict["txt_ids"] = feed_dict["txt_ids"][0]

        latent = self.infer(feed_dict=feed_dict)["latent"]

        return latent
