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
from typing import Any

from flux.trt.mixin.base_mixin import BaseMixin


class TransformerMixin(BaseMixin):
    def __init__(
        self,
        guidance_embed: bool,
        vec_in_dim: int,
        context_in_dim: int,
        in_channels: int,
        out_channels: int,
        compression_factor: int,
        text_maxlen=512,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.guidance_embed = guidance_embed
        self.vec_in_dim = vec_in_dim
        self.context_in_dim = context_in_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.compression_factor = compression_factor
        self.text_maxlen = text_maxlen

    def get_mixin_params(self) -> dict[str, Any]:
        """helper class that return the parameters used for construction"""
        return {
            "guidance_embed": self.guidance_embed,
            "vec_in_dim": self.vec_in_dim,
            "context_in_dim": self.context_in_dim,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "compression_factor": self.compression_factor,
            "text_maxlen": self.text_maxlen,
        }

    def get_latent_dim(
        self,
        image_height: int,
        image_width: int,
    ) -> tuple[int, int]:
        latent_height = 2 * ceil(image_height / (2 * self.compression_factor))
        latent_width = 2 * ceil(image_width / (2 * self.compression_factor))

        return (latent_height, latent_width)
