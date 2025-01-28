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


class VAEMixin(BaseMixin):
    def __init__(
        self,
        z_channels: int,
        compression_factor: int,
        scale_factor: float,
        shift_factor: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.z_channels = z_channels
        self.compression_factor = compression_factor
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

    def get_mixin_params(self) -> dict[str, Any]:
        """helper class that return the parameters used for construction"""

        mixin_params = {
            "z_channels": self.z_channels,
            "compression_factor": self.compression_factor,
            "scale_factor": self.scale_factor,
            "shift_factor": self.shift_factor,
        }

        return mixin_params

    def get_latent_dim(
        self,
        image_height: int,
        image_width: int,
    ) -> tuple[int, int]:
        latent_height = 2 * ceil(image_height / (2 * self.compression_factor))
        latent_width = 2 * ceil(image_width / (2 * self.compression_factor))

        return (latent_height, latent_width)

    def get_img_dim(
        self,
        latent_height: int,
        latent_width: int,
    ) -> tuple[int, int]:
        image_height = latent_height * self.compression_factor
        image_width = latent_width * self.compression_factor

        return (image_height, image_width)
