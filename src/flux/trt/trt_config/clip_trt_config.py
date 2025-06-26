#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dataclasses import dataclass

from flux.trt.trt_config.base_trt_config import ModuleName, TRTBaseConfig, register_config
from flux.util import configs


@register_config(module_name=ModuleName.CLIP, precision="bf16")
@dataclass
class ClipConfig(TRTBaseConfig):
    text_maxlen: int | None = None
    hidden_size: int | None = None
    trt_tf32: bool = True
    trt_bf16: bool = False
    trt_fp8: bool = False
    trt_fp4: bool = False
    trt_build_strongly_typed: bool = True

    @classmethod
    def from_args(
        cls,
        model_name: str,
        **kwargs,
    ):
        return cls(
            text_maxlen=77,
            hidden_size=configs[model_name].params.vec_in_dim,
            model_name=model_name,
            module_name=ModuleName.CLIP,
            **kwargs,
        )

    def check_dims(self, batch_size: int) -> None:
        self._check_batch(batch_size)

    def get_input_profile(
        self,
        batch_size: int,
        image_height=None,
        image_width=None,
    ):
        min_batch = batch_size if self.trt_static_batch else self.min_batch
        max_batch = batch_size if self.trt_static_batch else self.max_batch

        self.check_dims(batch_size)
        return {
            "input_ids": [
                (min_batch, self.text_maxlen),
                (batch_size, self.text_maxlen),
                (max_batch, self.text_maxlen),
            ]
        }
