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
from typing import Any
from flux.modules.conditioner import HFEmbedder
from flux.trt.trt_config.base_trt_config import TRTBaseConfig, register_config


@dataclass
class T5BaseConfig(TRTBaseConfig):
    text_maxlen: int | None = None
    hidden_size: int | None = None

    @classmethod
    def from_model(
        cls,
        model: HFEmbedder,
        **kwargs,
    ):
        return cls(
            text_maxlen=model.max_length,
            hidden_size=model.hf_module.config.hidden_size,
            **kwargs,
        )

    def check_dims(
        self,
        batch_size: int,
    ) -> None:
        assert batch_size >= self.min_batch and batch_size <= self.max_batch

    def get_input_profile(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
        static_batch: bool,
        static_shape: bool,
    ):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch

        self.check_dims(batch_size)
        return {
            "input_ids": [
                (min_batch, self.text_maxlen),
                (batch_size, self.text_maxlen),
                (max_batch, self.text_maxlen),
            ]
        }

    def get_engine_params(self) -> dict[str, Any]:
        """helper class that return the parameters used for construction"""
        return {
            "text_maxlen": self.text_maxlen,
            "hidden_size": self.hidden_size,
            "engine_path": self.engine_path,
        }


@register_config(model_name="t5", tf32=True, bf16=True, fp8=False, fp4=False)
@register_config(model_name="t5", tf32=True, bf16=False, fp8=True, fp4=False)
@register_config(model_name="t5", tf32=True, bf16=False, fp8=False, fp4=True)
@dataclass
class T5Config(T5BaseConfig):
    model_name: str = "t5"
    trt_tf32: bool = True
    trt_bf16: bool = False
    trt_fp8: bool = False
    trt_fp4: bool = False
    trt_build_strongly_typed: bool = True


@register_config(model_name="t5", tf32=True, bf16=True, fp8=False, fp4=False, t5_fp8=True)
@register_config(model_name="t5", tf32=True, bf16=False, fp8=True, fp4=False, t5_fp8=True)
@register_config(model_name="t5", tf32=True, bf16=False, fp8=False, fp4=True, t5_fp8=True)
@dataclass
class T5Fp8Config(T5BaseConfig):
    model_name: str = "t5"
    trt_tf32: bool = False
    trt_bf16: bool = True
    trt_fp8: bool = True
    trt_fp4: bool = False
    trt_build_strongly_typed: bool = False
