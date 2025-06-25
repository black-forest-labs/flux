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
from transformers import T5Tokenizer

from flux.trt.engine import Engine
from flux.trt.trt_config import T5Config


class T5Engine(Engine):
    def __init__(
        self,
        trt_config: T5Config,
        stream: cudaStream_t,
    ):
        super().__init__(
            trt_config=trt_config,
            stream=stream,
        )
        self.tokenizer = T5Tokenizer.from_pretrained(
            "google/t5-v1_1-xxl",
            max_length=self.trt_config.text_maxlen,
        )

    def __call__(
        self,
        prompt: list[str],
    ) -> torch.Tensor:
        shape_dict = self.get_shape_dict(batch_size=len(prompt))
        self.allocate_buffers(shape_dict=shape_dict, device=self.device)

        with torch.inference_mode():
            feed_dict = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.trt_config.text_maxlen,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            feed_dict = {"input_ids": feed_dict["input_ids"].to(torch.int32)}

            text_embeddings = self.infer(feed_dict)["text_embeddings"].clone()

        return text_embeddings

    def get_shape_dict(
        self,
        batch_size: int,
    ) -> dict[str, tuple]:
        return {
            "input_ids": (batch_size, self.trt_config.text_maxlen),
            "text_embeddings": (batch_size, self.trt_config.text_maxlen, self.trt_config.hidden_size),
        }
