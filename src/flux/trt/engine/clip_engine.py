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
from transformers import CLIPTokenizer

from flux.trt.engine import Engine
from flux.trt.trt_config import ClipConfig


class CLIPEngine(Engine):
    def __init__(self, trt_config: ClipConfig, stream: torch.cuda.Stream, **kwargs):
        super().__init__(trt_config=trt_config, stream=stream, **kwargs)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14",
            max_length=self.trt_config.text_maxlen,
        )

    @torch.inference_mode()
    def __call__(
        self,
        prompt: list[str],
    ) -> torch.Tensor:
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
            feed_dict = {"input_ids": feed_dict["input_ids"].to(dtype=self.get_dtype("input_ids"))}

            pooled_embeddings = self.infer(feed_dict)["pooled_embeddings"]

        return pooled_embeddings
