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

from flux.trt.engine import BaseEngine
from flux.trt.mixin import CLIPMixin
from transformers import CLIPTokenizer


class CLIPEngine(CLIPMixin, BaseEngine):
    def __init__(
        self,
        text_maxlen: int,
        hidden_size: int,
        engine_path: str,
    ):
        super().__init__(
            text_maxlen=text_maxlen,
            hidden_size=hidden_size,
            engine_path=engine_path,
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14",
            max_length=self.text_maxlen,
        )

    def __call__(
        self,
        prompt: list[str],
    ) -> torch.Tensor:
        with torch.inference_mode():
            feed_dict = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.text_maxlen,
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
        image_height: int,
        image_width: int,
    ) -> dict[str, tuple]:
        return {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.hidden_size),
        }
