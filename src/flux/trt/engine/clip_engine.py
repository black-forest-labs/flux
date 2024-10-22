from math import ceil

import torch

from flux.trt.engine.base_engine import BaseEngine
from flux.trt.mixin.clip_mixin import CLIPMixin
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
        input_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.text_maxlen,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        feed_dict = {"input_ids": input_ids}
        text_embeddings = self.infer(feed_dict=feed_dict)["text_embeddings"].clone()
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
