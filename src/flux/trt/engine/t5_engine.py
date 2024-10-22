import torch

from flux.trt.engine.base_engine import BaseEngine
from flux.trt.mixin.t5_mixin import T5Mixin
from transformers import T5Tokenizer


class T5Engine(T5Mixin, BaseEngine):
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
        self.tokenizer = T5Tokenizer.from_pretrained(
            "google/t5-v1_1-xxl",
            max_length=self.text_maxlen,
        )

    def __call__(
        self,
        prompt: list[str],
    ) -> torch.Tensor:
        feed_dict = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.text_maxlen,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        feed_dict.pop("attention_mask")
        feed_dict["input_ids"] = feed_dict["input_ids"].to(torch.int32)

        text_embeddings = self.infer(feed_dict=dict(input_ids=batch_encoding["input_ids"]))["text_embeddings"].clone()
        return text_embeddings

    def get_shape_dict(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ) -> dict[str, tuple]:
        return {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.hidden_size),
        }
