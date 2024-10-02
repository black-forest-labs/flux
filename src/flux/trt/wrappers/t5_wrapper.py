import torch
from flux.modules.conditioner import HFEmbedder
from .base_wrapper import BaseWrapper, Optimizer


class T5Wrapper(BaseWrapper):
    def __init__(
        self,
        model: HFEmbedder,
        fp16=False,
        tf32=False,
        bf16=False,
        max_batch=16,
        verbose=True,
    ):
        super().__init__(
            model=model,
            embedding_dim=model.hf_module.config.hidden_size,
            fp16=fp16,
            tf32=tf32,
            bf16=bf16,
            max_batch=max_batch,
            verbose=verbose,
        )

        self.text_maxlen = self.model.max_length
        self.hidden_layer_offset = -1

        # set proper dtype
        self.set_model_to_dtype()

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        output_names = ["text_embeddings"]
        return output_names

    def get_dynamic_axes(self):
        dynamic_axes = {
            "input_ids": {0: "B"},
            "text_embeddings": {0: "B"},
        }
        return dynamic_axes

    def check_dims(
        self,
        batch_size: int,
    ) -> None | tuple[int, int]:
        assert batch_size >= self.min_batch and batch_size <= self.max_batch

    def get_sample_input(
        self,
        batch_size: int,
        opt_image_height: int,
        opt_image_width: int,
        static_shape: bool,
    ) -> torch.Tensor:
        self.check_dims(batch_size)
        return torch.zeros(
            batch_size,
            self.text_maxlen,
            dtype=torch.int32,
            device=self.device,
        )

    def get_model(self) -> torch.nn.Module:
        return self.model.hf_module
