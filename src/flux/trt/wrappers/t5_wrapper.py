import torch
from transformers import T5EncoderModel
from flux.modules.conditioner import HFEmbedder
from .base_wrapper import BaseWrapper


class ExportT5EncoderModel(torch.nn.Module):
    def __init__(self, t5_encoder_model: T5EncoderModel):
        super().__init__()
        self.t5_encoder_model = t5_encoder_model

    def forward(self, input_ids, *args):
        outputs = self.t5_encoder_model.forward(
            input_ids=input_ids,
            attention_mask=None,
            output_hidden_states=False,
        )
        text_embeddings = outputs["last_hidden_state"]
        return text_embeddings

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
        self.text_maxlen = model.max_length
        exp_model = ExportT5EncoderModel(t5_encoder_model=model.hf_module)
        super().__init__(
            model=exp_model,
            fp16=fp16,
            tf32=tf32,
            bf16=bf16,
            max_batch=max_batch,
            verbose=verbose,
        )

        # set proper dtype
        self.prepare_model()

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
        return self.model
