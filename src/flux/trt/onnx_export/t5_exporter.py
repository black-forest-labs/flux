import torch

from flux.modules.conditioner import HFEmbedder
from flux.trt.onnx_export.base_exporter import BaseExporter, TransformersModelWrapper
from flux.trt.mixin.t5_mixin import T5Mixin


class T5Exporter(T5Mixin, BaseExporter):
    def __init__(
        self,
        model: HFEmbedder,
        fp16=False,
        tf32=False,
        bf16=False,
        max_batch=8,
        verbose=True,
    ):
        exp_model = TransformersModelWrapper(model=model, output_name="last_hidden_state")
        super().__init__(
            text_maxlen=model.max_length,
            hidden_size=model.hf_module.config.hidden_size,
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
        return ["text_embeddings"]

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
    ) -> torch.Tensor:
        self.check_dims(batch_size)
        return torch.zeros(
            (batch_size, self.text_maxlen),
            dtype=torch.int32,
            device=self.device,
        )

    def get_input_profile(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ):
        self.check_dims(batch_size)
        return {
            "input_ids": [
                (self.min_batch, self.text_maxlen),
                (batch_size, self.text_maxlen),
                (self.max_batch, self.text_maxlen),
            ]
        }
