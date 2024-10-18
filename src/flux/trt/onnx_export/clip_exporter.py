import torch
from flux.trt.onnx_export.base_exporter import (
    BaseExporter,
    Optimizer,
    TransformersModelWrapper,
)
from flux.modules.conditioner import HFEmbedder
from flux.trt.mixin.clip_mixin import CLIPMixin


class CLIPExporter(CLIPMixin, BaseExporter):
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
            text_maxlen=model.max_length,
            hidden_size=model.hf_module.config.hidden_size,
            model=TransformersModelWrapper(model=model, output_name="pooler_output"),
            fp16=fp16,
            tf32=tf32,
            bf16=bf16,
            max_batch=max_batch,
            verbose=verbose,
        )

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
            batch_size,
            self.text_maxlen,
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

    def optimize(self, onnx_graph, return_onnx=True):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ": original")
        keep_outputs = [0]
        opt.select_outputs(keep_outputs)
        opt.cleanup()
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        opt.select_outputs(keep_outputs, names=self.get_output_names())  # rename network outputs
        opt.info(self.name + ": rename network output(s)")
        opt_onnx_graph = opt.cleanup(return_onnx=return_onnx)
        opt.info(self.name + ": finished")
        return opt_onnx_graph
