import torch
from flux.modules.conditioner import HFEmbedder
from .base_wrapper import BaseWrapper, Optimizer


class CLIPWrapper(BaseWrapper):
    def __init__(
        self,
        model: HFEmbedder,
        fp16=False,
        tf32=False,
        bf16=False,
        max_batch=16,
        verbose=True,
        output_hidden_states=False,
        keep_pooled_output=False,
    ):
        super().__init__(
            model=model,
            fp16=fp16,
            tf32=tf32,
            bf16=bf16,
            max_batch=max_batch,
            verbose=verbose,
        )

        self.text_maxlen = self.model.max_length
        self.keep_pooled_output = keep_pooled_output
        self.hidden_layer_offset = -1

        # Output the final hidden state
        if output_hidden_states:
            self.extra_output_names = ["hidden_states"]

        # set proper dtype
        self.prepare_model()

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        output_names = ["text_embeddings"]
        if self.keep_pooled_output:
            output_names += ["pooled_embeddings"]
        return output_names

    def get_dynamic_axes(self):
        dynamic_axes = {
            "input_ids": {0: "B"},
            "text_embeddings": {0: "B"},
        }
        if self.keep_pooled_output:
            dynamic_axes["pooled_embeddings"] = {0: "B"}
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

    def optimize(self, onnx_graph, return_onnx=True):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ": original")
        keep_outputs = [0, 1] if self.keep_pooled_output else [0]
        opt.select_outputs(keep_outputs)
        opt.cleanup()
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        opt.select_outputs(keep_outputs, names=self.get_output_names())  # rename network outputs
        opt.info(self.name + ": rename network output(s)")
        opt_onnx_graph = opt.cleanup(return_onnx=return_onnx)
        if "hidden_states" in self.extra_output_names:
            opt_onnx_graph = opt.clip_add_hidden_states(
                hidden_layer_offset=self.hidden_layer_offset,
                return_onnx=return_onnx,
            )
            opt.info(self.name + ": added hidden_states")
        opt.info(self.name + ": finished")
        return opt_onnx_graph
