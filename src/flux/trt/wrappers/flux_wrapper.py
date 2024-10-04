import torch
from math import ceil
from flux.model import Flux
from .base_wrapper import BaseWrapper


class FluxWrapper(BaseWrapper):
    def __init__(
        self,
        model: Flux,
        fp16=False,
        tf32=False,
        bf16=False,
        max_batch=16,
        verbose=True,
        compression_factor=8,
        build_strongly_typed=False,
    ):
        super().__init__(
            model=model,
            fp16=fp16,
            tf32=tf32,
            bf16=bf16,
            max_batch=max_batch,
            verbose=verbose,
        )
        self.compression_factor = compression_factor
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1360  # max image resolution: 1344x1344
        self.min_latent_shape = 2 * ceil(self.min_image_shape / (self.compression_factor * 2))
        self.max_latent_shape = 2 * ceil(self.max_image_shape / (self.compression_factor * 2))
        self.build_strongly_typed = build_strongly_typed

        # set proper dtype
        self.prepare_model()

    def get_input_names(self):
        return [
            "img",
            "img_ids",
            "txt",
            "txt_ids",
            "timesteps",
            "y",
            # "guidance",
        ]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        dynamic_axes = {
            "img": {0: "B", 1: "latent_dim"},
            "img_ids": {0: "B", 1: "latent_dim"},
            "txt": {0: "B"},
            "txt_ids": {0: "B"},
            "timesteps": {0: "B"},
            "y": {0: "B"},
            # "guidance": {0: "B"},
            "latent": {0: "B", 1: "latent_dim"},
        }
        return dynamic_axes

    def check_dims(self, batch_size: int, image_height: int, image_width: int) -> tuple[int, int] | None:
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % self.compression_factor == 0 or image_width % self.compression_factor == 0

        latent_height = 2 * ceil(image_height / (2 * self.compression_factor))
        latent_width = 2 * ceil(image_width / (2 * self.compression_factor))

        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_sample_input(
        self,
        batch_size: int,
        opt_image_height: int,
        opt_image_width: int,
        static_shape: bool,
    ) -> tuple:
        latent_height, latent_width = self.check_dims(
            batch_size=batch_size,
            image_height=opt_image_height,
            image_width=opt_image_width,
        )
        dtype = torch.float16 if self.fp16 else torch.float32

        return (
            torch.randn(
                batch_size,
                (latent_height // 2) * (latent_width // 2),
                self.model.params.in_channels,
                dtype=dtype,
                device=self.device,
            )
            * 0.002,
            torch.zeros(
                batch_size,
                (latent_height // 2) * (latent_width // 2),
                3,
                dtype=torch.float32,
                device=self.device,
            ),
            torch.randn(batch_size, 256, self.model.params.context_in_dim, dtype=dtype, device=self.device) * 0.002,
            torch.zeros(batch_size, 256, 3, dtype=torch.float32, device=self.device),
            torch.tensor(data=[1.0] * batch_size, dtype=dtype, device=self.device),
            torch.randn(batch_size, self.model.params.vec_in_dim, dtype=dtype, device=self.device),
            # torch.tensor(data=[3.5] * batch_size, dtype=dtype, device=self.device),
        )

    def get_model(self) -> torch.nn.Module:
        return self.model
