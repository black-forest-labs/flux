import torch
from math import ceil
from flux.model import Flux
from flux.trt.onnx_export.base_exporter import BaseExporter


class FluxExporter(BaseExporter):
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

        self.guidance_embed = self.model.params.guidance_embed
        self.compression_factor = compression_factor
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1360  # max image resolution: 1344x1344
        self.min_latent_shape = 2 * ceil(self.min_image_shape / (self.compression_factor * 2))
        self.max_latent_shape = 2 * ceil(self.max_image_shape / (self.compression_factor * 2))
        self.build_strongly_typed = build_strongly_typed

        # set proper dtype
        self.prepare_model()

    def get_input_names(self):
        inputs = [
            "img",
            "img_ids",
            "txt",
            "txt_ids",
            "timesteps",
            "y",
        ]
        if self.guidance_embed:
            inputs.append("guidance")

        return inputs

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
        }
        if self.guidance_embed:
            dynamic_axes["guidance"] = {0: "B"}

        dynamic_axes["latent"] = {0: "B", 1: "latent_dim"}
        return dynamic_axes

    def check_dims(self, batch_size: int, image_height: int, image_width: int) -> tuple[int, int] | None:
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % self.compression_factor == 0 or image_width % self.compression_factor == 0

        latent_height = 2 * ceil(image_height / (2 * self.compression_factor))
        latent_width = 2 * ceil(image_width / (2 * self.compression_factor))

        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_input_profile(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ) -> dict[str, list[tuple]]:
        latent_height, latent_width = self.check_dims(
            batch_size=batch_size,
            image_height=image_height,
            image_width=image_width,
        )
        input_profile = {
            "img": [
                (self.min_batch, (latent_height // 2) * (latent_width // 2), self.model.params.in_channels),
                (batch_size, (latent_height // 2) * (latent_width // 2), self.model.params.in_channels),
                (self.max_batch, (latent_height // 2) * (latent_width // 2), self.model.params.in_channels),
            ],
            "img_ids": [
                (self.min_batch, (latent_height // 2) * (latent_width // 2), 3),
                (batch_size, (latent_height // 2) * (latent_width // 2), 3),
                (self.max_batch, (latent_height // 2) * (latent_width // 2), 3),
            ],
            "txt": [
                (self.min_batch, 256, self.model.params.context_in_dim),
                (batch_size, 256, self.model.params.context_in_dim),
                (self.max_batch, 256, self.model.params.context_in_dim),
            ],
            "txt_ids": [
                (self.min_batch, 256, 3),
                (batch_size, 256, 3),
                (self.max_batch, 256, 3),
            ],
            "timesteps": [(self.min_batch,), (batch_size,), (self.max_batch,)],
            "y": [
                (self.min_batch, self.model.params.vec_in_dim),
                (batch_size, self.model.params.vec_in_dim),
                (self.max_batch, self.model.params.vec_in_dim),
            ],
        }

        if self.guidance_embed:
            input_profile["guidance"] = [(self.min_batch,), (batch_size,), (self.max_batch,)]

        return input_profile

    def get_sample_input(
        self,
        batch_size: int,
        opt_image_height: int,
        opt_image_width: int,
    ) -> tuple:
        latent_height, latent_width = self.check_dims(
            batch_size=batch_size,
            image_height=opt_image_height,
            image_width=opt_image_width,
        )
        if self.fp16:
            dtype = torch.float16
        elif self.bf16:
            dtype = torch.bfloat16
        elif self.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            dtype = torch.float32
        else:
            dtype = torch.float32

        inputs = (
            torch.randn(
                batch_size,
                (latent_height // 2) * (latent_width // 2),
                self.model.params.in_channels,
                dtype=dtype,
                device=self.device,
            ),
            torch.zeros(
                batch_size,
                (latent_height // 2) * (latent_width // 2),
                3,
                dtype=torch.float32,
                device=self.device,
            ),
            torch.randn(batch_size, 256, self.model.params.context_in_dim, dtype=dtype, device=self.device) * 0.5,
            torch.zeros(batch_size, 256, 3, dtype=torch.float32, device=self.device),
            torch.tensor(data=[1.0] * batch_size, dtype=dtype, device=self.device),
            torch.randn(batch_size, self.model.params.vec_in_dim, dtype=dtype, device=self.device),
        )

        if self.guidance_embed:
            inputs = inputs + (torch.full((batch_size,), 3.5, dtype=dtype, device=self.device),)
        return inputs
