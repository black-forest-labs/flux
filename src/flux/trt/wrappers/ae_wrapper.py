import torch
from math import ceil
from flux.modules.autoencoder import AutoEncoder
from .base_wrapper import BaseWrapper


class AEWrapper(BaseWrapper):
    def __init__(
        self,
        model: AutoEncoder,
        fp16=False,
        tf32=False,
        bf16=False,
        max_batch=16,
        verbose=True,
        compression_factor=8,
    ):
        super().__init__(
            model=model,
            fp16=fp16,
            tf32=tf32,
            bf16=bf16,
            max_batch=max_batch,
            verbose=verbose,
        )

        # TODO: fix compression_factor and model.param.z_channel
        self.compression_factor = compression_factor
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1360  # max image resolution: 1344x1344
        self.min_latent_shape = 2 * ceil(self.min_image_shape / (self.compression_factor * 2))
        self.max_latent_shape = 2 * ceil(self.max_image_shape / (self.compression_factor * 2))

        # set proper dtype
        self.set_model_to_dtype()

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {
            "latent": {0: "B", 2: "H", 3: "W"},
            "images": {0: "B", 2: f"{self.compression_factor}H", 3: f"{self.compression_factor}W"},
        }

    def check_dims(self, batch_size: int, image_height: int, image_width: int) -> None | tuple[int, int]:
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
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
    ) -> torch.Tensor:
        latent_height, latent_width = self.check_dims(
            batch_size=batch_size,
            image_height=opt_image_height,
            image_width=opt_image_width,
        )
        dtype = torch.float16 if self.fp16 else torch.float32

        return torch.randn(
            batch_size,
            self.model.params.z_channels,
            latent_height,
            latent_width,
            dtype=dtype,
            device=self.device,
        )
    def get_model(self) -> torch.nn.Module:
        self.model.forward = self.model.decode
        return self.model
