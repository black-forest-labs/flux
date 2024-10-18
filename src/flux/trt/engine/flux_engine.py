from math import ceil

import torch

from flux.trt.engine.base_engine import BaseEngine


class FluxEngine(BaseEngine):
    def __init__(
        self,
        engine_path: str,
        z_channels=16,
        compression_factor=8,
    ):
        super().__init__(engine_path)
        self.z_channels = z_channels
        self.compression_factor = compression_factor

    def __call__(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        assert latent.device == self.tensors["latent"].device, "device mismatch | expected {}; actual {}".format(
            self.tensors["latent"].device,
            latent.device,
        )

        assert latent.dtype == self.tensors["latent"].dtype, "dtype mismatch | expected {}; actual {}".format(
            self.tensors["latent"].dtype,
            latent.dtype,
        )

        feed_dict = {"latent": latent}
        images = self.infer(feed_dict=feed_dict)["images"].clone()
        return images

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.__call__(z)

    def get_shape_dict(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ) -> dict[str, tuple]:
        latent_height, latent_width = self.check_dims(
            batch_size=batch_size,
            image_height=image_height,
            image_width=image_width,
        )

        shape_dict = {
            "img": (batch_size, (latent_height // 2) * (latent_width // 2), self.model.params.in_channels),
            "img_ids": (batch_size, (latent_height // 2) * (latent_width // 2), 3),
            "txt": (batch_size, 256, self.model.params.context_in_dim),
            "txt_ids": (batch_size, 256, 3),
            "timesteps": (batch_size,),
            "y": (batch_size, self.model.params.vec_in_dim),
            "latent": (batch_size, (latent_height // 2) * (latent_width // 2), self.model.out_channels),
        }

        if self.guidance_embed:
            shape_dict["guidance"] = (batch_size,)

        return shape_dict