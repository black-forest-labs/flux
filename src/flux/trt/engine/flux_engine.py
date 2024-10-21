from math import ceil

import torch

from flux.trt.engine.base_engine import BaseEngine
from flux.trt.mixin.flux_mixin import FluxMixin


class FluxEngine(FluxMixin, BaseEngine):
    def __init__(
        self,
        guidance_embed: bool,
        vec_in_dim: int,
        context_in_dim: int,
        in_channels: int,
        out_channels: int,
        compression_factor: int,
        engine_path: str,
    ):
        super().__init__(
            guidance_embed=guidance_embed,
            vec_in_dim=vec_in_dim,
            context_in_dim=context_in_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            compression_factor=compression_factor,
            engine_path=engine_path,
        )

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

    def check_dims(self, batch_size: int, image_height: int, image_width: int) -> tuple[int, int] | None:
        latent_height = 2 * ceil(image_height / (2 * self.compression_factor))
        latent_width = 2 * ceil(image_width / (2 * self.compression_factor))

        return (latent_height, latent_width)

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
            "img": (batch_size, (latent_height // 2) * (latent_width // 2), self.in_channels),
            "img_ids": (batch_size, (latent_height // 2) * (latent_width // 2), 3),
            "txt": (batch_size, 256, self.context_in_dim),
            "txt_ids": (batch_size, 256, 3),
            "timesteps": (batch_size,),
            "y": (batch_size, self.vec_in_dim),
            "latent": (batch_size, (latent_height // 2) * (latent_width // 2), self.out_channels),
        }

        if self.guidance_embed:
            shape_dict["guidance"] = (batch_size,)

        return shape_dict
