import torch

from flux.trt.engine.base_engine import BaseEngine
from flux.trt.mixin.ae_mixin import AEMixin


class AEEngine(AEMixin, BaseEngine):
    def __init__(
        self,
        z_channels: int,
        compression_factor: int,
        engine_path: str,
    ):
        super().__init__(
            z_channels=z_channels,
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

    def get_shape_dict(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ) -> dict[str, tuple]:
        latent_height, latent_width = self.get_latent_dim(
            image_height=image_height,
            image_width=image_width,
        )
        return {
            "latent": (batch_size, self.z_channels, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }
