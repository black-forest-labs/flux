from math import ceil
import torch
from cuda import cudart

from flux.trt.engine.base_engine import BaseEngine


class AEEngine(BaseEngine):
    def __init__(
        self,
        stream: cudart.cudaStream_t,
        engine_path: str,
        z_channels=16,
        compression_factor=8,
    ):
        super().__init__(stream, engine_path)
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
        latent_height = 2 * ceil(image_height / (2 * self.compression_factor))
        latent_width = 2 * ceil(image_width / (2 * self.compression_factor))

        return {
            "latent": (batch_size, self.z_channels, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }
