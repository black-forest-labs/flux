from typing import Any
from math import ceil
from flux.trt.mixin.base_mixin import BaseMixin


class AEMixin(BaseMixin):
    def __init__(
        self,
        z_channels: int,
        compression_factor: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.z_channels = z_channels
        self.compression_factor = compression_factor

    def get_mixin_params(self) -> dict[str, Any]:
        """helper class that return the parameters used for construction"""
        return {
            "z_channels": self.z_channels,
            "compression_factor": self.compression_factor,
        }

    def get_latent_dim(
        self,
        image_height: int,
        image_width: int,
    ) -> tuple[int, int]:
        latent_height = 2 * ceil(image_height / (2 * self.compression_factor))
        latent_width = 2 * ceil(image_width / (2 * self.compression_factor))

        return (latent_height, latent_width)
