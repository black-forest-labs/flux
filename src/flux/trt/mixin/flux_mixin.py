from typing import Any
from math import ceil
from flux.trt.mixin.base_mixin import BaseMixin


class FluxMixin(BaseMixin):
    def __init__(
        self,
        guidance_embed: bool,
        vec_in_dim: int,
        context_in_dim: int,
        in_channels: int,
        out_channels: int,
        compression_factor: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.guidance_embed = guidance_embed
        self.vec_in_dim = vec_in_dim
        self.context_in_dim = context_in_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.compression_factor = compression_factor

    def get_mixin_params(self) -> dict[str, Any]:
        """helper class that return the parameters used for construction"""
        return {
            "guidance_embed": self.guidance_embed,
            "vec_in_dim": self.vec_in_dim,
            "context_in_dim": self.context_in_dim,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "compression_factor": self.compression_factor,
        }

    def get_latent_dims(
        self,
        image_height: int,
        image_width: int,
    ) -> tuple[int, int]:
        latent_height = 2 * ceil(image_height / (2 * self.compression_factor))
        latent_width = 2 * ceil(image_width / (2 * self.compression_factor))

        return (latent_height, latent_width)
