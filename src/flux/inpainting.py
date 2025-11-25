"""
FLUX.2-dev Inpainting Module

Extends FLUX.2-dev with native inpainting capabilities using mask conditioning.
"""

import io
import os
from typing import Optional, Union, Tuple

import numpy as np
import torch
import requests
from PIL import Image
from huggingface_hub import get_token


class Flux2Inpainter:
    """
    FLUX.2-dev inpainting generator using native mask conditioning.

    This class attempts to use FLUX.2-dev's built-in inpainting capabilities.
    If native support isn't available, it falls back to FluxInpaintPipeline.

    Usage:
        inpainter = Flux2Inpainter()
        result = inpainter.inpaint(
            image=source_image,
            mask=binary_mask,
            prompt="a red sports car"
        )
    """

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.2-dev",
        device: str = "cuda",
        use_remote_encoder: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize FLUX.2-dev inpainter.

        Args:
            model_name: HuggingFace model ID for FLUX.2-dev
            device: Device to run on ("cuda" or "cpu")
            use_remote_encoder: Whether to use remote text encoder (saves VRAM)
            torch_dtype: Torch dtype for model (bfloat16 recommended)
        """
        self.device = device
        self.model_name = model_name
        self.use_remote_encoder = use_remote_encoder
        self.torch_dtype = torch_dtype
        self.token = get_token()

        self.pipe = None
        self.pipe_type = None  # Track which pipeline we're using

        self._load_pipeline()

    def _load_pipeline(self):
        """Load the appropriate pipeline for inpainting."""
        # Try loading in order of preference:
        # 1. Flux2InpaintPipeline (if it exists for FLUX.2)
        # 2. FluxInpaintPipeline with FLUX.2-dev weights
        # 3. Flux2Pipeline with manual mask conditioning

        # First, try FLUX.2-specific inpaint pipeline
        try:
            from diffusers import Flux2InpaintPipeline

            print(f"Loading {self.model_name} with Flux2InpaintPipeline...")
            self.pipe = Flux2InpaintPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                text_encoder=None if self.use_remote_encoder else ...,
            ).to(self.device)
            self.pipe_type = "flux2_inpaint"
            print("Loaded Flux2InpaintPipeline successfully")
            return

        except (ImportError, Exception) as e:
            print(f"Flux2InpaintPipeline not available: {e}")

        # Try standard FluxInpaintPipeline
        try:
            from diffusers import FluxInpaintPipeline

            print(f"Loading {self.model_name} with FluxInpaintPipeline...")
            self.pipe = FluxInpaintPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
            self.pipe_type = "flux_inpaint"
            print("Loaded FluxInpaintPipeline successfully")
            return

        except (ImportError, Exception) as e:
            print(f"FluxInpaintPipeline not available with FLUX.2-dev: {e}")

        # Try FluxFillPipeline as fallback (dedicated inpainting model)
        try:
            from diffusers import FluxFillPipeline

            fill_model = "black-forest-labs/FLUX.1-Fill-dev"
            print(f"Falling back to {fill_model} with FluxFillPipeline...")
            self.pipe = FluxFillPipeline.from_pretrained(
                fill_model,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
            self.pipe_type = "flux_fill"
            print("Loaded FluxFillPipeline successfully")
            return

        except (ImportError, Exception) as e:
            print(f"FluxFillPipeline not available: {e}")

        # Last resort: load base Flux2Pipeline and handle inpainting manually
        try:
            from diffusers import Flux2Pipeline, FluxPipeline

            pipeline_class = None
            try:
                from diffusers import Flux2Pipeline
                pipeline_class = Flux2Pipeline
            except ImportError:
                from diffusers import FluxPipeline
                pipeline_class = FluxPipeline

            print(f"Loading {self.model_name} with {pipeline_class.__name__} (manual inpaint mode)...")
            self.pipe = pipeline_class.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                text_encoder=None if self.use_remote_encoder else ...,
            ).to(self.device)
            self.pipe_type = "flux2_base"
            print(f"Loaded {pipeline_class.__name__} - will use img2img approach for inpainting")

        except Exception as e:
            raise RuntimeError(f"Could not load any FLUX pipeline: {e}")

    def enable_model_cpu_offload(self):
        """Enable CPU offloading to save VRAM."""
        if hasattr(self.pipe, 'enable_model_cpu_offload'):
            self.pipe.enable_model_cpu_offload()
            print("Enabled model CPU offload")

    def enable_sequential_cpu_offload(self):
        """Enable sequential CPU offloading for very low VRAM."""
        if hasattr(self.pipe, 'enable_sequential_cpu_offload'):
            self.pipe.enable_sequential_cpu_offload()
            print("Enabled sequential CPU offload")

    def remote_text_encoder(self, prompt: str) -> torch.Tensor:
        """
        Get text embeddings from remote encoder (saves VRAM).

        Args:
            prompt: Text prompt

        Returns:
            Prompt embeddings tensor
        """
        url = "https://remote-text-encoder-flux-2.huggingface.co/predict"
        response = requests.post(
            url,
            json={"prompt": prompt},
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get embeddings: {response.text}")

        prompt_embeds = torch.load(io.BytesIO(response.content))
        return prompt_embeds.to(self.device)

    def _prepare_image(
        self,
        image: Union[Image.Image, np.ndarray, str],
        size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """Load and prepare image."""
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        if size is not None:
            image = image.resize(size, Image.Resampling.LANCZOS)

        return image

    def _prepare_mask(
        self,
        mask: Union[Image.Image, np.ndarray, str],
        size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """Load and prepare mask (white=inpaint, black=keep)."""
        if isinstance(mask, str):
            mask = Image.open(mask)
        elif isinstance(mask, np.ndarray):
            # Handle different mask formats
            if mask.ndim == 2:
                mask = Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L")
            elif mask.ndim == 3:
                # Take first channel or convert to grayscale
                if mask.shape[2] == 3:
                    mask = Image.fromarray(mask).convert("L")
                else:
                    mask = Image.fromarray(mask[:, :, 0])

        if isinstance(mask, Image.Image):
            mask = mask.convert("L")  # Ensure grayscale

        if size is not None:
            mask = mask.resize(size, Image.Resampling.NEAREST)

        return mask

    def inpaint(
        self,
        image: Union[Image.Image, np.ndarray, str],
        mask: Union[Image.Image, np.ndarray, str],
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        strength: float = 0.85,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Inpaint image using mask and prompt.

        Args:
            image: Source image (PIL, numpy, or path)
            mask: Binary mask - white areas will be inpainted
            prompt: Text description of what to generate in masked area
            negative_prompt: What to avoid generating
            width: Output width (default: match input)
            height: Output height (default: match input)
            num_inference_steps: Denoising steps (more = better quality)
            guidance_scale: How closely to follow prompt
            strength: Inpainting strength (0-1, higher = more change)
            seed: Random seed for reproducibility

        Returns:
            Inpainted PIL Image
        """
        # Prepare inputs
        image = self._prepare_image(image)
        original_size = image.size

        # Determine output size
        if width is None:
            width = original_size[0]
        if height is None:
            height = original_size[1]

        # Ensure dimensions are divisible by 8 (required by VAE)
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Resize if needed
        if (width, height) != original_size:
            image = image.resize((width, height), Image.Resampling.LANCZOS)

        mask = self._prepare_mask(mask, size=(width, height))

        # Setup generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Run inpainting based on pipeline type
        print(f"Running inpainting with {self.pipe_type}...")

        if self.pipe_type == "flux_fill":
            # FluxFillPipeline - dedicated inpainting
            result = self.pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=512,
                generator=generator,
            ).images[0]

        elif self.pipe_type in ["flux_inpaint", "flux2_inpaint"]:
            # FluxInpaintPipeline
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                image=image,
                mask_image=mask,
                height=height,
                width=width,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]

        elif self.pipe_type == "flux2_base":
            # Manual inpainting with base pipeline
            # This is a simplified approach - blend original with generated
            result = self._manual_inpaint(
                image=image,
                mask=mask,
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        else:
            raise RuntimeError(f"Unknown pipeline type: {self.pipe_type}")

        return result

    def _manual_inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: Optional[torch.Generator],
    ) -> Image.Image:
        """
        Manual inpainting using base FLUX.2 pipeline.

        This is a fallback that generates a new image and blends it with
        the original using the mask.
        """
        # Get embeddings
        if self.use_remote_encoder:
            prompt_embeds = self.remote_text_encoder(prompt)
            output = self.pipe(
                prompt_embeds=prompt_embeds,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil"
            ).images[0]
        else:
            output = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil"
            ).images[0]

        # Blend using mask
        # Convert mask to float for blending
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_np = np.stack([mask_np] * 3, axis=-1)

        image_np = np.array(image).astype(np.float32)
        output_np = np.array(output).astype(np.float32)

        # Blend: mask=1 (white) uses generated, mask=0 (black) uses original
        result_np = image_np * (1 - mask_np) + output_np * mask_np
        result = Image.fromarray(result_np.astype(np.uint8))

        return result


# Convenience function
def create_inpainter(
    model: str = "black-forest-labs/FLUX.2-dev",
    device: str = "cuda",
    low_vram: bool = False,
) -> Flux2Inpainter:
    """
    Create a FLUX.2 inpainter with optional VRAM optimizations.

    Args:
        model: Model name or path
        device: Device to use
        low_vram: Enable CPU offloading for lower VRAM usage

    Returns:
        Configured Flux2Inpainter instance
    """
    inpainter = Flux2Inpainter(model_name=model, device=device)

    if low_vram:
        inpainter.enable_model_cpu_offload()

    return inpainter
