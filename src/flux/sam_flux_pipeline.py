"""
SAM3 + FLUX.2-dev Inpainting Pipeline

Combines SAM3 point-based segmentation with FLUX.2-dev inpainting
for interactive image editing.
"""

from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np


@dataclass
class InpaintResult:
    """Result from inpainting operation."""
    output_image: Image.Image
    mask: Image.Image
    mask_score: float
    prompt: str
    seed: Optional[int] = None


class SAMFluxPipeline:
    """
    End-to-end pipeline for point-prompted inpainting.

    Workflow:
    1. User clicks point(s) on image
    2. SAM3 generates segmentation mask
    3. FLUX.2-dev inpaints the masked region

    Usage:
        pipeline = SAMFluxPipeline()
        result = pipeline.inpaint_at_points(
            image=my_image,
            points=[(100, 200), (150, 220)],
            prompt="a golden retriever"
        )
    """

    def __init__(
        self,
        sam_device: str = "cuda",
        flux_device: str = "cuda",
        sam_model_size: str = "large",
        flux_model: str = "black-forest-labs/FLUX.2-dev",
        low_vram: bool = False,
        lazy_load: bool = True,
    ):
        """
        Initialize the SAM3 + FLUX pipeline.

        Args:
            sam_device: Device for SAM3 model
            flux_device: Device for FLUX model
            sam_model_size: SAM3 model size ("large", "base_plus", "small")
            flux_model: FLUX model name/path
            low_vram: Enable memory optimizations
            lazy_load: If True, models are loaded on first use
        """
        self.sam_device = sam_device
        self.flux_device = flux_device
        self.sam_model_size = sam_model_size
        self.flux_model = flux_model
        self.low_vram = low_vram

        self._segmenter = None
        self._inpainter = None

        if not lazy_load:
            self._load_models()

    def _load_models(self):
        """Load both models."""
        self._load_sam()
        self._load_flux()

    def _load_sam(self):
        """Load SAM3 segmenter."""
        if self._segmenter is None:
            from .sam3_segmenter import SAM3Segmenter
            print("Loading SAM3 segmenter...")
            self._segmenter = SAM3Segmenter(
                device=self.sam_device,
                model_size=self.sam_model_size,
            )

    def _load_flux(self):
        """Load FLUX inpainter."""
        if self._inpainter is None:
            from .inpainting import Flux2Inpainter
            print("Loading FLUX.2 inpainter...")
            self._inpainter = Flux2Inpainter(
                model_name=self.flux_model,
                device=self.flux_device,
            )
            if self.low_vram:
                self._inpainter.enable_model_cpu_offload()

    @property
    def segmenter(self):
        """Get SAM3 segmenter (lazy load)."""
        if self._segmenter is None:
            self._load_sam()
        return self._segmenter

    @property
    def inpainter(self):
        """Get FLUX inpainter (lazy load)."""
        if self._inpainter is None:
            self._load_flux()
        return self._inpainter

    def segment(
        self,
        image: Union[Image.Image, np.ndarray, str],
        points: List[Tuple[int, int]],
        labels: Optional[List[int]] = None,
        dilate_pixels: int = 0,
    ) -> Tuple[Image.Image, float]:
        """
        Generate segmentation mask from points.

        Args:
            image: Input image
            points: List of (x, y) click coordinates
            labels: Point labels (1=include, 0=exclude)
            dilate_pixels: Expand mask by this many pixels

        Returns:
            Tuple of (mask_image, confidence_score)
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image)

        # Generate mask
        mask = self.segmenter.segment_to_pil(
            image=image,
            points=points,
            labels=labels,
            dilate_pixels=dilate_pixels,
        )

        # Get score
        _, score = self.segmenter.segment(
            image=image,
            points=points,
            labels=labels,
        )

        return mask, float(score)

    def inpaint(
        self,
        image: Union[Image.Image, np.ndarray, str],
        mask: Union[Image.Image, np.ndarray, str],
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        strength: float = 0.85,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Inpaint masked region with prompt.

        Args:
            image: Source image
            mask: Binary mask (white=inpaint)
            prompt: What to generate
            negative_prompt: What to avoid
            num_inference_steps: Quality/speed tradeoff
            guidance_scale: Prompt adherence
            strength: How much to change masked area
            seed: Random seed

        Returns:
            Inpainted image
        """
        return self.inpainter.inpaint(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
        )

    def inpaint_at_points(
        self,
        image: Union[Image.Image, np.ndarray, str],
        points: List[Tuple[int, int]],
        prompt: str,
        labels: Optional[List[int]] = None,
        negative_prompt: str = "",
        dilate_pixels: int = 5,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        strength: float = 0.85,
        seed: Optional[int] = None,
        return_mask: bool = False,
    ) -> Union[Image.Image, InpaintResult]:
        """
        Full pipeline: click points -> segment -> inpaint.

        Args:
            image: Input image
            points: List of (x, y) coordinates to segment
            prompt: What to generate in segmented area
            labels: Point labels (1=foreground, 0=background)
            negative_prompt: What to avoid
            dilate_pixels: Expand mask edges
            num_inference_steps: Denoising steps
            guidance_scale: Prompt adherence
            strength: Inpainting strength
            seed: Random seed
            return_mask: If True, return InpaintResult with mask

        Returns:
            Inpainted image, or InpaintResult if return_mask=True
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image)

        print(f"Segmenting with {len(points)} point(s)...")

        # Step 1: Generate mask from points
        mask, score = self.segment(
            image=image,
            points=points,
            labels=labels,
            dilate_pixels=dilate_pixels,
        )

        print(f"Mask generated with confidence: {score:.3f}")
        print(f"Inpainting with prompt: '{prompt}'")

        # Step 2: Inpaint
        result_image = self.inpaint(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
        )

        if return_mask:
            return InpaintResult(
                output_image=result_image,
                mask=mask,
                mask_score=score,
                prompt=prompt,
                seed=seed,
            )

        return result_image

    def visualize_points(
        self,
        image: Union[Image.Image, np.ndarray],
        points: List[Tuple[int, int]],
        labels: Optional[List[int]] = None,
        point_radius: int = 5,
    ) -> Image.Image:
        """
        Draw points on image for visualization.

        Args:
            image: Input image
            points: Point coordinates
            labels: Point labels (affects color)
            point_radius: Size of point markers

        Returns:
            Image with point markers
        """
        from PIL import ImageDraw

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.copy().convert("RGB")
        draw = ImageDraw.Draw(image)

        if labels is None:
            labels = [1] * len(points)

        for (x, y), label in zip(points, labels):
            # Green for positive, red for negative
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            draw.ellipse(
                [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                fill=color,
                outline=(255, 255, 255),
                width=2,
            )

        return image

    def visualize_mask_overlay(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Union[Image.Image, np.ndarray],
        alpha: float = 0.5,
        color: Tuple[int, int, int] = (0, 150, 255),
    ) -> Image.Image:
        """
        Overlay mask on image for visualization.

        Args:
            image: Original image
            mask: Segmentation mask
            alpha: Mask transparency
            color: Mask color (RGB)

        Returns:
            Image with mask overlay
        """
        return self.segmenter.visualize_mask(
            image=image,
            mask=np.array(mask.convert("L")) if isinstance(mask, Image.Image) else mask,
            alpha=alpha,
            color=color,
        )


# Convenience factory function
def create_pipeline(
    low_vram: bool = False,
    sam_size: str = "large",
    flux_model: str = "black-forest-labs/FLUX.2-dev",
) -> SAMFluxPipeline:
    """
    Create configured SAM+FLUX pipeline.

    Args:
        low_vram: Enable memory optimizations for <24GB VRAM
        sam_size: SAM model size ("large", "base_plus", "small")
        flux_model: FLUX model to use

    Returns:
        Configured pipeline instance
    """
    return SAMFluxPipeline(
        sam_model_size=sam_size,
        flux_model=flux_model,
        low_vram=low_vram,
        lazy_load=True,
    )
