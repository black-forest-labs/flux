"""
SAM3 Segmenter Module

Generates binary masks from point prompts using Meta's Segment Anything Model 3.
"""

import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Optional, Union


class SAM3Segmenter:
    """
    Wrapper for SAM3 model to generate masks from point prompts.

    Usage:
        segmenter = SAM3Segmenter()
        mask = segmenter.segment(image, points=[(500, 375)], labels=[1])
    """

    def __init__(self, device: str = "cuda", model_size: str = "large"):
        """
        Initialize SAM3 segmenter.

        Args:
            device: Device to run model on ("cuda" or "cpu")
            model_size: Model size - "large", "base_plus", or "small"
        """
        self.device = device
        self.model_size = model_size
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load SAM3 model and processor."""
        try:
            from sam3.build_sam import build_sam3
            from sam3.sam3_image_predictor import SAM3ImagePredictor

            print(f"Loading SAM3 model ({self.model_size})...")

            # Build SAM3 model
            # Model configs follow SAM2 pattern
            model_cfg_map = {
                "large": "sam3_hiera_l.yaml",
                "base_plus": "sam3_hiera_b+.yaml",
                "small": "sam3_hiera_s.yaml",
            }

            model_cfg = model_cfg_map.get(self.model_size, "sam3_hiera_l.yaml")

            self.model = build_sam3(model_cfg, device=self.device)
            self.predictor = SAM3ImagePredictor(self.model)

            print("SAM3 model loaded successfully")

        except ImportError:
            # Fallback: try alternative import pattern from README
            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor

                print(f"Loading SAM3 model (alternative loader)...")

                self.model = build_sam3_image_model()
                self.processor = Sam3Processor(self.model)
                self.predictor = None  # Use processor instead

                print("SAM3 model loaded successfully (processor mode)")

            except ImportError as e:
                raise ImportError(
                    f"Could not import SAM3. Please install it:\n"
                    f"  git clone https://github.com/facebookresearch/sam3.git\n"
                    f"  cd sam3 && pip install -e .\n"
                    f"Original error: {e}"
                )

    def set_image(self, image: Union[Image.Image, np.ndarray]) -> None:
        """
        Set the image for segmentation.

        Args:
            image: PIL Image or numpy array (RGB)
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert("RGB"))
        else:
            image_np = image

        if self.predictor is not None:
            # SAM3ImagePredictor mode
            with torch.inference_mode():
                self.predictor.set_image(image_np)
        else:
            # Sam3Processor mode
            self._inference_state = self.processor.set_image(
                Image.fromarray(image_np) if isinstance(image_np, np.ndarray) else image_np
            )

    def segment(
        self,
        image: Optional[Union[Image.Image, np.ndarray]] = None,
        points: Optional[List[Tuple[int, int]]] = None,
        labels: Optional[List[int]] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
        multimask_output: bool = False,
        return_best_mask: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate segmentation mask from point prompts.

        Args:
            image: Input image (PIL or numpy). If None, uses previously set image.
            points: List of (x, y) point coordinates
            labels: List of labels (1=foreground, 0=background). Must match points length.
            box: Optional bounding box (x1, y1, x2, y2)
            multimask_output: If True, return multiple mask candidates
            return_best_mask: If True, return only the highest-scoring mask

        Returns:
            Tuple of (mask, score) where mask is a binary numpy array
        """
        # Set image if provided
        if image is not None:
            self.set_image(image)

        # Prepare point inputs
        point_coords = None
        point_labels = None

        if points is not None:
            point_coords = np.array(points, dtype=np.float32)
            if labels is None:
                # Default: all positive (foreground) points
                labels = [1] * len(points)
            point_labels = np.array(labels, dtype=np.int32)

        # Prepare box input
        input_box = None
        if box is not None:
            input_box = np.array(box, dtype=np.float32)

        # Run prediction
        if self.predictor is not None:
            # SAM3ImagePredictor mode
            with torch.inference_mode():
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=input_box,
                    multimask_output=multimask_output,
                )
        else:
            # Sam3Processor mode - use visual prompts
            # This mode might need adaptation based on actual SAM3 API
            if point_coords is not None:
                output = self.processor.set_point_prompt(
                    state=self._inference_state,
                    points=point_coords.tolist(),
                    labels=point_labels.tolist(),
                )
            elif input_box is not None:
                output = self.processor.set_box_prompt(
                    state=self._inference_state,
                    box=input_box.tolist(),
                )
            else:
                raise ValueError("Must provide either points or box for segmentation")

            masks = output["masks"]
            scores = output["scores"]

        # Handle output
        if return_best_mask and len(masks) > 0:
            # Return the highest scoring mask
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = float(scores[best_idx])
        else:
            mask = masks
            score = scores

        return mask, score

    def segment_to_pil(
        self,
        image: Union[Image.Image, np.ndarray],
        points: List[Tuple[int, int]],
        labels: Optional[List[int]] = None,
        dilate_pixels: int = 0,
    ) -> Image.Image:
        """
        Generate mask and return as PIL Image (white=mask, black=background).

        This format is ready for FLUX inpainting (white areas will be inpainted).

        Args:
            image: Input image
            points: List of (x, y) point coordinates
            labels: Point labels (1=foreground, 0=background)
            dilate_pixels: Optional dilation to expand mask edges

        Returns:
            PIL Image with white mask on black background
        """
        mask, score = self.segment(
            image=image,
            points=points,
            labels=labels,
            return_best_mask=True,
        )

        # Ensure mask is 2D binary
        if mask.ndim == 3:
            mask = mask.squeeze()

        # Convert to uint8 (0 or 255)
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255

        # Optional dilation to expand mask
        if dilate_pixels > 0:
            from scipy import ndimage
            struct = ndimage.generate_binary_structure(2, 1)
            mask_uint8 = ndimage.binary_dilation(
                mask_uint8 > 0,
                structure=struct,
                iterations=dilate_pixels
            ).astype(np.uint8) * 255

        # Convert to PIL
        mask_pil = Image.fromarray(mask_uint8, mode="L")

        # Convert to RGB (white on black) for FLUX compatibility
        mask_rgb = Image.merge("RGB", [mask_pil, mask_pil, mask_pil])

        return mask_rgb

    def visualize_mask(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: np.ndarray,
        alpha: float = 0.5,
        color: Tuple[int, int, int] = (255, 0, 0),
    ) -> Image.Image:
        """
        Overlay mask on image for visualization.

        Args:
            image: Original image
            mask: Binary mask array
            alpha: Transparency of mask overlay
            color: RGB color for mask

        Returns:
            PIL Image with mask overlay
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.convert("RGBA")

        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask.squeeze()

        # Create colored overlay
        overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
        overlay[mask > 0.5] = [*color, int(255 * alpha)]

        overlay_img = Image.fromarray(overlay, mode="RGBA")

        # Composite
        result = Image.alpha_composite(image, overlay_img)

        return result.convert("RGB")
