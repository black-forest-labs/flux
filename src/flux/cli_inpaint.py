"""
CLI for SAM3 + FLUX.2-dev Inpainting

Usage:
    python -m flux.cli_inpaint --image input.jpg --points "100,200;150,220" --prompt "a golden retriever"

Or launch the Gradio demo:
    python -m flux.cli_inpaint --demo
"""

import os
import time
from typing import Optional, List, Tuple

import torch
from fire import Fire
from PIL import Image


def parse_points(points_str: str) -> List[Tuple[int, int]]:
    """
    Parse points string into list of (x, y) tuples.

    Format: "x1,y1;x2,y2;..." or "x1,y1 x2,y2 ..."
    """
    if not points_str:
        return []

    points = []
    # Split by semicolon or space
    parts = points_str.replace(";", " ").split()

    for part in parts:
        if "," in part:
            x, y = part.split(",")
            points.append((int(x.strip()), int(y.strip())))

    return points


def parse_labels(labels_str: str, num_points: int) -> List[int]:
    """
    Parse labels string into list of integers.

    Format: "1,0,1" or "1 0 1" or just "1" (applied to all)
    """
    if not labels_str:
        return [1] * num_points  # Default: all foreground

    labels_str = labels_str.replace(",", " ")
    labels = [int(x) for x in labels_str.split()]

    # If single label, apply to all points
    if len(labels) == 1:
        return labels * num_points

    return labels


def inpaint(
    image: str,
    prompt: str,
    points: str = "",
    labels: str = "",
    mask: Optional[str] = None,
    output: str = "output_inpaint.png",
    negative_prompt: str = "",
    num_steps: int = 28,
    guidance: float = 3.5,
    strength: float = 0.85,
    dilate: int = 5,
    seed: Optional[int] = None,
    model: str = "black-forest-labs/FLUX.2-dev",
    low_vram: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Run inpainting on an image.

    Args:
        image: Path to input image
        prompt: What to generate in the masked region
        points: Click points as "x1,y1;x2,y2" (for SAM3 segmentation)
        labels: Point labels as "1,0,1" (1=foreground, 0=background)
        mask: Path to mask image (alternative to points)
        output: Output path
        negative_prompt: What to avoid
        num_steps: Number of inference steps
        guidance: Guidance scale
        strength: Inpainting strength
        dilate: Mask dilation in pixels
        seed: Random seed (-1 for random)
        model: FLUX model to use
        low_vram: Enable memory optimizations
        device: Device to use
    """
    from flux import create_pipeline

    if not os.path.exists(image):
        print(f"Error: Image not found: {image}")
        return

    if not points and not mask:
        print("Error: Must provide either --points or --mask")
        print("Use --points 'x1,y1;x2,y2' for SAM3 segmentation")
        print("Use --mask path/to/mask.png for direct mask input")
        return

    print(f"Loading pipeline (low_vram={low_vram})...")
    pipeline = create_pipeline(
        low_vram=low_vram,
        flux_model=model,
    )

    # Load image
    input_image = Image.open(image).convert("RGB")
    print(f"Input image: {input_image.size[0]}x{input_image.size[1]}")

    t0 = time.perf_counter()

    if mask:
        # Direct mask mode
        print(f"Using mask from: {mask}")
        mask_image = Image.open(mask).convert("L")

        result = pipeline.inpaint(
            image=input_image,
            mask=mask_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            strength=strength,
            seed=seed if seed and seed >= 0 else None,
        )
    else:
        # SAM3 point mode
        point_list = parse_points(points)
        label_list = parse_labels(labels, len(point_list))

        print(f"Points: {point_list}")
        print(f"Labels: {label_list}")

        result = pipeline.inpaint_at_points(
            image=input_image,
            points=point_list,
            labels=label_list,
            prompt=prompt,
            negative_prompt=negative_prompt,
            dilate_pixels=dilate,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            strength=strength,
            seed=seed if seed and seed >= 0 else None,
        )

    t1 = time.perf_counter()
    print(f"Inpainting completed in {t1 - t0:.1f}s")

    # Save result
    result.save(output, quality=95)
    print(f"Saved to: {output}")


def segment(
    image: str,
    points: str,
    labels: str = "",
    output: str = "mask.png",
    dilate: int = 0,
    visualize: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Generate segmentation mask from points (SAM3 only, no inpainting).

    Args:
        image: Path to input image
        points: Click points as "x1,y1;x2,y2"
        labels: Point labels as "1,0,1"
        output: Output mask path
        dilate: Mask dilation in pixels
        visualize: Save visualization overlay
        device: Device to use
    """
    from flux import SAM3Segmenter

    if not os.path.exists(image):
        print(f"Error: Image not found: {image}")
        return

    point_list = parse_points(points)
    label_list = parse_labels(labels, len(point_list))

    if not point_list:
        print("Error: No valid points provided")
        return

    print(f"Loading SAM3...")
    segmenter = SAM3Segmenter(device=device)

    input_image = Image.open(image).convert("RGB")
    print(f"Input image: {input_image.size[0]}x{input_image.size[1]}")
    print(f"Points: {point_list}")
    print(f"Labels: {label_list}")

    t0 = time.perf_counter()

    mask = segmenter.segment_to_pil(
        image=input_image,
        points=point_list,
        labels=label_list,
        dilate_pixels=dilate,
    )

    t1 = time.perf_counter()
    print(f"Segmentation completed in {t1 - t0:.3f}s")

    # Save mask
    mask.save(output)
    print(f"Saved mask to: {output}")

    if visualize:
        import numpy as np
        vis_output = output.replace(".png", "_vis.png")
        mask_np, score = segmenter.segment(
            image=input_image,
            points=point_list,
            labels=label_list,
        )
        vis = segmenter.visualize_mask(input_image, mask_np)
        vis.save(vis_output)
        print(f"Saved visualization to: {vis_output}")
        print(f"Mask confidence: {score:.3f}")


def demo(
    share: bool = False,
    port: int = 7860,
):
    """
    Launch the Gradio demo interface.

    Args:
        share: Create a public URL
        port: Port to run on
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    from demo_inpaint import create_demo

    print("Launching Gradio demo...")
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
    )


def main():
    """CLI entry point with subcommands."""
    Fire({
        "inpaint": inpaint,
        "segment": segment,
        "demo": demo,
    })


if __name__ == "__main__":
    main()
