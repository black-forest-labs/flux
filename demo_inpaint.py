"""
SAM3 + FLUX.2-dev Inpainting Demo

Interactive Gradio demo for point-prompted image inpainting.
Click on an image to select regions, then describe what you want to generate.
"""

import gradio as gr
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import torch

# Global pipeline instance (lazy loaded)
_pipeline = None


def get_pipeline():
    """Get or create the pipeline instance."""
    global _pipeline
    if _pipeline is None:
        from flux import create_pipeline

        # Check available VRAM
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            low_vram = vram_gb < 24
            print(f"Detected {vram_gb:.1f}GB VRAM, low_vram={low_vram}")
        else:
            low_vram = True

        _pipeline = create_pipeline(low_vram=low_vram)

    return _pipeline


def add_point(
    image: np.ndarray,
    points_state: List[Tuple[int, int]],
    labels_state: List[int],
    evt: gr.SelectData,
    point_type: str,
) -> Tuple[np.ndarray, List, List, Optional[np.ndarray]]:
    """Add a point when user clicks on image."""
    if image is None:
        return None, points_state, labels_state, None

    # Get click coordinates
    x, y = evt.index[0], evt.index[1]
    label = 1 if point_type == "Foreground (include)" else 0

    # Add to state
    points_state = points_state + [(x, y)]
    labels_state = labels_state + [label]

    # Draw points on image
    annotated = draw_points(image, points_state, labels_state)

    # Generate mask preview
    mask_preview = generate_mask_preview(image, points_state, labels_state)

    return annotated, points_state, labels_state, mask_preview


def draw_points(
    image: np.ndarray,
    points: List[Tuple[int, int]],
    labels: List[int],
    radius: int = 8,
) -> np.ndarray:
    """Draw point markers on image."""
    from PIL import ImageDraw

    img = Image.fromarray(image).copy()
    draw = ImageDraw.Draw(img)

    for (x, y), label in zip(points, labels):
        # Green for foreground, red for background
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )

    return np.array(img)


def generate_mask_preview(
    image: np.ndarray,
    points: List[Tuple[int, int]],
    labels: List[int],
) -> Optional[np.ndarray]:
    """Generate mask preview from current points."""
    if not points:
        return None

    try:
        pipeline = get_pipeline()
        mask, score = pipeline.segment(
            image=Image.fromarray(image),
            points=points,
            labels=labels,
            dilate_pixels=5,
        )

        # Create overlay visualization
        overlay = pipeline.visualize_mask_overlay(
            image=image,
            mask=mask,
            alpha=0.4,
            color=(0, 150, 255),
        )

        return np.array(overlay)

    except Exception as e:
        print(f"Mask preview error: {e}")
        return None


def clear_points(image: np.ndarray) -> Tuple[np.ndarray, List, List, None]:
    """Clear all points and reset."""
    return image, [], [], None


def undo_point(
    original_image: np.ndarray,
    points_state: List[Tuple[int, int]],
    labels_state: List[int],
) -> Tuple[np.ndarray, List, List, Optional[np.ndarray]]:
    """Remove the last point."""
    if not points_state:
        return original_image, [], [], None

    points_state = points_state[:-1]
    labels_state = labels_state[:-1]

    if points_state:
        annotated = draw_points(original_image, points_state, labels_state)
        mask_preview = generate_mask_preview(original_image, points_state, labels_state)
    else:
        annotated = original_image
        mask_preview = None

    return annotated, points_state, labels_state, mask_preview


def run_inpainting(
    original_image: np.ndarray,
    points_state: List[Tuple[int, int]],
    labels_state: List[int],
    prompt: str,
    negative_prompt: str,
    num_steps: int,
    guidance_scale: float,
    strength: float,
    dilate_pixels: int,
    seed: int,
    progress=gr.Progress(),
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Run the full inpainting pipeline."""
    if original_image is None:
        return None, None, "Please upload an image first."

    if not points_state:
        return None, None, "Please click on the image to select a region."

    if not prompt.strip():
        return None, None, "Please enter a prompt describing what to generate."

    try:
        progress(0.1, desc="Loading models...")
        pipeline = get_pipeline()

        progress(0.3, desc="Generating mask...")

        # Use -1 for random seed
        actual_seed = seed if seed >= 0 else None

        progress(0.5, desc="Running inpainting...")
        result = pipeline.inpaint_at_points(
            image=Image.fromarray(original_image),
            points=points_state,
            labels=labels_state,
            prompt=prompt,
            negative_prompt=negative_prompt,
            dilate_pixels=dilate_pixels,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=actual_seed,
            return_mask=True,
        )

        progress(1.0, desc="Done!")

        status = f"Inpainting complete! Mask confidence: {result.mask_score:.3f}"
        if actual_seed:
            status += f", Seed: {actual_seed}"

        return np.array(result.output_image), np.array(result.mask), status

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"Error: {str(e)}"


def create_demo():
    """Create the Gradio demo interface."""

    with gr.Blocks(
        title="SAM3 + FLUX.2-dev Inpainting",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # SAM3 + FLUX.2-dev Point-Prompted Inpainting

            **How to use:**
            1. Upload an image
            2. Click on the image to select the region you want to change
               - Use **Foreground** points to include areas
               - Use **Background** points to exclude areas
            3. Enter a prompt describing what you want to generate
            4. Click **Run Inpainting**

            The mask preview updates as you click, so you can see exactly what will be inpainted.
            """
        )

        # State for points
        points_state = gr.State([])
        labels_state = gr.State([])
        original_image_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                input_image = gr.Image(
                    label="Input Image (click to add points)",
                    type="numpy",
                    interactive=True,
                    height=512,
                )

                point_type = gr.Radio(
                    choices=["Foreground (include)", "Background (exclude)"],
                    value="Foreground (include)",
                    label="Point Type",
                )

                with gr.Row():
                    clear_btn = gr.Button("Clear Points", variant="secondary")
                    undo_btn = gr.Button("Undo Last Point", variant="secondary")

                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe what to generate in the selected region...",
                    lines=2,
                )

                negative_prompt = gr.Textbox(
                    label="Negative Prompt (optional)",
                    placeholder="What to avoid...",
                    lines=1,
                )

                with gr.Accordion("Advanced Settings", open=False):
                    num_steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=28,
                        step=1,
                        label="Inference Steps",
                    )
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=3.5,
                        step=0.5,
                        label="Guidance Scale",
                    )
                    strength = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                        label="Inpainting Strength",
                    )
                    dilate_pixels = gr.Slider(
                        minimum=0,
                        maximum=30,
                        value=5,
                        step=1,
                        label="Mask Dilation (pixels)",
                    )
                    seed = gr.Number(
                        value=-1,
                        label="Seed (-1 for random)",
                        precision=0,
                    )

                run_btn = gr.Button("Run Inpainting", variant="primary", size="lg")

            with gr.Column(scale=1):
                # Output displays
                mask_preview = gr.Image(
                    label="Mask Preview",
                    type="numpy",
                    interactive=False,
                    height=256,
                )

                output_image = gr.Image(
                    label="Inpainted Result",
                    type="numpy",
                    interactive=False,
                    height=512,
                )

                output_mask = gr.Image(
                    label="Final Mask",
                    type="numpy",
                    interactive=False,
                    height=256,
                )

                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                )

        # Event handlers

        # Store original image when uploaded
        def on_image_upload(img):
            if img is None:
                return None, [], [], None
            return img, [], [], None

        input_image.upload(
            fn=on_image_upload,
            inputs=[input_image],
            outputs=[original_image_state, points_state, labels_state, mask_preview],
        )

        # Also update original when image changes
        input_image.change(
            fn=lambda img: img,
            inputs=[input_image],
            outputs=[original_image_state],
        )

        # Handle clicks on image
        input_image.select(
            fn=add_point,
            inputs=[original_image_state, points_state, labels_state, point_type],
            outputs=[input_image, points_state, labels_state, mask_preview],
        )

        # Clear points
        clear_btn.click(
            fn=clear_points,
            inputs=[original_image_state],
            outputs=[input_image, points_state, labels_state, mask_preview],
        )

        # Undo last point
        undo_btn.click(
            fn=undo_point,
            inputs=[original_image_state, points_state, labels_state],
            outputs=[input_image, points_state, labels_state, mask_preview],
        )

        # Run inpainting
        run_btn.click(
            fn=run_inpainting,
            inputs=[
                original_image_state,
                points_state,
                labels_state,
                prompt,
                negative_prompt,
                num_steps,
                guidance_scale,
                strength,
                dilate_pixels,
                seed,
            ],
            outputs=[output_image, output_mask, status_text],
        )

        # Example section
        gr.Markdown("---")
        gr.Markdown("### Tips")
        gr.Markdown(
            """
            - **Multiple points**: Click multiple times to refine the selection
            - **Background points**: Use to exclude areas from the mask
            - **Mask dilation**: Increase to expand the inpainted area slightly
            - **Strength**: Lower values preserve more of the original
            - **Guidance scale**: Higher values follow the prompt more strictly
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
