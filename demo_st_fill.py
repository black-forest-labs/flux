import os
import re
import tempfile
import time
from glob import iglob
from io import BytesIO

import numpy as np
import streamlit as st
import torch
from einops import rearrange
from PIL import ExifTags, Image
from st_keyup import st_keyup
from streamlit_drawable_canvas import st_canvas
from transformers import pipeline

from flux.sampling import denoise, get_noise, get_schedule, prepare_fill, unpack
from flux.util import embed_watermark, load_ae, load_clip, load_flow_model, load_t5

NSFW_THRESHOLD = 0.85


def add_border_and_mask(image, zoom_all=1.0, zoom_left=0, zoom_right=0, zoom_up=0, zoom_down=0, overlap=0):
    """Adds a black border around the image with individual side control and mask overlap"""
    orig_width, orig_height = image.size

    # Calculate padding for each side (in pixels)
    left_pad = int(orig_width * zoom_left)
    right_pad = int(orig_width * zoom_right)
    top_pad = int(orig_height * zoom_up)
    bottom_pad = int(orig_height * zoom_down)

    # Calculate overlap in pixels
    overlap_left = int(orig_width * overlap)
    overlap_right = int(orig_width * overlap)
    overlap_top = int(orig_height * overlap)
    overlap_bottom = int(orig_height * overlap)

    # If using the all-sides zoom, add it to each side
    if zoom_all > 1.0:
        extra_each_side = (zoom_all - 1.0) / 2
        left_pad += int(orig_width * extra_each_side)
        right_pad += int(orig_width * extra_each_side)
        top_pad += int(orig_height * extra_each_side)
        bottom_pad += int(orig_height * extra_each_side)

    # Calculate new dimensions (ensure they're multiples of 32)
    new_width = 32 * round((orig_width + left_pad + right_pad) / 32)
    new_height = 32 * round((orig_height + top_pad + bottom_pad) / 32)

    # Create new image with black border
    bordered_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    # Paste original image in position
    paste_x = left_pad
    paste_y = top_pad
    bordered_image.paste(image, (paste_x, paste_y))

    # Create mask (white where the border is, black where the original image was)
    mask = Image.new("L", (new_width, new_height), 255)  # White background
    # Paste black rectangle with overlap adjustment
    mask.paste(
        0,
        (
            paste_x + overlap_left,  # Left edge moves right
            paste_y + overlap_top,  # Top edge moves down
            paste_x + orig_width - overlap_right,  # Right edge moves left
            paste_y + orig_height - overlap_bottom,  # Bottom edge moves up
        ),
    )

    return bordered_image, mask


@st.cache_resource()
def get_models(name: str, device: torch.device, offload: bool):
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if offload else device)
    ae = load_ae(name, device="cpu" if offload else device)
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
    return model, ae, t5, clip, nsfw_classifier


def resize(img: Image.Image, min_mp: float = 0.5, max_mp: float = 2.0) -> Image.Image:
    width, height = img.size
    mp = (width * height) / 1_000_000  # Current megapixels

    if min_mp <= mp <= max_mp:
        # Even if MP is in range, ensure dimensions are multiples of 32
        new_width = int(32 * round(width / 32))
        new_height = int(32 * round(height / 32))
        if new_width != width or new_height != height:
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img

    # Calculate scaling factor
    if mp < min_mp:
        scale = (min_mp / mp) ** 0.5
    else:  # mp > max_mp
        scale = (max_mp / mp) ** 0.5

    new_width = int(32 * round(width * scale / 32))
    new_height = int(32 * round(height * scale / 32))

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def clear_canvas_state():
    """Clear all canvas-related state"""
    keys_to_clear = ["canvas", "last_image_dims"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def set_new_image(img: Image.Image):
    """Safely set a new image and clear relevant state"""
    st.session_state["current_image"] = img
    clear_canvas_state()
    st.rerun()


def downscale_image(img: Image.Image, scale_factor: float) -> Image.Image:
    """Downscale image by a given factor while maintaining 32-pixel multiple dimensions"""
    if scale_factor >= 1.0:
        return img

    width, height = img.size
    new_width = int(32 * round(width * scale_factor / 32))
    new_height = int(32 * round(height * scale_factor / 32))

    # Ensure minimum dimensions
    new_width = max(64, new_width)  # minimum 64 pixels
    new_height = max(64, new_height)  # minimum 64 pixels

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


@torch.inference_mode()
def main(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offload: bool = False,
    output_dir: str = "output",
):
    torch_device = torch.device(device)
    st.title("Flux Fill: Inpainting & Outpainting")

    # Model selection and loading
    name = "flux-dev-fill"
    if not st.checkbox("Load model", False):
        return

    try:
        model, ae, t5, clip, nsfw_classifier = get_models(
            name,
            device=torch_device,
            offload=offload,
        )
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    # Mode selection
    mode = st.radio("Select Mode", ["Inpainting", "Outpainting"])

    # Image handling - either from previous generation or new upload
    if "input_image" in st.session_state:
        image = st.session_state["input_image"]
        del st.session_state["input_image"]
        set_new_image(image)
        st.write("Continuing from previous result")
    else:
        uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if uploaded_image is None:
            st.warning("Please upload an image")
            return

        if (
            "current_image_name" not in st.session_state
            or st.session_state["current_image_name"] != uploaded_image.name
        ):
            try:
                image = Image.open(uploaded_image).convert("RGB")
                st.session_state["current_image_name"] = uploaded_image.name
                set_new_image(image)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                return
        else:
            image = st.session_state.get("current_image")
            if image is None:
                st.error("Error: Image state is invalid. Please reupload the image.")
                clear_canvas_state()
                return

    # Add downscale control
    with st.expander("Image Size Control"):
        current_mp = (image.size[0] * image.size[1]) / 1_000_000
        st.write(f"Current image size: {image.size[0]}x{image.size[1]} ({current_mp:.1f}MP)")

        scale_factor = st.slider(
            "Downscale Factor",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="1.0 = original size, 0.5 = half size, etc.",
        )

        if scale_factor < 1.0 and st.button("Apply Downscaling"):
            image = downscale_image(image, scale_factor)
            set_new_image(image)
            st.rerun()

    # Resize image with validation
    try:
        original_mp = (image.size[0] * image.size[1]) / 1_000_000
        image = resize(image)
        width, height = image.size
        current_mp = (width * height) / 1_000_000

        if width % 32 != 0 or height % 32 != 0:
            st.error("Error: Image dimensions must be multiples of 32")
            return

        st.write(f"Image dimensions: {width}x{height} pixels")
        if original_mp != current_mp:
            st.write(
                f"Image has been resized from {original_mp:.1f}MP to {current_mp:.1f}MP to stay within bounds (0.5MP - 2MP)"
            )
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return

    if mode == "Outpainting":
        # Outpainting controls
        zoom_all = st.slider("Zoom Out Amount (All Sides)", min_value=1.0, max_value=3.0, value=1.0, step=0.1)

        with st.expander("Advanced Zoom Controls"):
            st.info("These controls add additional zoom to specific sides")
            col1, col2 = st.columns(2)
            with col1:
                zoom_left = st.slider("Left", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
                zoom_right = st.slider("Right", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
            with col2:
                zoom_up = st.slider("Up", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
                zoom_down = st.slider("Down", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

        overlap = st.slider("Overlap", min_value=0.01, max_value=0.25, value=0.01, step=0.01)

        # Generate bordered image and mask
        image_for_generation, mask = add_border_and_mask(
            image,
            zoom_all=zoom_all,
            zoom_left=zoom_left,
            zoom_right=zoom_right,
            zoom_up=zoom_up,
            zoom_down=zoom_down,
            overlap=overlap,
        )
        width, height = image_for_generation.size

        # Show preview
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_for_generation, caption="Image with Border")
        with col2:
            st.image(mask, caption="Mask (white areas will be generated)")

    else:  # Inpainting mode
        # Canvas setup with dimension tracking
        canvas_key = f"canvas_{width}_{height}"
        if "last_image_dims" not in st.session_state:
            st.session_state.last_image_dims = (width, height)
        elif st.session_state.last_image_dims != (width, height):
            clear_canvas_state()
            st.session_state.last_image_dims = (width, height)
            st.rerun()

        try:
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=st.slider("Brush size", 1, 500, 50),
                stroke_color="#fff",
                background_image=image,
                height=height,
                width=width,
                drawing_mode="freedraw",
                key=canvas_key,
                display_toolbar=True,
            )
        except Exception as e:
            st.error(f"Error creating canvas: {e}")
            clear_canvas_state()
            st.rerun()
            return

    # Sampling parameters
    num_steps = int(st.number_input("Number of steps", min_value=1, value=50))
    guidance = float(st.number_input("Guidance", min_value=1.0, value=30.0))
    seed_str = st.text_input("Seed")
    if seed_str.isdecimal():
        seed = int(seed_str)
    else:
        st.info("No seed set, using random seed")
        seed = None

    save_samples = st.checkbox("Save samples?", True)
    add_sampling_metadata = st.checkbox("Add sampling parameters to metadata?", True)

    # Prompt input
    prompt = st_keyup("Enter a prompt", value="", debounce=300, key="interactive_text")

    # Setup output path
    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        idx = len(fns)

    if st.button("Generate"):
        valid_input = False

        if mode == "Inpainting" and canvas_result.image_data is not None:
            valid_input = True
            # Create mask from canvas
            try:
                mask = Image.fromarray(canvas_result.image_data)
                mask = mask.getchannel("A")  # Get alpha channel
                mask_array = np.array(mask)
                mask_array = (mask_array > 0).astype(np.uint8) * 255
                mask = Image.fromarray(mask_array)
                image_for_generation = image
            except Exception as e:
                st.error(f"Error creating mask: {e}")
                return

        elif mode == "Outpainting":
            valid_input = True
            # image_for_generation and mask are already set above

        if not valid_input:
            st.error("Please draw a mask or configure outpainting settings")
            return

        # Create temporary files
        with (
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img,
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mask,
        ):
            try:
                image_for_generation.save(tmp_img.name)
                mask.save(tmp_mask.name)
            except Exception as e:
                st.error(f"Error saving temporary files: {e}")
                return

            try:
                # Generate inpainting/outpainting
                rng = torch.Generator(device="cpu")
                if seed is None:
                    seed = rng.seed()

                print(f"Generating with seed {seed}:\n{prompt}")
                t0 = time.perf_counter()

                x = get_noise(
                    1,
                    height,
                    width,
                    device=torch_device,
                    dtype=torch.bfloat16,
                    seed=seed,
                )

                if offload:
                    t5, clip, ae = t5.to(torch_device), clip.to(torch_device), ae.to(torch_device)

                inp = prepare_fill(
                    t5,
                    clip,
                    x,
                    prompt=prompt,
                    ae=ae,
                    img_cond_path=tmp_img.name,
                    mask_path=tmp_mask.name,
                )

                timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=True)

                if offload:
                    t5, clip, ae = t5.cpu(), clip.cpu(), ae.cpu()
                    torch.cuda.empty_cache()
                    model = model.to(torch_device)

                x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)

                if offload:
                    model.cpu()
                    torch.cuda.empty_cache()
                    ae.decoder.to(x.device)

                x = unpack(x.float(), height, width)
                with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)

                t1 = time.perf_counter()
                print(f"Done in {t1 - t0:.1f}s")

                # Process and display result
                x = x.clamp(-1, 1)
                x = embed_watermark(x.float())
                x = rearrange(x[0], "c h w -> h w c")
                img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

                nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]

                if nsfw_score < NSFW_THRESHOLD:
                    buffer = BytesIO()
                    exif_data = Image.Exif()
                    exif_data[ExifTags.Base.Software] = "AI generated;inpainting;flux"
                    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                    exif_data[ExifTags.Base.Model] = name
                    if add_sampling_metadata:
                        exif_data[ExifTags.Base.ImageDescription] = prompt
                    img.save(buffer, format="jpeg", exif=exif_data, quality=95, subsampling=0)

                    img_bytes = buffer.getvalue()
                    if save_samples:
                        fn = output_name.format(idx=idx)
                        print(f"Saving {fn}")
                        with open(fn, "wb") as file:
                            file.write(img_bytes)

                    st.session_state["samples"] = {
                        "prompt": prompt,
                        "img": img,
                        "seed": seed,
                        "bytes": img_bytes,
                    }
                else:
                    st.warning("Your generated image may contain NSFW content.")
                    st.session_state["samples"] = None

            except Exception as e:
                st.error(f"Error during generation: {e}")
                return
            finally:
                # Clean up temporary files
                try:
                    os.unlink(tmp_img.name)
                    os.unlink(tmp_mask.name)
                except Exception as e:
                    print(f"Error cleaning up temporary files: {e}")

    # Display results
    samples = st.session_state.get("samples", None)
    if samples is not None:
        st.image(samples["img"], caption=samples["prompt"])
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download full-resolution",
                samples["bytes"],
                file_name="generated.jpg",
                mime="image/jpg",
            )
        with col2:
            if st.button("Continue from this image"):
                # Store the generated image
                new_image = samples["img"]
                # Clear ALL canvas state
                clear_canvas_state()
                if "samples" in st.session_state:
                    del st.session_state["samples"]
                # Set as current image
                st.session_state["current_image"] = new_image
                st.rerun()

        st.write(f"Seed: {samples['seed']}")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
