import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from fire import Fire
from PIL import Image
from transformers import pipeline

from flux.sampling import denoise, get_noise, get_schedule, prepare_fill, unpack
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5, save_image


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None
    img_cond_path: str
    img_mask_path: str


def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            options.guidance = float(guidance)
            print(f"Setting guidance to {options.guidance}")
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            options.seed = int(seed)
            print(f"Setting seed to {options.seed}")
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            options.num_steps = int(steps)
            print(f"Setting number of steps to {options.num_steps}")
        elif prompt.startswith("/q"):
            print("Quitting")
            return None
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    if prompt != "":
        options.prompt = prompt
    return options


def parse_img_cond_path(options: SamplingOptions | None) -> SamplingOptions | None:
    if options is None:
        return None

    user_question = "Next conditioning image (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the conditioning image or write a command starting with a slash:\n"
        "- '/q' to quit"
    )

    while True:
        img_cond_path = input(user_question)

        if img_cond_path.startswith("/"):
            if img_cond_path.startswith("/q"):
                print("Quitting")
                return None
            else:
                if not img_cond_path.startswith("/h"):
                    print(f"Got invalid command '{img_cond_path}'\n{usage}")
                print(usage)
            continue

        if img_cond_path == "":
            break

        if not os.path.isfile(img_cond_path) or not img_cond_path.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            print(f"File '{img_cond_path}' does not exist or is not a valid image file")
            continue
        else:
            with Image.open(img_cond_path) as img:
                width, height = img.size

            if width % 32 != 0 or height % 32 != 0:
                print(f"Image dimensions must be divisible by 32, got {width}x{height}")
                continue

        options.img_cond_path = img_cond_path
        break

    return options


def parse_img_mask_path(options: SamplingOptions | None) -> SamplingOptions | None:
    if options is None:
        return None

    user_question = "Next conditioning mask (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the conditioning mask or write a command starting with a slash:\n"
        "- '/q' to quit"
    )

    while True:
        img_mask_path = input(user_question)

        if img_mask_path.startswith("/"):
            if img_mask_path.startswith("/q"):
                print("Quitting")
                return None
            else:
                if not img_mask_path.startswith("/h"):
                    print(f"Got invalid command '{img_mask_path}'\n{usage}")
                print(usage)
            continue

        if img_mask_path == "":
            break

        if not os.path.isfile(img_mask_path) or not img_mask_path.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            print(f"File '{img_mask_path}' does not exist or is not a valid image file")
            continue
        else:
            with Image.open(img_mask_path) as img:
                width, height = img.size

            if width % 32 != 0 or height % 32 != 0:
                print(f"Image dimensions must be divisible by 32, got {width}x{height}")
                continue
            else:
                with Image.open(options.img_cond_path) as img_cond:
                    img_cond_width, img_cond_height = img_cond.size

                if width != img_cond_width or height != img_cond_height:
                    print(
                        f"Mask dimensions must match conditioning image, got {width}x{height} and {img_cond_width}x{img_cond_height}"
                    )
                    continue

        options.img_mask_path = img_mask_path
        break

    return options


@torch.inference_mode()
def main(
    seed: int | None = None,
    prompt: str = "a white paper cup",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int = 50,
    loop: bool = False,
    guidance: float = 30.0,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    img_cond_path: str = "assets/cup.png",
    img_mask_path: str = "assets/cup_mask.png",
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image. This demo assumes that the conditioning image and mask have
    the same shape and that height and width are divisible by 32.

    Args:
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
        img_cond_path: path to conditioning image (jpeg/png/webp)
        img_mask_path: path to conditioning mask (jpeg/png/webp
    """
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    name = "flux-dev-fill"
    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)

    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    # init all components
    t5 = load_t5(torch_device, max_length=128)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    rng = torch.Generator(device="cpu")
    with Image.open(img_cond_path) as img:
        width, height = img.size
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
        img_cond_path=img_cond_path,
        img_mask_path=img_mask_path,
    )

    if loop:
        opts = parse_prompt(opts)
        opts = parse_img_cond_path(opts)

        with Image.open(opts.img_cond_path) as img:
            width, height = img.size
        opts.height = height
        opts.width = width

        opts = parse_img_mask_path(opts)

    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        t0 = time.perf_counter()

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=torch_device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        opts.seed = None
        if offload:
            t5, clip, ae = t5.to(torch_device), clip.to(torch_device), ae.to(torch.device)
        inp = prepare_fill(
            t5,
            clip,
            x,
            prompt=opts.prompt,
            ae=ae,
            img_cond_path=opts.img_cond_path,
            mask_path=opts.img_mask_path,
        )

        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        # offload TEs and AE to CPU, load model to gpu
        if offload:
            t5, clip, ae = t5.cpu(), clip.cpu(), ae.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        # denoise initial noise
        x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

        # offload model, load autoencoder to gpu
        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s")

        idx = save_image(nsfw_classifier, name, output_name, idx, x, add_sampling_metadata, prompt)

        if loop:
            print("-" * 80)
            opts = parse_prompt(opts)
            opts = parse_img_cond_path(opts)

            with Image.open(opts.img_cond_path) as img:
                width, height = img.size
            opts.height = height
            opts.width = width

            opts = parse_img_mask_path(opts)
        else:
            opts = None


def app():
    Fire(main)


if __name__ == "__main__":
    app()
