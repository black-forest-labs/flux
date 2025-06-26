import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from fire import Fire
from transformers import pipeline

from flux.modules.image_embedders import ReduxImageEncoder
from flux.sampling import denoise, get_noise, get_schedule, prepare_redux, unpack
from flux.util import (
    get_checkpoint_path,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    save_image,
)


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None
    img_cond_path: str


def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Leave this field empty to do nothing "
        "or write a command starting with a slash:\n"
        "- '/w <width>' will set the width of the generated image\n"
        "- '/h <height>' will set the height of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/w"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, width = prompt.split()
            options.width = 16 * (int(width) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height * options.width / 1e6:.2f}MP)"
            )
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            options.height = 16 * (int(height) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height * options.width / 1e6:.2f}MP)"
            )
        elif prompt.startswith("/g"):
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

        options.img_cond_path = img_cond_path
        break

    return options


@torch.inference_mode()
def main(
    name: str = "flux-dev",
    width: int = 1360,
    height: int = 768,
    seed: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    guidance: float = 2.5,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    img_cond_path: str = "assets/robot.webp",
    track_usage: bool = False,
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        name: Name of the base model to use (either 'flux-dev' or 'flux-schnell')
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        offload: offload models to CPU when not in use
        output_dir: where to save the output images
        add_sampling_metadata: Add the prompt to the image Exif metadata
        img_cond_path: path to conditioning image (jpeg/png/webp)
        track_usage: track usage of the model for licensing purposes
    """

    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    if name not in (available := ["flux-dev", "flux-schnell"]):
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 50

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
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    # Download and initialize the Redux adapter
    redux_path = str(
        get_checkpoint_path("black-forest-labs/FLUX.1-Redux-dev", "flux1-redux-dev.safetensors", "FLUX_REDUX")
    )
    img_embedder = ReduxImageEncoder(torch_device, redux_path=redux_path)

    rng = torch.Generator(device="cpu")
    prompt = ""
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
        img_cond_path=img_cond_path,
    )

    if loop:
        opts = parse_prompt(opts)
        opts = parse_img_cond_path(opts)

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
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)
        inp = prepare_redux(
            t5,
            clip,
            x,
            prompt=opts.prompt,
            encoder=img_embedder,
            img_cond_path=opts.img_cond_path,
        )
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        # offload TEs to CPU, load model to gpu
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
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

        idx = save_image(
            nsfw_classifier, name, output_name, idx, x, add_sampling_metadata, prompt, track_usage=track_usage
        )

        if loop:
            print("-" * 80)
            opts = parse_prompt(opts)
            opts = parse_img_cond_path(opts)
        else:
            opts = None


if __name__ == "__main__":
    Fire(main)
