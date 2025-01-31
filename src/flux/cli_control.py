import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from cuda import cudart
from fire import Fire
from transformers import pipeline

from flux.modules.image_embedders import CannyImageEncoder, DepthImageEncoder
from flux.sampling import denoise, get_noise, get_schedule, prepare_control, unpack
from flux.trt.trt_manager import TRTManager
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
    lora_scale: float | None


def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
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
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            options.height = 16 * (int(height) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
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

        options.img_cond_path = img_cond_path
        break

    return options


def parse_lora_scale(options: SamplingOptions | None) -> tuple[SamplingOptions | None, bool]:
    changed = False

    if options is None:
        return None, changed

    user_question = "Next lora scale (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the lora scale or write a command starting with a slash:\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/q"):
            print("Quitting")
            return None, changed
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    if prompt != "":
        options.lora_scale = float(prompt)
        changed = True
    return options, changed


@torch.inference_mode()
def main(
    name: str,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    prompt: str = "a robot made out of gold",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int = 50,
    loop: bool = False,
    guidance: float | None = None,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    img_cond_path: str = "assets/robot.webp",
    lora_scale: float | None = 0.85,
    trt: bool = False,
    trt_transformer_precision: str = "bf16",
    **kwargs: dict | None,
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
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
        trt: use TensorRT backend for optimized inference
    """
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    assert name in [
        "flux-dev-canny",
        "flux-dev-depth",
        "flux-dev-canny-lora",
        "flux-dev-depth-lora",
    ], f"Got unknown model name: {name}"
    if guidance is None:
        if name in ["flux-dev-canny", "flux-dev-canny-lora"]:
            guidance = 30.0
        elif name in ["flux-dev-depth", "flux-dev-depth-lora"]:
            guidance = 10.0
        else:
            raise NotImplementedError()

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
    t5 = load_t5(torch_device, max_length=512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    # set lora scale
    if "lora" in name and lora_scale is not None:
        assert not trt, "TRT does not support LORA yet"
        for _, module in model.named_modules():
            if hasattr(module, "set_scale"):
                module.set_scale(lora_scale)

    if name in ["flux-dev-depth", "flux-dev-depth-lora"]:
        img_embedder = DepthImageEncoder(torch_device)
    elif name in ["flux-dev-canny", "flux-dev-canny-lora"]:
        img_embedder = CannyImageEncoder(torch_device)
    else:
        raise NotImplementedError()

    if trt:
        trt_ctx_manager = TRTManager(
            bf16=True,
            device=torch_device,
            static_batch=kwargs.get("static_batch", True),
            static_shape=kwargs.get("static_shape", True),
        )
        ae.decoder.params = ae.params
        ae.encoder.params = ae.params
        engines = trt_ctx_manager.load_engines(
            models={
                "clip": clip.cpu(),
                "transformer": model.cpu(),
                "t5": t5.cpu(),
                "vae": ae.decoder.cpu(),
                "vae_encoder": ae.encoder.cpu(),
            },
            engine_dir=os.environ.get("TRT_ENGINE_DIR", "./engines"),
            onnx_dir=os.environ.get("ONNX_DIR", "./onnx"),
            opt_image_height=height,
            opt_image_width=width,
            transformer_precision=trt_transformer_precision,
        )
        torch.cuda.synchronize()

        trt_ctx_manager.init_runtime()
        # TODO: refactor. stream should be part of engine constructor maybe !!
        for _, engine in engines.items():
            engine.set_stream(stream=trt_ctx_manager.stream)

        if not offload:
            for _, engine in engines.items():
                engine.load()

            calculate_max_device_memory = trt_ctx_manager.calculate_max_device_memory(engines)
            _, shared_device_memory = cudart.cudaMalloc(calculate_max_device_memory)

            for _, engine in engines.items():
                engine.activate(device=torch_device, device_memory=shared_device_memory)

        ae = engines["vae"]
        model = engines["transformer"]
        clip = engines["clip"]
        t5 = engines["t5"]

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
        img_cond_path=img_cond_path,
        lora_scale=lora_scale,
    )

    if loop:
        opts = parse_prompt(opts)
        opts = parse_img_cond_path(opts)
        if "lora" in name:
            opts, changed = parse_lora_scale(opts)
            if changed:
                # update the lora scale:
                for _, module in model.named_modules():
                    if hasattr(module, "set_scale"):
                        module.set_scale(opts.lora_scale)

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
            t5, clip, ae = t5.to(torch_device), clip.to(torch_device), ae.to(torch_device)
        inp = prepare_control(
            t5,
            clip,
            x,
            prompt=opts.prompt,
            ae=ae,
            encoder=img_embedder,
            img_cond_path=opts.img_cond_path,
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
            if "lora" in name:
                opts, changed = parse_lora_scale(opts)
                if changed:
                    # update the lora scale:
                    for _, module in model.named_modules():
                        if hasattr(module, "set_scale"):
                            module.set_scale(opts.lora_scale)
        else:
            opts = None


def app():
    Fire(main)


if __name__ == "__main__":
    app()
