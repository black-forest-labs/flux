import getpass
import math
import os
from dataclasses import dataclass
from pathlib import Path

import requests
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download, login
from imwatermark import WatermarkEncoder
from PIL import ExifTags, Image
from safetensors.torch import load_file as load_sft

from flux.model import Flux, FluxLoraWrapper, FluxParams
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from flux.modules.conditioner import HFEmbedder

CHECKPOINTS_DIR = Path("checkpoints")
CHECKPOINTS_DIR.mkdir(exist_ok=True)
BFL_API_KEY = os.getenv("BFL_API_KEY")

os.environ.setdefault("TRT_ENGINE_DIR", str(CHECKPOINTS_DIR / "trt_engines"))
(CHECKPOINTS_DIR / "trt_engines").mkdir(exist_ok=True)


def ensure_hf_auth():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Trying to authenticate to HuggingFace with the HF_TOKEN environment variable.")
        try:
            login(token=hf_token)
            print("Successfully authenticated with HuggingFace using HF_TOKEN")
            return True
        except Exception as e:
            print(f"Warning: Failed to authenticate with HF_TOKEN: {e}")

    if os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
        print("Already authenticated with HuggingFace")
        return True

    return False


def prompt_for_hf_auth():
    try:
        token = getpass.getpass("HF Token (hidden input): ").strip()
        if not token:
            print("No token provided. Aborting.")
            return False

        login(token=token)
        print("Successfully authenticated!")
        return True
    except KeyboardInterrupt:
        print("\nAuthentication cancelled by user.")
        return False
    except Exception as auth_e:
        print(f"Authentication failed: {auth_e}")
        print("Tip: You can also run 'huggingface-cli login' or set HF_TOKEN environment variable")
        return False


def get_checkpoint_path(repo_id: str, filename: str, env_var: str) -> Path:
    """Get the local path for a checkpoint file, downloading if necessary."""
    if os.environ.get(env_var) is not None:
        local_path = os.environ[env_var]
        if os.path.exists(local_path):
            return Path(local_path)

        print(
            f"Trying to load model {repo_id}, {filename} from environment "
            f"variable {env_var}. But file {local_path} does not exist. "
            "Falling back to default location."
        )

    # Create a safe directory name from repo_id
    safe_repo_name = repo_id.replace("/", "_")
    checkpoint_dir = CHECKPOINTS_DIR / safe_repo_name
    checkpoint_dir.mkdir(exist_ok=True)

    local_path = checkpoint_dir / filename

    if not local_path.exists():
        print(f"Downloading {filename} from {repo_id} to {local_path}")
        try:
            ensure_hf_auth()
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=checkpoint_dir)
        except Exception as e:
            if "gated repo" in str(e).lower() or "restricted" in str(e).lower():
                print(f"\nError: Cannot access {repo_id} -- this is a gated repository.")

                # Try one more time to authenticate
                if prompt_for_hf_auth():
                    # Retry the download after authentication
                    print("Retrying download...")
                    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=checkpoint_dir)
                else:
                    print("Authentication failed or cancelled.")
                    print("You can also run 'huggingface-cli login' or set HF_TOKEN environment variable")
                    raise RuntimeError(f"Authentication required for {repo_id}")
            else:
                raise e

    return local_path


def download_onnx_models_for_trt(model_name: str, trt_transformer_precision: str = "bf16") -> str | None:
    """Download ONNX models for TRT to our checkpoints directory"""
    onnx_repo_map = {
        "flux-dev": "black-forest-labs/FLUX.1-dev-onnx",
        "flux-schnell": "black-forest-labs/FLUX.1-schnell-onnx",
        "flux-dev-canny": "black-forest-labs/FLUX.1-Canny-dev-onnx",
        "flux-dev-depth": "black-forest-labs/FLUX.1-Depth-dev-onnx",
        "flux-dev-redux": "black-forest-labs/FLUX.1-Redux-dev-onnx",
        "flux-dev-fill": "black-forest-labs/FLUX.1-Fill-dev-onnx",
        "flux-dev-kontext": "black-forest-labs/FLUX.1-Kontext-dev-onnx",
    }

    if model_name not in onnx_repo_map:
        return None  # No ONNX repository required for this model

    repo_id = onnx_repo_map[model_name]
    safe_repo_name = repo_id.replace("/", "_")
    onnx_dir = CHECKPOINTS_DIR / safe_repo_name

    # Map of module names to their ONNX file paths (using specified precision)
    onnx_file_map = {
        "clip": "clip.opt/model.onnx",
        "transformer": f"transformer.opt/{trt_transformer_precision}/model.onnx",
        "transformer_data": f"transformer.opt/{trt_transformer_precision}/backbone.onnx_data",
        "t5": "t5.opt/model.onnx",
        "t5_data": "t5.opt/backbone.onnx_data",
        "vae": "vae.opt/model.onnx",
    }

    # If all files exist locally, return the custom_onnx_paths format
    if onnx_dir.exists():
        all_files_exist = True
        custom_paths = []
        for module, onnx_file in onnx_file_map.items():
            if module.endswith("_data"):
                continue  # Skip data files
            local_path = onnx_dir / onnx_file
            if not local_path.exists():
                all_files_exist = False
                break
            custom_paths.append(f"{module}:{local_path}")

        if all_files_exist:
            print(f"ONNX models ready in {onnx_dir}")
            return ",".join(custom_paths)

    # If not all files exist, download them
    print(f"Downloading ONNX models from {repo_id} to {onnx_dir}")
    print(f"Using transformer precision: {trt_transformer_precision}")
    onnx_dir.mkdir(exist_ok=True)

    # Download all ONNX files
    for module, onnx_file in onnx_file_map.items():
        local_path = onnx_dir / onnx_file
        if local_path.exists():
            continue  # Already downloaded

        # Create parent directories
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            print(f"Downloading {onnx_file}")
            hf_hub_download(repo_id=repo_id, filename=onnx_file, local_dir=onnx_dir)
        except Exception as e:
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                continue
            elif "gated repo" in str(e).lower() or "restricted" in str(e).lower():
                print(f"Cannot access {repo_id} - requires license acceptance")
                print("Please follow these steps:")
                print(f"   1. Visit: https://huggingface.co/{repo_id}")
                print("   2. Log in to your HuggingFace account")
                print("   3. Accept the license terms and conditions")
                print("   4. Then retry this command")
                raise RuntimeError(f"License acceptance required for {model_name}")
            else:
                # Re-raise other errors
                raise

    print(f"ONNX models ready in {onnx_dir}")

    # Return the custom_onnx_paths format that TRT expects: "module1:path1,module2:path2"
    # Note: Only return the actual module paths, not the data file
    custom_paths = []
    for module, onnx_file in onnx_file_map.items():
        if module.endswith("_data"):
            continue  # Skip the data file in the return paths
        full_path = onnx_dir / onnx_file
        if full_path.exists():
            custom_paths.append(f"{module}:{full_path}")

    return ",".join(custom_paths)


def check_onnx_access_for_trt(model_name: str, trt_transformer_precision: str = "bf16") -> str | None:
    """Check ONNX access and download models for TRT - returns ONNX directory path"""
    return download_onnx_models_for_trt(model_name, trt_transformer_precision)


def track_usage_via_api(name: str, n=1) -> None:
    """
    Track usage of licensed models via the BFL API for commercial licensing compliance.

    For more information on licensing BFL's models for commercial use and usage reporting,
    see the README.md or visit: https://dashboard.bfl.ai/licensing/subscriptions?showInstructions=true
    """
    assert BFL_API_KEY is not None, "BFL_API_KEY is not set"

    model_slug_map = {
        "flux-dev": "flux-1-dev",
        "flux-dev-kontext": "flux-1-kontext-dev",
        "flux-dev-fill": "flux-tools",
        "flux-dev-depth": "flux-tools",
        "flux-dev-canny": "flux-tools",
        "flux-dev-canny-lora": "flux-tools",
        "flux-dev-depth-lora": "flux-tools",
        "flux-dev-redux": "flux-tools",
    }

    if name not in model_slug_map:
        print(f"Skipping tracking usage for {name}, as it cannot be tracked. Please check the model name.")
        return

    model_slug = model_slug_map[name]
    url = f"https://api.bfl.ai/v1/licenses/models/{model_slug}/usage"
    headers = {"x-key": BFL_API_KEY, "Content-Type": "application/json"}
    payload = {"number_of_generations": n}

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Failed to track usage: {response.status_code} {response.text}")
    else:
        print(f"Successfully tracked usage for {name} with {n} generations")


def save_image(
    nsfw_classifier,
    name: str,
    output_name: str,
    idx: int,
    x: torch.Tensor,
    add_sampling_metadata: bool,
    prompt: str,
    nsfw_threshold: float = 0.85,
    track_usage: bool = False,
) -> int:
    fn = output_name.format(idx=idx)
    print(f"Saving {fn}")
    # bring into PIL format and save
    x = x.clamp(-1, 1)
    x = embed_watermark(x.float())
    x = rearrange(x[0], "c h w -> h w c")

    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    if nsfw_classifier is not None:
        nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]
    else:
        nsfw_score = nsfw_threshold - 1.0

    if nsfw_score < nsfw_threshold:
        exif_data = Image.Exif()
        if name in ["flux-dev", "flux-schnell"]:
            exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        else:
            exif_data[ExifTags.Base.Software] = "AI generated;img2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = name
        if add_sampling_metadata:
            exif_data[ExifTags.Base.ImageDescription] = prompt
        img.save(fn, exif=exif_data, quality=95, subsampling=0)
        if track_usage:
            track_usage_via_api(name, 1)
        idx += 1
    else:
        print("Your generated image may contain NSFW content.")

    return idx


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    repo_id: str
    repo_flow: str
    repo_ae: str
    lora_repo_id: str | None = None
    lora_filename: str | None = None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-canny": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Canny-dev",
        repo_flow="flux1-canny-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-canny-lora": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        lora_repo_id="black-forest-labs/FLUX.1-Canny-dev-lora",
        lora_filename="flux1-canny-dev-lora.safetensors",
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-depth": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Depth-dev",
        repo_flow="flux1-depth-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-depth-lora": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        lora_repo_id="black-forest-labs/FLUX.1-Depth-dev-lora",
        lora_filename="flux1-depth-dev-lora.safetensors",
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-redux": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Redux-dev",
        repo_flow="flux1-redux-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-fill": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Fill-dev",
        repo_flow="flux1-fill-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=384,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-kontext": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Kontext-dev",
        repo_flow="flux1-kontext-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def aspect_ratio_to_height_width(aspect_ratio: str, area: int = 1024**2) -> tuple[int, int]:
    width = float(aspect_ratio.split(":")[0])
    height = float(aspect_ratio.split(":")[1])
    ratio = width / height
    width = round(math.sqrt(area * ratio))
    height = round(math.sqrt(area / ratio))
    return 16 * (width // 16), 16 * (height // 16)


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_flow_model(name: str, device: str | torch.device = "cuda", verbose: bool = True) -> Flux:
    # Loading Flux
    print("Init model")
    config = configs[name]

    ckpt_path = str(get_checkpoint_path(config.repo_id, config.repo_flow, "FLUX_MODEL"))

    with torch.device("meta"):
        if config.lora_repo_id is not None and config.lora_filename is not None:
            model = FluxLoraWrapper(params=config.params).to(torch.bfloat16)
        else:
            model = Flux(config.params).to(torch.bfloat16)

    print(f"Loading checkpoint: {ckpt_path}")
    # load_sft doesn't support torch.device
    sd = load_sft(ckpt_path, device=str(device))
    sd = optionally_expand_state_dict(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if verbose:
        print_load_warning(missing, unexpected)

    if config.lora_repo_id is not None and config.lora_filename is not None:
        print("Loading LoRA")
        lora_path = str(get_checkpoint_path(config.lora_repo_id, config.lora_filename, "FLUX_LORA"))
        lora_sd = load_sft(lora_path, device=str(device))
        # loading the lora params + overwriting scale values in the norms
        missing, unexpected = model.load_state_dict(lora_sd, strict=False, assign=True)
        if verbose:
            print_load_warning(missing, unexpected)
    return model


def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder("google/t5-v1_1-xxl", max_length=max_length, torch_dtype=torch.bfloat16).to(device)


def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)


def load_ae(name: str, device: str | torch.device = "cuda") -> AutoEncoder:
    config = configs[name]
    ckpt_path = str(get_checkpoint_path(config.repo_id, config.repo_ae, "FLUX_AE"))

    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta"):
        ae = AutoEncoder(config.ae_params)

    print(f"Loading AE checkpoint: {ckpt_path}")
    sd = load_sft(ckpt_path, device=str(device))
    missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
    print_load_warning(missing, unexpected)
    return ae


def optionally_expand_state_dict(model: torch.nn.Module, state_dict: dict) -> dict:
    """
    Optionally expand the state dict to match the model's parameters shapes.
    """
    for name, param in model.named_parameters():
        if name in state_dict:
            if state_dict[name].shape != param.shape:
                print(
                    f"Expanding '{name}' with shape {state_dict[name].shape} to model parameter with shape {param.shape}."
                )
                # expand with zeros:
                expanded_state_dict_weight = torch.zeros_like(param, device=state_dict[name].device)
                slices = tuple(slice(0, dim) for dim in state_dict[name].shape)
                expanded_state_dict_weight[slices] = state_dict[name]
                state_dict[name] = expanded_state_dict_weight

    return state_dict


class WatermarkEmbedder:
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Adds a predefined watermark to the input image

        Args:
            image: ([N,] B, RGB, H, W) in range [-1, 1]

        Returns:
            same as input but watermarked
        """
        image = 0.5 * image + 0.5
        squeeze = len(image.shape) == 4
        if squeeze:
            image = image[None, ...]
        n = image.shape[0]
        image_np = rearrange((255 * image).detach().cpu(), "n b c h w -> (n b) h w c").numpy()[:, :, :, ::-1]
        # torch (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
        # watermarking libary expects input as cv2 BGR format
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
        image = torch.from_numpy(rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)).to(
            image.device
        )
        image = torch.clamp(image / 255, min=0.0, max=1.0)
        if squeeze:
            image = image[0]
        image = 2 * image - 1
        return image


# A fixed 48-bit message that was chosen at random
WATERMARK_MESSAGE = 0b001010101111111010000111100111001111010100101110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
embed_watermark = WatermarkEmbedder(WATERMARK_BITS)
