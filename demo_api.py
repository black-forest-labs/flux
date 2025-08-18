# demo_api.py

import torch
from PIL import Image
from io import BytesIO
from einops import rearrange
from flux.util import embed_watermark, load_ae, load_clip, load_flow_model, load_t5
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack

def run_inference(prompt: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "flux-sdxl"  # если используешь другую модель, замени

    # загрузка моделей
    t5 = load_t5(device)
    clip = load_clip(device)
    model = load_flow_model(model_name, device=device)
    ae = load_ae(model_name, device=device)

    # параметры
    width, height = 1024, 1024
    steps = 50
    guidance = 5.0
    seed = 42

    x = get_noise(1, height, width, device=device, dtype=torch.float32, seed=seed)
    timesteps = get_schedule(steps, (x.shape[-1] * x.shape[-2]) // 4, shift=True)

    inp = prepare(t5=t5, clip=clip, img=x, prompt=prompt)

    x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)
    x = unpack(x.float(), height, width)
    x = ae.decode(x)

    x = x.clamp(-1, 1)
    x = embed_watermark(x.float())
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    buf = BytesIO()
    img.save(buf, format="JPEG")

    return {"image_base64": buf.getvalue()}
