## Open-weight models

We currently offer two open-weight text-to-image models.

| Name                      | HuggingFace repo                                         | License                                                                  | sha256sum                                                        |
| --------------------------|----------------------------------------------------------| -------------------------------------------------------------------------|----------------------------------------------------------------- |
| `FLUX.1 [schnell]`        | https://huggingface.co/black-forest-labs/FLUX.1-schnell  | [apache-2.0](../model_licenses/LICENSE-FLUX1-schnell)                    | 9403429e0052277ac2a87ad800adece5481eecefd9ed334e1f348723621d2a0a |
| `FLUX.1 [dev]`            | https://huggingface.co/black-forest-labs/FLUX.1-dev      | [FLUX.1-dev Non-Commercial License](../model_licenses/LICENSE-FLUX1-dev) | 4610115bb0c89560703c892c59ac2742fa821e60ef5871b33493ba544683abd7 |
| `FLUX.1 Krea [dev]`       | https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev | [FLUX.1-dev Non-Commercial License](../model_licenses/LICENSE-FLUX1-dev) | 4610115bb0c89560703c892c59ac2742fa821e60ef5871b33493ba544683abd7 |

## Open-weights usage

The weights will be downloaded automatically to `checkpoints/` from HuggingFace once you start one of the demos. Alternatively, you may download the weights manually and put them in `checkpoints/`, or you can also manually link them with the following environment variables:
```bash
export FLUX_MODEL=<your model path here>
export FLUX_AE=<your autoencoder path here>
```

For interactive sampling run

```bash
python -m flux t2i --name <name> --loop
```

where `name` is one of `flux-dev` or `flux-schnell`. Or to generate a single sample run

```bash
python -m flux t2i --name <name> \
  --height <height> --width <width> \
  --prompt "<prompt>"
```

### TRT engine infernece

We provide exports in BF16, FP8, and FP4 precision. Note that you need to install the repository with TensorRT support as outlined [here](../README.md).

```bash
python -m flux t2i --name=<name> --loop --trt --trt_transformer_precision <precision>
```
where `<trt_transformer_precision>` is either `bf16`, `fp8`, or `fp4`. For ONNX exports, `height` and `width` have to be within 768 and 1344.

### Streamlit and Gradio

We also provide a streamlit demo that does both text-to-image and image-to-image. The demo can be run via

```bash
streamlit run demo_st.py
```

We also offer a Gradio-based demo for an interactive experience. To run the Gradio demo:

```bash
python demo_gr.py --name flux-schnell --device cuda
```

Options:

- `--name`: Choose the model to use (options: "flux-schnell", "flux-dev")
- `--device`: Specify the device to use (default: "cuda" if available, otherwise "cpu")
- `--offload`: Offload model to CPU when not in use
- `--share`: Create a public link to your demo

To run the demo with the dev model and create a public link:

```bash
python demo_gr.py --name flux-dev --share
```

## Diffusers integration

`FLUX.1 [schnell]` and `FLUX.1 [dev]` are integrated with the [🧨 diffusers](https://github.com/huggingface/diffusers) library. To use it with diffusers, install it:

```shell
pip install git+https://github.com/huggingface/diffusers.git
```

Then you can use `FluxPipeline` to run the model

```python
import torch
from diffusers import FluxPipeline

model_id = "black-forest-labs/FLUX.1-schnell" #you can also use `black-forest-labs/FLUX.1-dev`

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
seed = 42
image = pipe(
    prompt,
    output_type="pil",
    num_inference_steps=4, #use a larger number if you are using [dev]
    generator=torch.Generator("cpu").manual_seed(seed)
).images[0]
image.save("flux-schnell.png")
```

To learn more check out the [diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux) documentation
