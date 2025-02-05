## Models

We currently offer four text-to-image models. `FLUX1.1 [pro]` is our most capable model which can generate images at up to 4MP while maintaining an impressive generation time of only 10 seconds per sample.

| Name                      | HuggingFace repo                                        | License                                                               | sha256sum                                                        |
| ------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `FLUX.1 [schnell]`        | https://huggingface.co/black-forest-labs/FLUX.1-schnell | [apache-2.0](model_licenses/LICENSE-FLUX1-schnell)                    | 9403429e0052277ac2a87ad800adece5481eecefd9ed334e1f348723621d2a0a |
| `FLUX.1 [dev]`            | https://huggingface.co/black-forest-labs/FLUX.1-dev     | [FLUX.1-dev Non-Commercial License](model_licenses/LICENSE-FLUX1-dev) | 4610115bb0c89560703c892c59ac2742fa821e60ef5871b33493ba544683abd7 |
| `FLUX.1 [pro]`            | [Available in our API](https://docs.bfl.ml/).           |
| `FLUX1.1 [pro]`           | [Available in our API](https://docs.bfl.ml/).           |
| `FLUX1.1 [pro] Ultra/raw` | [Available in our API](https://docs.bfl.ml/).           |

## Open-weights usage

The weights will be downloaded automatically from HuggingFace once you start one of the demos. To download `FLUX.1 [dev]`, you will need to be logged in, see [here](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-login).
If you have downloaded the model weights manually, you can specify the downloaded paths via environment-variables:

```bash
export FLUX_SCHNELL=<path_to_flux_schnell_sft_file>
export FLUX_DEV=<path_to_flux_dev_sft_file>
export AE=<path_to_ae_sft_file>
```

For interactive sampling run

```bash
python -m flux --name <name> --loop
```

Or to generate a single sample run

```bash
python -m flux --name <name> \
  --height <height> --width <width> \
  --prompt "<prompt>"
```

### TRT engine infernece

You may also download ONNX exports of [FLUX.1 \[dev\]](https://huggingface.co/black-forest-labs/FLUX.1-dev-onnx) and [FLUX.1 \[schnell\]](https://huggingface.co/black-forest-labs/FLUX.1-schnell-onnx). We provide exports in BF16, FP8, and FP4 precision. Note that you need to install the repository with TensorRT support as outlined [here](../README.md).

```bash
TRT_ENGINE_DIR=<your_trt_engine_will_be_saved_here> ONNX_DIR=<path_of_downloaded_onnx_export> python src/flux/cli.py --prompt "<prompt>" --trt --static_shape=False --name=<name> --trt_transformer_precision <precision> --width 1344
```
where `<precision>` is either bf16, fp8, or fp4. For fp4, you need a NVIDIA GPU based on the [Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/). For ONNX exports, `height` and `width` have to be within 768 and 1344.

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
