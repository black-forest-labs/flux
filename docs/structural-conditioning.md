## Models

Structural conditioning uses canny edge or depth detection to maintain precise control during image transformations. By preserving the original image's structure through edge or depth maps, users can make text-guided edits while keeping the core composition intact. This is particularly effective for retexturing images. We release four variations: two based on edge maps (full model and LoRA for FLUX.1 [dev]) and two based on depth maps (full model and LoRA for FLUX.1 [dev]).

| Name                      | HuggingFace repo                                               | License                                                               | sha256sum                                                        |
| ------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `FLUX.1 Canny [dev]`      | https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev      | [FLUX.1-dev Non-Commercial License](../model_licenses/LICENSE-FLUX1-dev) | 996876670169591cb412b937fbd46ea14cbed6933aef17c48a2dcd9685c98cdb |
| `FLUX.1 Depth [dev]`      | https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev      | [FLUX.1-dev Non-Commercial License](../model_licenses/LICENSE-FLUX1-dev) | 41360d1662f44ca45bc1b665fe6387e91802f53911001630d970a4f8be8dac21 |
| `FLUX.1 Canny [dev] LoRA` | https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora | [FLUX.1-dev Non-Commercial License](../model_licenses/LICENSE-FLUX1-dev) | 8eaa21b9c43d5e7242844deb64b8cf22ae9010f813f955ca8c05f240b8a98f7e |
| `FLUX.1 Depth [dev] LoRA` | https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora | [FLUX.1-dev Non-Commercial License](../model_licenses/LICENSE-FLUX1-dev) | 1938b38ea0fdd98080fa3e48beb2bedfbc7ad102d8b65e6614de704a46d8b907 |

## Examples

![canny](../assets/docs/canny.png)
![depth](../assets/docs/depth.png)

## Open-weights usage

The weights will be downloaded automatically to `checkpoints/` from HuggingFace once you start one of the demos. Alternatively, you may download the weights manually and put them in `checkpoints/`, or you can also manually link them with the following environment variables:
```bash
export FLUX_MODEL=<your model path here>
export FLUX_AE=<your autoencoder path here>

# optional (see below)
export FLUX_LORA=<your lora path here>
```

Note that the LoRA models (`flux-dev-canny-lora` and `flux-dev-depth-lora`) require the base FLUX.1 [dev] model to be downloaded first. The system will automatically download both the base model and the LoRA adapter when using these variants.

For interactive sampling run

```bash
python -m flux control --name <name> --loop
```

where `name` is one of `flux-dev-canny`, `flux-dev-depth`, `flux-dev-canny-lora`, or `flux-dev-depth-lora`.

### TRT engine inference

 We provide exports in BF16, FP8, and FP4 precision. Note that you need to install the repository with TensorRT support as outlined [here](../README.md).

```bash
python flux control --name=<name> --loop --img_cond_path="assets/robot.webp" --trt --static_shape=False --trt_transformer_precision <precision>
```
where `<precision>` is either `bf16`, `fp8`, or `fp4`.

## Diffusers usage

Flux Control (including the LoRAs) is also compatible with the `diffusers` Python library. Check out the [documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux) to learn more.
