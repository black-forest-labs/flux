## Open-weight models

We currently offer two open-weight text-to-image models.

| Name                      | HuggingFace repo                                                | License                                                               | sha256sum                                                        |
| ------------------------- | ----------------------------------------------------------------| --------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `FLUX.1 Kontext [dev]`    | https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev     | [FLUX.1-dev Non-Commercial License](../model_licenses/LICENSE-FLUX1-dev) | 843a26dc765d3105dba081c30bce7b14c65b0988f9e8d14e9fbc8856a6deebd5 |

## Examples

![FLUX.1 [dev] Grid](../assets/docs/kontext.png)

## Open-weights usage

The weights will be downloaded automatically to `checkpoints/` from HuggingFace once you start one of the demos. Alternatively, you may download the weights manually and put them in `checkpoints/`, or you can also manually link them with the following environment variables:
```bash
export FLUX_MODEL=<your model path here>
export FLUX_AE=<your autoencoder path here>
```

For interactive sampling run

```bash
python -m flux kontext --loop
```
Or to generate a single sample run

```bash
python -m flux kontext \
  --img_cond_path <path_to_input_image> \
  --prompt <your_prompt> \
  --num_steps 30 --aspect_ratio "16:9" --guidance 2.5 --seed 1
```
Note that the flags `num_steps`, `aspect_ratio`, `guidance` and `seed` are
optional. For more available flags see [the code](../src/flux/cli_kontext.py).

### TRT engine infernece

We provide exports in BF16, FP8, and FP4 precision. Note that you need to install the repository with TensorRT support as outlined [here](../README.md).

```bash
python -m flux kontext --loop --trt --trt_transformer_precision <precision>
```
where `<trt_transformer_precision>` is either `bf16`, `fp8`, or `fp4_sdvd32`.
