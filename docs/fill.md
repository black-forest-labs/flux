## Models

FLUX.1 Fill introduces advanced inpainting and outpainting capabilities. It allows for seamless edits that integrate naturally with existing images.

| Name                | HuggingFace repo                                         | License                                                               | sha256sum                                                        |
| ------------------- | -------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `FLUX.1 Fill [dev]` | https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev | [FLUX.1-dev Non-Commercial License](model_licenses/LICENSE-FLUX1-dev) | 03e289f530df51d014f48e675a9ffa2141bc003259bf5f25d75b957e920a41ca |
| `FLUX.1 Fill [pro]` | Only available in our API.                               |

## Examples

![inpainting](../assets/docs/inpainting.png)
![outpainting](../assets/docs/outpainting.png)

## Open-weights usage

The weights will be downloaded automatically from HuggingFace once you start one of the demos. To download `FLUX.1 Fill [dev]`, you will need to be logged in, see [here](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-login). Alternatively, if you have downloaded the model weights manually from [here](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev), you can specify the downloaded paths via environment variables:

```bash
export FLUX_DEV_FILL=<path_to_flux_dev_fill_sft_file>
export AE=<path_to_ae_sft_file>
```

For interactive sampling run

```bash
python -m src.flux.cli_fill --loop
```

Or to generate a single sample run

```bash
python -m src.flux.cli_fill \
  --img_cond_path <path_to_input_image> \
  --img_mask_path <path_to_input_mask>
```

The input_mask should be an image of the same size as the conditioning image that only contains black and white pixels; see [an example mask](../assets/cup_mask.png) for [this image](../assets/cup.png).

We also provide an interactive streamlit demo. The demo can be run via

```bash
streamlit run demo_st_fill.py
```
