# FLUX
by Black Forest Labs: https://blackforestlabs.ai. Documentation for our API can be found here: [docs.bfl.ml](https://docs.bfl.ml/).

![grid](assets/grid.jpg)

This repo contains minimal inference code to run image generation & editing with our Flux models.

## Local installation

```bash
cd $HOME && git clone https://github.com/black-forest-labs/flux
cd $HOME/flux

# Using pyvenv
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

### Models

We are offering an extensive suite of models. For more information about the individual models, please refer to the link under **Usage**.

| Name                        | Usage                                                      | HuggingFace repo                                               | License                                                               |
| --------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------- |
| `FLUX.1 [schnell]`          | [Text to Image](docs/text-to-image.md)                     | https://huggingface.co/black-forest-labs/FLUX.1-schnell        | [apache-2.0](model_licenses/LICENSE-FLUX1-schnell)                    |
| `FLUX.1 [dev]`              | [Text to Image](docs/text-to-image.md)                     | https://huggingface.co/black-forest-labs/FLUX.1-dev            | [FLUX.1-dev Non-Commercial License](model_licenses/LICENSE-FLUX1-dev) |
| `FLUX.1 Fill [dev]`         | [In/Out-painting](docs/fill.md)                            | https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev       | [FLUX.1-dev Non-Commercial License](model_licenses/LICENSE-FLUX1-dev) |
| `FLUX.1 Canny [dev]`        | [Structural Conditioning](docs/structural-conditioning.md) | https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev      | [FLUX.1-dev Non-Commercial License](model_licenses/LICENSE-FLUX1-dev) |
| `FLUX.1 Depth [dev]`        | [Structural Conditioning](docs/structural-conditioning.md) | https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev      | [FLUX.1-dev Non-Commercial License](model_licenses/LICENSE-FLUX1-dev) |
| `FLUX.1 Canny [dev] LoRA`   | [Structural Conditioning](docs/structural-conditioning.md) | https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora | [FLUX.1-dev Non-Commercial License](model_licenses/LICENSE-FLUX1-dev) |
| `FLUX.1 Depth [dev] LoRA`   | [Structural Conditioning](docs/structural-conditioning.md) | https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora | [FLUX.1-dev Non-Commercial License](model_licenses/LICENSE-FLUX1-dev) |
| `FLUX.1 Redux [dev]`        | [Image variation](docs/image-variation.md)                 | https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev      | [FLUX.1-dev Non-Commercial License](model_licenses/LICENSE-FLUX1-dev) |
| `FLUX.1 [pro]`              | [Text to Image](docs/text-to-image.md)                     | [Available in our API.](https://docs.bfl.ml/)                  |                                                                       |
| `FLUX1.1 [pro]`             | [Text to Image](docs/text-to-image.md)                     | [Available in our API.](https://docs.bfl.ml/)                  |                                                                       |
| `FLUX1.1 [pro] Ultra/raw`   | [Text to Image](docs/text-to-image.md)                     | [Available in our API.](https://docs.bfl.ml/)                  |                                                                       |
| `FLUX.1 Fill [pro]`         | [In/Out-painting](docs/fill.md)                            | [Available in our API.](https://docs.bfl.ml/)                  |                                                                       |
| `FLUX.1 Canny [pro]`        | [Structural Conditioning](docs/structural-conditioning.md) | [Available in our API.](https://docs.bfl.ml/)                  |                                                                       |
| `FLUX.1 Depth [pro]`        | [Structural Conditioning](docs/structural-conditioning.md) | [Available in our API.](https://docs.bfl.ml/)                  |                                                                       |
| `FLUX1.1 Redux [pro]`       | [Image variation](docs/image-variation.md)                 | [Available in our API.](https://docs.bfl.ml/)                  |                                                                       |
| `FLUX1.1 Redux [pro] Ultra` | [Image variation](docs/image-variation.md)                 | [Available in our API.](https://docs.bfl.ml/)                  |                                                                       |

The weights of the autoencoder are also released under [apache-2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) and can be found in the HuggingFace repos above.

## API usage

Our API offers access to our models. It is documented here:
[docs.bfl.ml](https://docs.bfl.ml/).

In this repository we also offer an easy python interface. To use this, you
first need to register with the API on [api.bfl.ml](https://api.bfl.ml/), and
create a new API key.

To use the API key either run `export BFL_API_KEY=<your_key_here>` or provide
it via the `api_key=<your_key_here>` parameter. It is also expected that you
have installed the package as above.

Usage from python:

```python
from flux.api import ImageRequest

# this will create an api request directly but not block until the generation is finished
request = ImageRequest("A beautiful beach", name="flux.1.1-pro")
# or: request = ImageRequest("A beautiful beach", name="flux.1.1-pro", api_key="your_key_here")

# any of the following will block until the generation is finished
request.url
# -> https:<...>/sample.jpg
request.bytes
# -> b"..." bytes for the generated image
request.save("outputs/api.jpg")
# saves the sample to local storage
request.image
# -> a PIL image
```

Usage from the command line:

```bash
$ python -m flux.api --prompt="A beautiful beach" url
https:<...>/sample.jpg

# generate and save the result
$ python -m flux.api --prompt="A beautiful beach" save outputs/api

# open the image directly
$ python -m flux.api --prompt="A beautiful beach" image show
```

## Citation

If you find the provided code or models useful for your research, consider citing them as:

```bib
@misc{flux2023,
    author={Black Forest Labs},
    title={FLUX},
    year={2023},
    howpublished={\url{https://github.com/black-forest-labs/flux}},
}
```
