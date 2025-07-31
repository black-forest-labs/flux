## Models

FLUX.1 Redux is an adapter for the FLUX.1 text-to-image base models, FLUX.1 [dev] and FLUX.1 [schnell], which can be used to generate image variations.

| Name                        | HuggingFace repo                                            | License                                                               | sha256sum                                                        |
| --------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `FLUX.1 Redux [dev]`        | https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev   | [FLUX.1-dev Non-Commercial License](../model_licenses/LICENSE-FLUX1-dev) | a1b3bdcb4bdc58ce04874b9ca776d61fc3e914bb6beab41efb63e4e2694dca45 |


## Examples

![redux](../assets/docs/redux.png)

## Open-weights usage

The weights will be downloaded automatically to `checkpoints/` from HuggingFace once you start one of the demos. Alternatively, you may download the weights manually and put them in `checkpoints/`, or you can also manually link them with the following environment variables:
```bash
export FLUX_MODEL=<your model path here>
export FLUX_REDUX=<your redux path here>
export FLUX_AE=<your autoencoder path here>
```

For interactive sampling run:

```bash
python -m flux redux --name <name> --loop
```

where `name` specifies the base model, which should be one of `flux-dev` or `flux-schnell`.
