![FLUX.1 [dev] Grid](../assets/dev_grid.jpg)

`FLUX.1 [dev]` is a 12 billion parameter rectified flow transformer capable of generating images from text descriptions.
For more information, please read our [blog post](https://blackforestlabs.ai/announcing-black-forest-labs/).

# Key Features
1. Cutting-edge output quality, second only to our state-of-the-art model `FLUX.1 [pro]`.
2. Competitive prompt following, matching the performance of closed source alternatives .
3. Trained using guidance distillation, making `FLUX.1 [dev]` more efficient.
4. Open weights to drive new scientific research, and empower artists to develop innovative workflows.
5. Generated outputs can be used for personal, scientific, and commercial purposes, as described in the [flux-1-dev-non-commercial-license](./licence.md).

# Usage
We provide a reference implementation of `FLUX.1 [dev]`, as well as sampling code, in a dedicated [github repository](https://github.com/black-forest-labs/flux).
Developers and creatives looking to build on top of `FLUX.1 [dev]` are encouraged to use this as a starting point.

## API Endpoints
The FLUX.1 models are also available via API from the following sources
1. [bfl.ml](https://docs.bfl.ml/) (currently `FLUX.1 [pro]`)
2. [replicate.com](https://replicate.com/collections/flux)
3. [fal.ai](https://fal.ai/models/fal-ai/flux/dev)

## ComfyUI
`FLUX.1 [dev]` is also available in [Comfy UI](https://github.com/comfyanonymous/ComfyUI) for local inference with a node-based workflow.

---
# Limitations
- This model is not intended or able to provide factual information.
- As a statistical model this checkpoint might amplify existing societal biases.
- The model may fail to generate output that matches the prompts.
- Prompt following is heavily influenced by the prompting-style.

# Out-of-Scope Use
The model and its derivatives may not be used

- In any way that violates any applicable national, federal, state, local or international law or regulation.
- For the purpose of exploiting, harming or attempting to exploit or harm minors in any way; including but not limited to the solicitation, creation, acquisition, or dissemination of child exploitative content.
- To generate or disseminate verifiably false information and/or content with the purpose of harming others.
- To generate or disseminate personal identifiable information that can be used to harm an individual.
- To harass, abuse, threaten, stalk, or bully individuals or groups of individuals.
- To create non-consensual nudity or illegal pornographic content.
- For fully automated decision making that adversely impacts an individual's legal rights or otherwise creates or modifies a binding, enforceable obligation.
- Generating or facilitating large-scale disinformation campaigns.

# License
This model falls under the [`FLUX.1 [dev]` Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
