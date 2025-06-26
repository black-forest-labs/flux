![FLUX.1 [dev] Grid](../assets/docs/kontext.png)

`FLUX.1 Kontext [dev]` is a 12 billion parameter rectified flow transformer capable of editing images based on text instructions.
For more information, please read our [blog post](https://bfl.ai/announcements/flux-1-kontext-dev) and our [technical report](https://arxiv.org/abs/2506.15742).
You can find information about the `[pro]` version in [here](https://bfl.ai/models/flux-kontext).

# Key Features
1. Change existing images based on an edit instruction.
2. Have character, style and object reference without any finetuning.
3. Robust consistency allows users to refine an image through multiple successive edits with minimal visual drift.
4. Trained using guidance distillation, making `FLUX.1 Kontext [dev]` more efficient.
5. Open weights to drive new scientific research, and empower artists to develop innovative workflows.
6. Generated outputs can be used for personal, scientific, and commercial purposes, as described in the [FLUX.1 \[dev\] Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev).

# Usage
We provide a reference implementation of `FLUX.1 Kontext [dev]`, as well as sampling code, in a dedicated [github repository](https://github.com/black-forest-labs/flux).
Developers and creatives looking to build on top of `FLUX.1 Kontext [dev]` are encouraged to use this as a starting point.

`FLUX.1 Kontext [dev]` is also available in both [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [Diffusers](https://github.com/huggingface/diffusers).

## API Endpoints
The FLUX.1 Kontext models are also available via API from the following sources
- bfl.ai: https://docs.bfl.ai/
- DataCrunch: https://datacrunch.io/flux-kontext
- fal: https://fal.ai/flux-kontext
- Replicate: https://replicate.com/blog/flux-kontext
    - https://replicate.com/black-forest-labs/flux-kontext-dev
    - https://replicate.com/black-forest-labs/flux-kontext-pro
    - https://replicate.com/black-forest-labs/flux-kontext-max
- Runware: https://runware.ai/blog/introducing-flux1-kontext-instruction-based-image-editing-with-ai?utm_source=bfl
- TogetherAI: https://www.together.ai/models/flux-1-kontext-dev

---

# Risks

Risks
Black Forest Labs is committed to the responsible development of generative AI technology. Prior to releasing FLUX.1 Kontext, we evaluated and mitigated a number of risks in our models and services, including the generation of unlawful content. We implemented a series of pre-release mitigations to help prevent misuse by third parties, with additional post-release mitigations to help address residual risks:
1. **Pre-training mitigation**. We filtered pre-training data for multiple categories of “not safe for work” (NSFW) content to help prevent a user generating unlawful content in response to text prompts or uploaded images.
2. **Post-training mitigation.** We have partnered with the Internet Watch Foundation, an independent nonprofit organization dedicated to preventing online abuse, to filter known child sexual abuse material (CSAM) from post-training data. Subsequently, we undertook multiple rounds of targeted fine-tuning to provide additional mitigation against potential abuse. By inhibiting certain behaviors and concepts in the trained model, these techniques can help to prevent a user generating synthetic CSAM or nonconsensual intimate imagery (NCII) from a text prompt, or transforming an uploaded image into synthetic CSAM or NCII.
3. **Pre-release evaluation.** Throughout this process, we conducted multiple internal and external third-party evaluations of model checkpoints to identify further opportunities for improvement. The third-party evaluations—which included 21 checkpoints of FLUX.1 Kontext [pro] and [dev]—focused on eliciting CSAM and NCII through adversarial testing with text-only prompts, as well as uploaded images with text prompts. Next, we conducted a final third-party evaluation of the proposed release checkpoints, focused on text-to-image and image-to-image CSAM and NCII generation. The final FLUX.1 Kontext [pro] (as offered through the FLUX API only) and FLUX.1 Kontext [dev] (released as an open-weight model) checkpoints demonstrated very high resilience against violative inputs, and FLUX.1 Kontext [dev] demonstrated higher resilience than other similar open-weight models across these risk categories.  Based on these findings, we approved the release of the FLUX.1 Kontext [pro] model via API, and the release of the FLUX.1 Kontext [dev] model as openly-available weights under a non-commercial license to support third-party research and development.
4. **Inference filters.** We are applying multiple filters to intercept text prompts, uploaded images, and output images on the FLUX API for FLUX.1 Kontext [pro]. Filters for CSAM and NCII are provided by Hive, a third-party provider, and cannot be adjusted or removed by developers. We provide filters for other categories of potentially harmful content, including gore, which can be adjusted by developers based on their specific risk profile. Additionally, the repository for the open FLUX.1 Kontext [dev] model includes filters for illegal or infringing content. Filters or manual review must be used with the model under the terms of the FLUX.1 [dev] Non-Commercial License. We may approach known deployers of the FLUX.1 Kontext [dev] model at random to verify that filters or manual review processes are in place.
5. **Content provenance.** The FLUX API applies cryptographically-signed metadata to output content to indicate that images were produced with our model. Our API implements the Coalition for Content Provenance and Authenticity (C2PA) standard for metadata.
6. **Policies.** Access to our API and use of our models are governed by our Developer Terms of Service, Usage Policy, and FLUX.1 [dev] Non-Commercial License, which prohibit the generation of unlawful content or the use of generated content for unlawful, defamatory, or abusive purposes. Developers and users must consent to these conditions to access the FLUX Kontext models.
7. **Monitoring.** We are monitoring for patterns of violative use after release, and may ban developers who we detect intentionally and repeatedly violate our policies via the FLUX API. Additionally, we provide a dedicated email address (safety@blackforestlabs.ai) to solicit feedback from the community. We maintain a reporting relationship with organizations such as the Internet Watch Foundation and the National Center for Missing and Exploited Children, and we welcome ongoing engagement with authorities, developers, and researchers to share intelligence about emerging risks and develop effective mitigations.


# License
This model falls under the [FLUX.1 \[dev\] Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev).


# Citation

```bib
@misc{labs2025flux1kontextflowmatching,
      title={FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space}, Add commentMore actions
      author={Black Forest Labs and Stephen Batifol and Andreas Blattmann and Frederic Boesel and Saksham Consul and Cyril Diagne and Tim Dockhorn and Jack English and Zion English and Patrick Esser and Sumith Kulal and Kyle Lacey and Yam Levi and Cheng Li and Dominik Lorenz and Jonas Müller and Dustin Podell and Robin Rombach and Harry Saini and Axel Sauer and Luke Smith},
      year={2025},
      eprint={2506.15742},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2506.15742},
}
```
