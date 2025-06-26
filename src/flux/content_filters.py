import torch
from einops import rearrange
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, pipeline

PROMPT_IMAGE_INTEGRITY = """
Task: Analyze an image to identify potential copyright concerns or depictions of public figures.

Output: Respond with only "yes" or "no"

Criteria for "yes":
- The image contains a recognizable character from copyrighted media (movies, TV, comics, games, etc.)
- The image displays a trademarked logo or brand
- The image depicts a recognizable public figure (celebrities, politicians, athletes, influencers, historical figures, etc.)

Criteria for "no":
- All other cases
- When you cannot identify the specific copyrighted work or named individual

Critical Requirements:
1. You must be able to name the exact copyrighted work or specific person depicted
2. General references to demographics or characteristics are not sufficient
3. Base your decision solely on visual content, not interpretation
4. Provide only the one-word answer: "yes" or "no"
""".strip()


PROMPT_IMAGE_INTEGRITY_FOLLOW_UP = "Does this image have copyright concerns or includes public figures?"

PROMPT_TEXT_INTEGRITY = """
Task: Analyze a text prompt to identify potential copyright concerns or requests to depict living public figures.

Output: Respond with only "yes" or "no"

Criteria for "Yes":
- The prompt explicitly names a character from copyrighted media (movies, TV, comics, games, etc.)
- The prompt explicitly mentions a trademarked logo or brand
- The prompt names or describes a specific living public figure (celebrities, politicians, athletes, influencers, etc.)

Criteria for "No":
- All other cases
- When you cannot identify the specific copyrighted work or named individual

Critical Requirements:
1. You must be able to name the exact copyrighted work or specific person referenced
2. General demographic descriptions or characteristics are not sufficient
3. Analyze only the prompt text, not potential image outcomes
4. Provide only the one-word answer: "yes" or "no"

The prompt to check is:
-----
{prompt}
-----

Does this prompt have copyright concerns or includes public figures?
""".strip()


class PixtralContentFilter(torch.nn.Module):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        nsfw_threshold: float = 0.85,
    ):
        super().__init__()

        model_id = "mistral-community/pixtral-12b"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map=device)

        self.yes_token, self.no_token = self.processor.tokenizer.encode(["yes", "no"])

        self.nsfw_classifier = pipeline(
            "image-classification", model="Falconsai/nsfw_image_detection", device=device
        )
        self.nsfw_threshold = nsfw_threshold

    def yes_no_logit_processor(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Sets all tokens but yes/no to the minimum.
        """
        scores_yes_token = scores[:, self.yes_token].clone()
        scores_no_token = scores[:, self.no_token].clone()
        scores_min = scores.min()
        scores[:, :] = scores_min - 1
        scores[:, self.yes_token] = scores_yes_token
        scores[:, self.no_token] = scores_no_token
        return scores

    def test_image(self, image: Image.Image | str | torch.Tensor) -> bool:
        if isinstance(image, torch.Tensor):
            image = rearrange(image[0].clamp(-1.0, 1.0), "c h w -> h w c")
            image = Image.fromarray((127.5 * (image + 1.0)).cpu().byte().numpy())
        elif isinstance(image, str):
            image = Image.open(image)

        classification = next(c for c in self.nsfw_classifier(image) if c["label"] == "nsfw")
        if classification["score"] > self.nsfw_threshold:
            return True

        # 512^2 pixels are enough for checking
        w, h = image.size
        f = (512**2 / (w * h)) ** 0.5
        image = image.resize((int(f * w), int(f * h)))

        chat = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "content": PROMPT_IMAGE_INTEGRITY,
                    },
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "content": PROMPT_IMAGE_INTEGRITY_FOLLOW_UP,
                    },
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=[self.yes_no_logit_processor],
            do_sample=False,
        )
        return generate_ids[0, -1].item() == self.yes_token

    def test_txt(self, txt: str) -> bool:
        chat = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "content": PROMPT_TEXT_INTEGRITY.format(prompt=txt),
                    },
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=[self.yes_no_logit_processor],
            do_sample=False,
        )
        return generate_ids[0, -1].item() == self.yes_token
