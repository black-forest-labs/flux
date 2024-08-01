import io
import os
import time
from pathlib import Path

import requests
from PIL import Image

API_ENDPOINT = "https://api.bfl.ml"


class ApiException(Exception):
    def __init__(self, status_code: int, detail: str | list[dict] | None = None):
        super().__init__()
        self.detail = detail
        self.status_code = status_code

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        if self.detail is None:
            message = None
        elif isinstance(self.detail, str):
            message = self.detail
        else:
            message = "[" + ",".join(d["msg"] for d in self.detail) + "]"
        return f"ApiException({self.status_code=}, {message=}, detail={self.detail})"


class ImageRequest:
    def __init__(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        name: str = "flux.1-pro",
        num_steps: int = 50,
        prompt_upsampling: bool = False,
        seed: int | None = None,
        validate: bool = True,
        launch: bool = True,
        api_key: str | None = None,
    ):
        """
        Manages an image generation request to the API.

        Args:
            prompt: Prompt to sample
            width: Width of the image in pixel
            height: Height of the image in pixel
            name: Name of the model
            num_steps: Number of network evaluations
            prompt_upsampling: Use prompt upsampling
            seed: Fix the generation seed
            validate: Run input validation
            launch: Directly launches request
            api_key: Your API key if not provided by the environment

        Raises:
            ValueError: For invalid input
            ApiException: For errors raised from the API
        """
        if validate:
            if name not in ["flux.1-pro"]:
                raise ValueError(f"Invalid model {name}")
            elif width % 32 != 0:
                raise ValueError(f"width must be divisible by 32, got {width}")
            elif not (256 <= width <= 1440):
                raise ValueError(f"width must be between 256 and 1440, got {width}")
            elif height % 32 != 0:
                raise ValueError(f"height must be divisible by 32, got {height}")
            elif not (256 <= height <= 1440):
                raise ValueError(f"height must be between 256 and 1440, got {height}")
            elif not (1 <= num_steps <= 50):
                raise ValueError(f"steps must be between 1 and 50, got {num_steps}")

        self.request_json = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "variant": name,
            "steps": num_steps,
            "prompt_upsampling": prompt_upsampling,
        }
        if seed is not None:
            self.request_json["seed"] = seed

        self.request_id: str | None = None
        self.result: dict | None = None
        self._image_bytes: bytes | None = None
        self._url: str | None = None
        if api_key is None:
            self.api_key = os.environ.get("BFL_API_KEY")
        else:
            self.api_key = api_key

        if launch:
            self.request()

    def request(self):
        """
        Request to generate the image.
        """
        if self.request_id is not None:
            return
        response = requests.post(
            f"{API_ENDPOINT}/v1/image",
            headers={
                "accept": "application/json",
                "x-key": self.api_key,
                "Content-Type": "application/json",
            },
            json=self.request_json,
        )
        result = response.json()
        if response.status_code != 200:
            raise ApiException(status_code=response.status_code, detail=result.get("detail"))
        self.request_id = response.json()["id"]

    def retrieve(self) -> dict:
        """
        Wait for the generation to finish and retrieve response.
        """
        if self.request_id is None:
            self.request()
        while self.result is None:
            response = requests.get(
                f"{API_ENDPOINT}/v1/get_result",
                headers={
                    "accept": "application/json",
                    "x-key": self.api_key,
                },
                params={
                    "id": self.request_id,
                },
            )
            result = response.json()
            if "status" not in result:
                raise ApiException(status_code=response.status_code, detail=result.get("detail"))
            elif result["status"] == "Ready":
                self.result = result["result"]
            elif result["status"] == "Pending":
                time.sleep(0.5)
            else:
                raise ApiException(status_code=200, detail=f"API returned status '{result['status']}'")
        return self.result

    @property
    def bytes(self) -> bytes:
        """
        Generated image as bytes.
        """
        if self._image_bytes is None:
            response = requests.get(self.url)
            if response.status_code == 200:
                self._image_bytes = response.content
            else:
                raise ApiException(status_code=response.status_code)
        return self._image_bytes

    @property
    def url(self) -> str:
        """
        Public url to retrieve the image from
        """
        if self._url is None:
            result = self.retrieve()
            self._url = result["sample"]
        return self._url

    @property
    def image(self) -> Image.Image:
        """
        Load the image as a PIL Image
        """
        return Image.open(io.BytesIO(self.bytes))

    def save(self, path: str):
        """
        Save the generated image to a local path
        """
        suffix = Path(self.url).suffix
        if not path.endswith(suffix):
            path = path + suffix
        Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            file.write(self.bytes)


if __name__ == "__main__":
    from fire import Fire

    Fire(ImageRequest)
