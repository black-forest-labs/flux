import io
import os
import time
from pathlib import Path

import requests
from PIL import Image

API_URL = "https://api.bfl.ml"
API_ENDPOINTS = {
    "flux.1-pro": "flux-pro",
    "flux.1-dev": "flux-dev",
    "flux.1.1-pro": "flux-pro-1.1",
}


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
        # api inputs
        prompt: str,
        name: str = "flux.1.1-pro",
        width: int | None = None,
        height: int | None = None,
        num_steps: int | None = None,
        prompt_upsampling: bool | None = None,
        seed: int | None = None,
        guidance: float | None = None,
        interval: float | None = None,
        safety_tolerance: int | None = None,
        # behavior of this class
        validate: bool = True,
        launch: bool = True,
        api_key: str | None = None,
    ):
        """
        Manages an image generation request to the API.

        All parameters not specified will use the API defaults.

        Args:
            prompt: Text prompt for image generation.
            width: Width of the generated image in pixels. Must be a multiple of 32.
            height: Height of the generated image in pixels. Must be a multiple of 32.
            name: Which model version to use
            num_steps: Number of steps for the image generation process.
            prompt_upsampling: Whether to perform upsampling on the prompt.
            seed: Optional seed for reproducibility.
            guidance: Guidance scale for image generation.
            safety_tolerance: Tolerance level for input and output moderation.
                 Between 0 and 6, 0 being most strict, 6 being least strict.
            validate: Run input validation
            launch: Directly launches request
            api_key: Your API key if not provided by the environment

        Raises:
            ValueError: For invalid input, when `validate`
            ApiException: For errors raised from the API
        """
        if validate:
            if name not in API_ENDPOINTS.keys():
                raise ValueError(f"Invalid model {name}")
            elif width is not None and width % 32 != 0:
                raise ValueError(f"width must be divisible by 32, got {width}")
            elif width is not None and not (256 <= width <= 1440):
                raise ValueError(f"width must be between 256 and 1440, got {width}")
            elif height is not None and height % 32 != 0:
                raise ValueError(f"height must be divisible by 32, got {height}")
            elif height is not None and not (256 <= height <= 1440):
                raise ValueError(f"height must be between 256 and 1440, got {height}")
            elif num_steps is not None and not (1 <= num_steps <= 50):
                raise ValueError(f"steps must be between 1 and 50, got {num_steps}")
            elif guidance is not None and not (1.5 <= guidance <= 5.0):
                raise ValueError(f"guidance must be between 1.5 and 4, got {guidance}")
            elif interval is not None and not (1.0 <= interval <= 4.0):
                raise ValueError(f"interval must be between 1 and 4, got {interval}")
            elif safety_tolerance is not None and not (0 <= safety_tolerance <= 6.0):
                raise ValueError(f"safety_tolerance must be between 0 and 6, got {interval}")

            if name == "flux.1-dev":
                if interval is not None:
                    raise ValueError("Interval is not supported for flux.1-dev")
            if name == "flux.1.1-pro":
                if interval is not None or num_steps is not None or guidance is not None:
                    raise ValueError("Interval, num_steps and guidance are not supported for " "flux.1.1-pro")

        self.name = name
        self.request_json = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": num_steps,
            "prompt_upsampling": prompt_upsampling,
            "seed": seed,
            "guidance": guidance,
            "interval": interval,
            "safety_tolerance": safety_tolerance,
        }
        self.request_json = {key: value for key, value in self.request_json.items() if value is not None}

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
            f"{API_URL}/v1/{API_ENDPOINTS[self.name]}",
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
                f"{API_URL}/v1/get_result",
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
