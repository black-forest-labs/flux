from types import SimpleNamespace

import numpy as np
import pytest
import torch

from demo_gr import FluxGenerator


class EncodingReached(Exception):
    pass


def encode_input(image, width, height):
    generator = FluxGenerator.__new__(FluxGenerator)
    generator.device = torch.device("cpu")
    generator.offload = False
    captured = []

    def encode(tensor):
        captured.append(tensor.clone())
        raise EncodingReached

    generator.ae = SimpleNamespace(encode=encode)
    with pytest.raises(EncodingReached):
        generator.generate_image(width, height, 1, 1.0, 0, "test", init_image=image)
    return captured[0]


def test_numpy_pixels_are_normalized_before_encoding():
    image = np.array([[[0, 128, 255], [255, 0, 128]]], dtype=np.uint8)
    actual = encode_input(image, 2, 1)
    expected = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_normalized_numpy_image_is_resized_in_nchw_layout():
    image = np.array([[[0, 128, 255], [255, 0, 128]]], dtype=np.uint8)
    actual = encode_input(image, 4, 2)
    expected = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    expected = torch.nn.functional.interpolate(expected, (2, 4))
    assert actual.shape == (1, 3, 2, 4)
    torch.testing.assert_close(actual, expected)


def test_pre_normalized_tensor_is_not_normalized_again():
    image = torch.tensor([-1.0, 0.0, 1.0]).reshape(1, 3, 1, 1)
    torch.testing.assert_close(encode_input(image, 1, 1), image)
