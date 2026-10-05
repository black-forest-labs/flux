from dataclasses import fields
from importlib import import_module

import pytest
from PIL import Image


@pytest.fixture(
    params=[
        ("control", "img_cond_path"),
        ("redux", "img_cond_path"),
        ("kontext", "img_cond_path"),
        ("fill", "img_cond_path"),
        ("fill", "img_mask_path"),
    ]
)
def image_parser(request, tmp_path):
    variant, attribute = request.param
    module = import_module(f"flux.cli_{variant}")
    image = tmp_path / "input image.png"
    Image.new("RGB", (32, 32)).save(image)
    values = dict(
        prompt="test",
        width=32,
        height=32,
        num_steps=1,
        guidance=1.0,
        seed=0,
        img_cond_path=str(image),
        img_mask_path=str(image),
        lora_scale=1.0,
    )
    options = module.SamplingOptions(
        **{field.name: values[field.name] for field in fields(module.SamplingOptions)}
    )
    return getattr(module, f"parse_{attribute}"), options, attribute, image


@pytest.mark.parametrize("absolute", [True, False])
def test_image_parser_accepts_existing_paths(image_parser, monkeypatch, absolute):
    parser, options, attribute, image = image_parser
    monkeypatch.chdir(image.parent)
    path = str(image) if absolute else image.name
    answers = iter([path, "/q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert parser(options) is options
    assert getattr(options, attribute) == path


@pytest.mark.parametrize("answers", [["/q"], [""], ["/h", ""]])
def test_image_parser_preserves_commands_and_repeat(image_parser, monkeypatch, answers):
    parser, options, _, _ = image_parser
    responses = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    assert parser(options) is (None if answers == ["/q"] else options)


def test_image_parser_keeps_file_validation(image_parser, monkeypatch, capsys):
    parser, options, attribute, image = image_parser
    previous = getattr(options, attribute)
    invalid = image.with_suffix(".txt")
    invalid.write_text("not an image")
    responses = iter([str(invalid), "/q"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    assert parser(options) is None
    assert getattr(options, attribute) == previous
    assert "not a valid image file" in capsys.readouterr().out


@pytest.mark.parametrize("size", [(31, 32), (64, 32)])
def test_fill_mask_keeps_dimension_validation(tmp_path, monkeypatch, capsys, size):
    from flux.cli_fill import SamplingOptions, parse_img_mask_path

    image, mask = tmp_path / "image.png", tmp_path / "mask.png"
    Image.new("RGB", (32, 32)).save(image)
    Image.new("RGB", size).save(mask)
    options = SamplingOptions("test", 32, 32, 1, 1.0, 0, str(image), str(image))
    responses = iter([str(mask), "/q"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    assert parse_img_mask_path(options) is None
    assert options.img_mask_path == str(image)
    assert "dimensions must" in capsys.readouterr().out
