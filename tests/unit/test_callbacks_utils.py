import pytest
import torch

from lighter.callbacks.utils import get_lighter_mode, preprocess_image


def test_get_lighter_mode():
    assert get_lighter_mode("train") == "train"
    assert get_lighter_mode("validate") == "val"
    assert get_lighter_mode("test") == "test"
    with pytest.raises(KeyError):
        get_lighter_mode("invalid_stage")
    image = torch.rand(1, 3, 64, 64)  # Batch of 2D images
    processed_image = preprocess_image(image)
    assert processed_image.shape == (3, 64, 64)


def test_preprocess_image_single_3d():
    depth = 20
    height = 64
    width = 64
    image = torch.rand(1, 1, depth, height, width)  # Single 3D image
    processed_image = preprocess_image(image)
    assert processed_image.shape == (1, depth * height, width)


def test_preprocess_image_batch_3d():
    batch_size = 8
    depth = 20
    height = 64
    width = 64
    image = torch.rand(batch_size, 1, depth, height, width)  # Batch of 3D images
    processed_image = preprocess_image(image)
    assert processed_image.shape == (1, depth * height, batch_size * width)
