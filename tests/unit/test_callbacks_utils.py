import pytest
import torch

from lighter.callbacks.utils import get_lighter_mode, preprocess_image


def test_get_lighter_mode():
    """
    Test the get_lighter_mode function's stage name mapping and error handling.

    Tests:
        - Mapping of 'train' stage
        - Mapping of 'validate' stage to 'val'
        - Mapping of 'test' stage
        - Raising KeyError for invalid stage names
    """
    assert get_lighter_mode("train") == "train"
    assert get_lighter_mode("validate") == "val"
    assert get_lighter_mode("test") == "test"
    with pytest.raises(KeyError):
        get_lighter_mode("invalid_stage")
    image = torch.rand(1, 3, 64, 64)  # Batch of 2D images
    processed_image = preprocess_image(image)
    assert processed_image.shape == (3, 64, 64)


def test_preprocess_image_single_3d():
    """
    Test preprocess_image function with a single 3D image input.

    Tests the reshaping of a single 3D image with dimensions:
    - Input: (1, 1, depth, height, width)
    - Expected output: (1, depth*height, width)

    The function verifies that a 3D medical image is correctly
    reshaped while preserving spatial relationships.
    """
    depth = 20
    height = 64
    width = 64
    image = torch.rand(1, 1, depth, height, width)  # Single 3D image
    processed_image = preprocess_image(image)
    assert processed_image.shape == (1, depth * height, width)


def test_preprocess_image_batch_3d():
    """
    Test preprocess_image function with a batch of 3D images.

    Tests the reshaping of multiple 3D images with dimensions:
    - Input: (batch_size, 1, depth, height, width)
    - Expected output: (1, depth*height, batch_size*width)

    The function verifies that a batch of 3D medical images is correctly
    reshaped into a single 2D representation while maintaining the
    spatial relationships and batch information.

    Args used in test:
        batch_size: 8
        depth: 20
        height: 64
        width: 64
    """
    batch_size = 8
    depth = 20
    height = 64
    width = 64
    image = torch.rand(batch_size, 1, depth, height, width)  # Batch of 3D images
    processed_image = preprocess_image(image)
    assert processed_image.shape == (1, depth * height, batch_size * width)
