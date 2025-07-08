import torch

from lighter.callbacks.utils import preprocess_image


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
