import torch

from lighter.callbacks.utils import preprocess_image


def test_preprocess_image_2d():
    image = torch.rand(1, 3, 64, 64)  # Batch of 2D images
    processed_image = preprocess_image(image)
    assert processed_image.shape == (3, 64, 64)


def test_preprocess_image_3d():
    batch_size = 8
    depth = 20
    height = 64
    width = 64
    image = torch.rand(batch_size, 1, depth, height, width)  # Batch of 3D images
    processed_image = preprocess_image(image)
    assert processed_image.shape == (1, depth * height, batch_size * width)
