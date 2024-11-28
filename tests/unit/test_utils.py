import pytest
import torch
from lighter.callbacks.utils import preprocess_image

def test_preprocess_image_2d():
    image = torch.rand(1, 3, 64, 64)
    processed_image = preprocess_image(image)
    assert processed_image.shape == (3, 64, 64)

def test_preprocess_image_3d():
    image = torch.rand(1, 3, 4, 64, 64)
    processed_image = preprocess_image(image)
    assert processed_image.shape[0] == 3
