import shutil
from pathlib import Path

import monai
import numpy as np
import pytest
import torch
from PIL import Image

from lighter.callbacks.writer.file import LighterFileWriter


def test_file_writer_initialization():
    """Test LighterFileWriter initialization with proper attributes."""
    path = Path("test_dir")
    path.mkdir(exist_ok=True)  # Ensure the directory exists
    try:
        writer = LighterFileWriter(path=path, writer="tensor")
        assert writer.path == Path("test_dir")
        assert writer.writer.__name__ == "write_tensor"  # Verify writer function
    finally:
        shutil.rmtree(path)  # Clean up after test


def test_file_writer_write_tensor():
    """Test LighterFileWriter's ability to write and persist tensors correctly."""
    test_dir = Path("test_dir")
    test_dir.mkdir(exist_ok=True)
    try:
        writer = LighterFileWriter(path=test_dir, writer="tensor")
        tensor = torch.tensor([1, 2, 3])
        writer.write(tensor, id=1)

        # Verify file exists
        saved_path = writer.path / "1.pt"
        assert saved_path.exists()

        # Verify tensor contents
        loaded_tensor = torch.load(saved_path)  # nosec B614
        assert torch.equal(loaded_tensor, tensor)
    finally:
        shutil.rmtree(test_dir)


def test_file_writer_write_image():
    """Test LighterFileWriter's ability to write and persist images correctly."""
    test_dir = Path("test_dir")
    test_dir.mkdir(exist_ok=True)
    try:
        writer = LighterFileWriter(path=test_dir, writer="image")
        tensor = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        writer.write(tensor, id="image_test")

        # Verify file exists
        saved_path = writer.path / "image_test.png"
        assert saved_path.exists()

        # Verify image contents
        image = Image.open(saved_path)
        image_array = np.array(image)
        assert image_array.shape == (64, 64, 3)
    finally:
        shutil.rmtree(test_dir)


def test_file_writer_write_video():
    """Test LighterFileWriter's ability to write and persist videos correctly."""
    test_dir = Path("test_dir")
    test_dir.mkdir(exist_ok=True)
    try:
        writer = LighterFileWriter(path=test_dir, writer="video")
        tensor = torch.randint(0, 256, (3, 10, 64, 64), dtype=torch.uint8)
        writer.write(tensor, id="video_test")

        # Verify file exists
        saved_path = writer.path / "video_test.mp4"
        assert saved_path.exists()
    finally:
        shutil.rmtree(test_dir)


def test_file_writer_write_grayscale_video():
    """Test LighterFileWriter's ability to convert grayscale video to RGB correctly."""
    test_dir = Path("test_dir")
    test_dir.mkdir(exist_ok=True)
    try:
        writer = LighterFileWriter(path=test_dir, writer="video")
        # Create a grayscale video tensor with 1 channel
        tensor = torch.randint(0, 256, (1, 10, 64, 64), dtype=torch.uint8)
        writer.write(tensor, id="grayscale_video_test")

        # Verify file exists
        saved_path = writer.path / "grayscale_video_test.mp4"
        assert saved_path.exists()
    finally:
        shutil.rmtree(test_dir)


def test_file_writer_write_itk_image():
    """Test LighterFileWriter's ability to write and persist ITK images correctly."""
    test_dir = Path("test_dir")
    test_dir.mkdir(exist_ok=True)
    try:
        writer = LighterFileWriter(path=test_dir, writer="itk_nrrd")
        tensor = torch.rand(1, 1, 64, 64, 64)  # Example 3D tensor
        meta_tensor = monai.data.MetaTensor(tensor, affine=torch.eye(4), meta={"original_channel_dim": 1})
        writer.write(meta_tensor, id="itk_image_test")

        # Verify file exists
        saved_path = writer.path / "itk_image_test.nrrd"
        assert saved_path.exists()
    finally:
        shutil.rmtree(test_dir)


def test_file_writer_invalid_directory():
    """Test LighterFileWriter raises an error when path is not a directory."""
    test_file = Path("test_file.txt")
    test_file.touch()  # Create a file instead of a directory
    try:
        with pytest.raises(RuntimeError, match="LighterFileWriter expects a directory path"):
            writer = LighterFileWriter(path=test_file, writer="tensor")
            writer.write(torch.tensor([1, 2, 3]), id=1)
    finally:
        test_file.unlink()  # Clean up the file after test

    with pytest.raises(RuntimeError):
        writer = LighterFileWriter(path=Path("invalid_dir"), writer="tensor")
        writer.write(torch.tensor([1, 2, 3]), id=1)
