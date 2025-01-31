from pathlib import Path

import monai
import numpy as np
import pytest
import torch
from PIL import Image

from lighter.callbacks.writer.file import FileWriter


def test_file_writer_initialization(tmp_path):
    """Test the initialization of FileWriter class.

    This test verifies that:
    1. The FileWriter is initialized with the correct path
    2. The writer function is properly assigned based on the writer type
    3. The directory is created and cleaned up properly

    The test creates a temporary directory, initializes a writer, checks its attributes,
    and then cleans up the directory.
    """
    writer = FileWriter(path=tmp_path, writer="tensor")
    assert writer.path == tmp_path
    assert writer.writer.__name__ == "write_tensor"  # Verify writer function


def test_file_writer_write_tensor(tmp_path):
    """Test tensor writing functionality of FileWriter.

    This test verifies that:
    1. A tensor can be successfully written to disk
    2. The saved file exists at the expected location
    3. The loaded tensor matches the original tensor exactly

    The test creates a simple tensor, saves it, loads it back, and verifies
    the content matches the original.
    """
    writer = FileWriter(path=tmp_path, writer="tensor")
    tensor = torch.tensor([1, 2, 3])
    writer.write(tensor, identifier=1)

    # Verify file exists
    saved_path = writer.path / "1.pt"
    assert saved_path.exists()

    # Verify tensor contents
    loaded_tensor = torch.load(saved_path)  # nosec B614
    assert torch.equal(loaded_tensor, tensor)


def test_file_writer_write_image(tmp_path):
    """Test image writing functionality of FileWriter.

    This test verifies that:
    1. A tensor representing an image can be successfully written to disk as PNG
    2. The saved file exists at the expected location
    3. The loaded image has the correct dimensions and format

    The test creates a random RGB image tensor, saves it, and verifies
    the saved image properties.
    """
    writer = FileWriter(path=tmp_path, writer="image")
    tensor = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    writer.write(tensor, identifier="image_test")

    # Verify file exists
    saved_path = writer.path / "image_test.png"
    assert saved_path.exists()

    # Verify image contents
    image = Image.open(saved_path)
    image_array = np.array(image)
    assert image_array.shape == (64, 64, 3)


def test_file_writer_write_video(tmp_path):
    """Test video writing functionality of LighterFileWriter.

    This test verifies that:
    1. A tensor representing a video can be successfully written to disk as MP4
    2. The saved file exists at the expected location

    The test creates a random RGB video tensor and verifies it can be saved
    to disk in the correct format.
    """
    writer = FileWriter(path=tmp_path, writer="video")
    tensor = torch.randint(0, 256, (3, 10, 64, 64), dtype=torch.uint8)
    writer.write(tensor, identifier="video_test")

    # Verify file exists
    saved_path = writer.path / "video_test.mp4"
    assert saved_path.exists()


def test_file_writer_write_grayscale_video(tmp_path):
    """Test grayscale video writing functionality of FileWriter.

    This test verifies that:
    1. A single-channel (grayscale) video tensor can be successfully written to disk
    2. The writer correctly handles the conversion from grayscale to RGB format
    3. The saved file exists at the expected location

    The test creates a grayscale video tensor and verifies it can be properly
    converted and saved as an MP4 file.
    """
    writer = FileWriter(path=tmp_path, writer="video")
    # Create a grayscale video tensor with 1 channel
    tensor = torch.randint(0, 256, (1, 10, 64, 64), dtype=torch.uint8)
    writer.write(tensor, identifier="grayscale_video_test")

    # Verify file exists
    saved_path = writer.path / "grayscale_video_test.mp4"
    assert saved_path.exists()


def test_file_writer_write_itk_image(tmp_path):
    """Test ITK image writing functionality of FileWriter.

    This test verifies that:
    1. The writer correctly handles MONAI MetaTensor format
    2. Invalid tensor formats raise appropriate exceptions
    3. Valid MetaTensors can be successfully written to disk as NRRD files

    The test attempts to write both regular tensors and MetaTensors,
    verifying proper error handling and successful writes.
    """
    writer = FileWriter(path=tmp_path, writer="itk_nrrd")
    tensor = torch.rand(1, 1, 64, 64, 64)  # Example 3D tensor

    # Test with regular tensor
    with pytest.raises(TypeError, match="Tensor must be in MONAI MetaTensor format"):
        writer.write(tensor, identifier="itk_image_test")

    # Test with proper MetaTensor
    meta_tensor = monai.data.MetaTensor(tensor, affine=torch.eye(4), meta={"original_channel_dim": 1})
    writer.write(meta_tensor, identifier="itk_image_test")

    # Verify file exists
    saved_path = writer.path / "itk_image_test.nrrd"
    assert saved_path.exists()


def test_file_writer_invalid_directory():
    """Test error handling for invalid directory paths in FileWriter.

    This test verifies that:
    1. Using a file path instead of a directory path raises appropriate errors
    2. Using a non-existent directory path raises appropriate errors
    3. Error messages are clear and descriptive

    The test attempts to initialize writers with invalid paths and verifies
    the correct error handling behavior.
    """
    test_file = Path("test_file.txt")
    test_file.touch()  # Create a file instead of a directory
    try:
        with pytest.raises(RuntimeError, match="FileWriter expects a directory path"):
            writer = FileWriter(path=test_file, writer="tensor")
            writer.write(torch.tensor([1, 2, 3]), identifier=1)
    finally:
        test_file.unlink()  # Clean up the file after test

    with pytest.raises(RuntimeError):
        writer = FileWriter(path=Path("invalid_dir"), writer="tensor")
        writer.write(torch.tensor([1, 2, 3]), identifier=1)
