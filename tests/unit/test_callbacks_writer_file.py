import shutil
from pathlib import Path

import torch

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
        loaded_tensor = torch.load(saved_path)
        assert torch.equal(loaded_tensor, tensor)
    finally:
        shutil.rmtree(test_dir)
