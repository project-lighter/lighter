from pathlib import Path

import torch

from lighter.callbacks.writer.file import LighterFileWriter


import shutil

def test_file_writer_initialization():
    """Test LighterFileWriter initialization with proper attributes."""
    path = Path("test_dir")
    path.mkdir(exist_ok=True)  # Ensure the directory exists
    try:
        writer = LighterFileWriter(path=path, writer="tensor")
        assert writer.path == Path("test_dir")
        assert writer.writer == "tensor"  # Verify writer type
    finally:
        shutil.rmtree(path)  # Clean up after test

import pytest

def test_file_writer_write_tensor():
    """Test LighterFileWriter's ability to write and persist tensors correctly."""
    test_dir = Path("test_dir_tensor")
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

def test_file_writer_write_tensor_errors():
    """Test error handling in LighterFileWriter."""
    writer = LighterFileWriter(path="test_dir_errors", writer="tensor")
    
    # Test invalid tensor
    with pytest.raises(TypeError):
        writer.write("not a tensor", id=1)
    
    # Test invalid ID
    with pytest.raises(ValueError):
        writer.write(torch.tensor([1]), id=-1)
