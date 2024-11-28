import pytest
import torch
from pathlib import Path
from lighter.callbacks.writer.file import LighterFileWriter

def test_file_writer_initialization():
    writer = LighterFileWriter(path="test_dir", writer="tensor")
    assert writer.path == Path("test_dir")

def test_file_writer_write_tensor():
    writer = LighterFileWriter(path="test_dir", writer="tensor")
    tensor = torch.tensor([1, 2, 3])
    writer.write(tensor, id=1)
    assert (writer.path / "1.pt").exists()
