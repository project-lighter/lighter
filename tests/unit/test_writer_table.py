import pytest
import torch
from lighter.callbacks.writer.table import LighterTableWriter

def test_table_writer_initialization():
    writer = LighterTableWriter(path="test.csv", writer="tensor")
    assert writer.path == Path("test.csv")

def test_table_writer_write():
    writer = LighterTableWriter(path="test.csv", writer="tensor")
    writer.write(tensor=torch.tensor([1]), id=1)
    assert len(writer.csv_records) == 1
