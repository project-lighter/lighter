from unittest import mock
from pathlib import Path
import pandas as pd
import pytest
import torch
from pytorch_lightning import Trainer
from lighter.callbacks.writer.table import LighterTableWriter
from lighter.system import LighterSystem

def custom_writer(tensor):
    return {"custom": tensor.sum().item()}

def test_table_writer_initialization():
    writer = LighterTableWriter(path="test.csv", writer="tensor")
    assert writer.path == Path("test.csv")

def test_table_writer_custom_writer():
    writer = LighterTableWriter(path="test.csv", writer=custom_writer)
    test_tensor = torch.tensor([1, 2, 3])
    writer.write(tensor=test_tensor, id=1)
    assert writer.csv_records[0]["pred"] == {"custom": 6}

def test_table_writer_distributed_gather(tmp_path, monkeypatch):
    writer = LighterTableWriter(path=tmp_path / "test.csv", writer="tensor")
    trainer = Trainer(max_epochs=1)
    
    # Mock the distributed environment methods
    monkeypatch.setattr(trainer, "world_size", 2)
    monkeypatch.setattr(trainer, "is_global_zero", True)
    writer.csv_records = [{"id": 1, "pred": [1, 2, 3]}]
    writer.on_predict_epoch_end(trainer, mock.Mock())
    assert (tmp_path / "test.csv").exists()

def test_table_writer_edge_cases():
    writer = LighterTableWriter(path="test.csv", writer="tensor")
    writer.write(tensor=torch.tensor([]), id=2)  # empty tensor
    writer.write(tensor=torch.randn(1000), id=3)  # large tensor
    writer.write(tensor=torch.tensor([1.5, 2.5]), id=4)  # float tensor


def test_table_writer_write():
    """Test LighterTableWriter write functionality with various inputs."""
    test_file = Path("test.csv")
    writer = LighterTableWriter(path="test.csv", writer="tensor")
    
    expected_records = [
        {"id": 1, "pred": [1, 2, 3]},
        {"id": "some_id", "pred": -1},
        {"id": 1331, "pred": [1.5, 2.5]},
    ]
    # Test basic write
    writer.write(tensor=torch.tensor(expected_records[0]["pred"]), id=expected_records[0]["id"])
    assert len(writer.csv_records) == 1
    assert writer.csv_records[0]["pred"] == expected_records[0]["pred"]
    assert writer.csv_records[0]["id"] == expected_records[0]["id"]
    
    # Test edge cases
    writer.write(tensor=torch.tensor(expected_records[1]["pred"]), id=expected_records[1]["id"])
    writer.write(tensor=torch.tensor(expected_records[2]["pred"]), id=expected_records[2]["id"])
    trainer = Trainer(max_epochs=1)
    writer.on_predict_epoch_end(trainer, mock.Mock())

    # Verify file creation and content
    assert test_file.exists()
    df = pd.read_csv(test_file)
    for record in expected_records:
        row = df[df['id'] == record['id']]
        assert not row.empty
        assert row['pred'].tolist() == record['pred']
    
    # Cleanup
    test_file.unlink()
