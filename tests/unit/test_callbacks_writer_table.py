from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import torch
from pytorch_lightning import Trainer

from lighter.callbacks.writer.table import TableWriter
from lighter.system import System


def custom_writer(tensor):
    """
    Custom writer function that sums a tensor and returns it in a dictionary.

    Args:
        tensor (torch.Tensor): Input tensor to be processed

    Returns:
        dict: Dictionary with key 'custom' and value as the sum of the input tensor
    """
    return {"custom": tensor.sum().item()}


def test_table_writer_initialization():
    """
    Test proper initialization of TableWriter.

    Verifies that:
    - The writer is correctly instantiated with the given path
    - The path attribute is properly converted to a Path object
    """
    writer = TableWriter(path="test.csv", writer="tensor")
    assert writer.path == Path("test.csv")


def test_table_writer_custom_writer():
    """
    Test TableWriter with a custom writer function.

    Verifies that:
    - Custom writer function is properly integrated
    - Writer correctly processes tensor input using custom function
    - Resulting CSV records contain expected values
    """
    writer = TableWriter(path="test.csv", writer=custom_writer)
    test_tensor = torch.tensor([1, 2, 3])
    writer.write(tensor=test_tensor, id=1)
    assert writer.csv_records[0]["pred"] == {"custom": 6}


def test_table_writer_write():
    """
    Test TableWriter write functionality with various inputs.

    Tests:
    - Basic tensor writing with integer ID
    - Writing with string ID
    - Writing floating point tensors
    - CSV file creation and content verification
    - Proper handling of different tensor shapes and types

    File Operations:
    - Creates a temporary CSV file
    - Writes multiple records with different formats
    - Verifies file content matches expected records
    - Cleans up by removing the test file
    """
    test_file = Path("test.csv")
    writer = TableWriter(path="test.csv", writer="tensor")

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
    df["id"] = df["id"].astype(str)
    df["pred"] = df["pred"].apply(eval)

    for record in expected_records:
        row = df[df["id"] == str(record["id"])]
        assert not row.empty
        pred_value = row["pred"].iloc[0]  # get the value from the Series
        assert pred_value == record["pred"]

    # Cleanup
    test_file.unlink()


def test_table_writer_write_multi_process(tmp_path, monkeypatch):
    """
    Test TableWriter in a multi-process environment.

    Tests:
    - Writing records from multiple processes
    - Proper gathering of records across processes
    - Correct file creation and content verification in distributed setting

    Args:
        tmp_path (Path): Pytest fixture providing temporary directory path
        monkeypatch (MonkeyPatch): Pytest fixture for mocking

    Mocked Behaviors:
    - Simulates 2-process distributed environment
    - Mocks torch.distributed functions for testing
    - Simulates gathering of records from multiple ranks

    Verifies:
    - All records from different processes are properly gathered
    - CSV file contains correct combined records
    - Record order and content integrity is maintained
    """
    test_file = tmp_path / "test.csv"
    writer = TableWriter(path=test_file, writer="tensor")
    trainer = Trainer(max_epochs=1)

    # Expected records after gathering from all processes
    rank0_records = [{"id": 1, "pred": [1, 2, 3]}]  # records from rank 0
    rank1_records = [{"id": 2, "pred": [4, 5, 6]}]  # records from rank 1
    expected_records = rank0_records + rank1_records

    # Mock distributed functions for multi-process simulation
    def mock_gather(obj, gather_list, dst=0):
        if gather_list is not None:
            # Fill gather_list with records from each rank
            gather_list[0] = rank0_records
            gather_list[1] = rank1_records

    def mock_get_rank():
        return 0

    monkeypatch.setattr(torch.distributed, "gather_object", mock_gather)
    monkeypatch.setattr(torch.distributed, "get_rank", mock_get_rank)
    monkeypatch.setattr(trainer.strategy, "world_size", 2)

    writer.on_predict_epoch_end(trainer, mock.Mock())

    # Verify file creation
    assert test_file.exists()

    # Verify file content
    df = pd.read_csv(test_file)
    df["id"] = df["id"].astype(str)
    df["pred"] = df["pred"].apply(eval)

    # Check that all expected records are in the CSV
    for record in expected_records:
        row = df[df["id"] == str(record["id"])]
        assert not row.empty
        pred_value = row["pred"].iloc[0]
        assert pred_value == record["pred"]

    # Verify total number of records
    assert len(df) == len(expected_records)
