from pathlib import Path
from unittest import mock

import pandas as pd
import torch
from pytorch_lightning import Trainer

from lighter.callbacks.writer.table import TableWriter


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
    writer.write(tensor=test_tensor, identifier=1)
    assert writer.csv_records[0]["pred"] == {"custom": 6}


def test_table_writer_write():
    """
    Test TableWriter write functionality with various inputs.

    Tests:
    - Basic tensor writing with integer IDENTIFIER
    - Writing with string IDENTIFIER
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
        {"identifier": 1, "pred": [1, 2, 3]},
        {"identifier": "some_id", "pred": -1},
        {"identifier": 1331, "pred": [1.5, 2.5]},
    ]
    # Test basic write
    writer.write(tensor=torch.tensor(expected_records[0]["pred"]), identifier=expected_records[0]["identifier"])
    assert len(writer.csv_records) == 1
    assert writer.csv_records[0]["pred"] == expected_records[0]["pred"]
    assert writer.csv_records[0]["identifier"] == expected_records[0]["identifier"]

    # Test edge cases
    writer.write(tensor=torch.tensor(expected_records[1]["pred"]), identifier=expected_records[1]["identifier"])
    writer.write(tensor=torch.tensor(expected_records[2]["pred"]), identifier=expected_records[2]["identifier"])
    trainer = Trainer(max_epochs=1)
    writer.on_predict_epoch_end(trainer, mock.Mock())

    # Verify file creation and content
    assert test_file.exists()
    df = pd.read_csv(test_file)
    df["identifier"] = df["identifier"].astype(str)
    df["pred"] = df["pred"].apply(eval)

    for record in expected_records:
        row = df[df["identifier"] == str(record["identifier"])]
        assert not row.empty
        pred_value = row["pred"].iloc[0]  # get the value from the Series
        assert pred_value == record["pred"]

    # Cleanup
    test_file.unlink()


def test_table_writer_write_multi_process_rank0(tmp_path, monkeypatch):
    """
    Test TableWriter in a multi-process environment from rank 0.

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

    # Expected records after gathering from all processes
    rank0_records = [{"identifier": 1, "pred": [1, 2, 3]}]  # records from rank 0
    rank1_records = [{"identifier": 2, "pred": [4, 5, 6]}]  # records from rank 1
    expected_records = rank0_records + rank1_records

    # Mock distributed functions for multi-process simulation
    def mock_gather(obj, gather_list, dst=0):
        if gather_list is not None:
            # Fill gather_list with records from each rank
            gather_list[0] = rank0_records
            gather_list[1] = rank1_records

    def mock_get_rank():
        return 0

    # Create a mock strategy with is_global_zero property
    mock_strategy = mock.MagicMock()
    mock_strategy.world_size = 2
    type(mock_strategy).is_global_zero = mock.PropertyMock(return_value=True)

    # Create a mock trainer with the strategy
    trainer = mock.MagicMock()
    type(trainer).strategy = mock.PropertyMock(return_value=mock_strategy)
    type(trainer).world_size = mock.PropertyMock(return_value=2)

    monkeypatch.setattr(torch.distributed, "gather_object", mock_gather)
    monkeypatch.setattr(torch.distributed, "get_rank", mock_get_rank)

    writer.on_predict_epoch_end(trainer, mock.Mock())

    # Verify file creation
    assert test_file.exists()

    # Verify file content
    df = pd.read_csv(test_file)
    df["identifier"] = df["identifier"].astype(str)
    df["pred"] = df["pred"].apply(eval)

    # Check that all expected records are in the CSV
    for record in expected_records:
        row = df[df["identifier"] == str(record["identifier"])]
        assert not row.empty
        pred_value = row["pred"].iloc[0]
        assert pred_value == record["pred"]

    # Verify total number of records
    assert len(df) == len(expected_records)


def test_table_writer_write_multi_process_rank1(tmp_path, monkeypatch):
    """
    Test TableWriter in a multi-process environment from rank 1.

    Tests:
    - Writing records from non-zero rank
    - Proper gathering of records to rank 0
    - No file creation on non-zero rank

    Args:
        tmp_path (Path): Pytest fixture providing temporary directory path
        monkeypatch (MonkeyPatch): Pytest fixture for mocking

    Mocked Behaviors:
    - Simulates 2-process distributed environment from rank 1
    - Mocks torch.distributed functions for testing
    - Simulates gathering of records to rank 0

    Verifies:
    - Records are properly gathered to rank 0
    - No file is created on rank 1
    """
    test_file = tmp_path / "test.csv"
    writer = TableWriter(path=test_file, writer="tensor")

    # Create some records for rank 1
    writer.write(tensor=torch.tensor([4, 5, 6]), identifier=2)

    # Mock distributed functions for multi-process simulation
    def mock_gather(obj, gather_list, dst=0):
        # Just verify obj is our records
        assert len(obj) == 1
        assert obj[0]["identifier"] == 2
        assert obj[0]["pred"] == [4, 5, 6]
        # In a real distributed environment, gather_object would handle sending
        # our records to rank 0, but we don't need to simulate that in the test
        # since we're just verifying rank 1's behavior

    def mock_get_rank():
        return 1

    # Create a mock trainer for rank 1
    trainer = mock.MagicMock()
    trainer.world_size = 2
    trainer.is_global_zero = False

    # Mock torch.distributed.get_rank to return 1 (non-zero rank)
    monkeypatch.setattr(torch.distributed, "gather_object", mock_gather)
    monkeypatch.setattr(torch.distributed, "get_rank", mock_get_rank)

    # Run the test
    writer.on_predict_epoch_end(trainer, mock.Mock())

    # Verify no file was created on rank 1
    assert not test_file.exists()
    # Verify records were cleared
    assert len(writer.csv_records) == 0


def test_table_writer_unsortable_identifiers(tmp_path):
    """
    Test TableWriter with identifiers that cannot be sorted.

    Tests:
    - Writing records with unsortable identifiers
    - Proper handling of TypeError during sorting
    - Successful file creation despite sorting failure

    Args:
        tmp_path (Path): Pytest fixture providing temporary directory path
    """
    test_file = tmp_path / "test.csv"
    writer = TableWriter(path=test_file, writer="tensor")

    # Create records with unsortable identifiers (mix of types)
    records = [
        {"identifier": 1, "pred": [1, 2]},
        {"identifier": "a", "pred": [3, 4]},
        {"identifier": [1, 2], "pred": [5, 6]},  # List is not comparable with str/int
    ]

    for record in records:
        writer.write(tensor=torch.tensor(record["pred"]), identifier=record["identifier"])

    trainer = Trainer(max_epochs=1)
    writer.on_predict_epoch_end(trainer, mock.Mock())

    # Verify file creation and content
    assert test_file.exists()
    df = pd.read_csv(test_file)
    df["pred"] = df["pred"].apply(eval)

    # Check that all records are in the CSV (order doesn't matter)
    assert len(df) == len(records)
    for record in records:
        # Convert identifier to string since pandas reads it as string
        identifier = str(record["identifier"])
        row = df[df["identifier"] == identifier]
        assert not row.empty
        pred_value = row["pred"].iloc[0]
        assert pred_value == record["pred"]
