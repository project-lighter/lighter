"""Unit tests for the CSVWriter and FileWriter callbacks."""

import os
from unittest.mock import MagicMock

import pytest
import torch

from lighter.callbacks.csv_writer import CsvWriter
from lighter.callbacks.file_writer import FileWriter
from lighter.system import System
from lighter.utils.types.enums import Data


@pytest.fixture
def dummy_system():
    """Provides a mock System instance."""
    system = MagicMock(spec=System)
    system.trainer = MagicMock()
    system.trainer.log_dir = "./test_logs"
    system.trainer.global_step = 0
    system.trainer.current_epoch = 0
    return system


@pytest.fixture
def mock_batch_output():
    """Provides a mock batch output dictionary."""
    return {
        Data.PRED: torch.tensor([0.1, 0.2, 0.3]),
        Data.TARGET: torch.tensor([0, 1, 0]),
        Data.LOSS: torch.tensor(0.5),
        Data.METRICS: {"accuracy": torch.tensor(0.8), "f1": torch.tensor(0.7)},
    }


@pytest.fixture
def mock_epoch_output():
    """Provides a mock epoch output list."""
    return [
        {
            Data.PRED: torch.tensor([0.1, 0.2, 0.3]),
            Data.TARGET: torch.tensor([0, 1, 0]),
            Data.LOSS: torch.tensor(0.5),
            Data.METRICS: {"accuracy": torch.tensor(0.8), "f1": torch.tensor(0.7)},
        },
        {
            Data.PRED: torch.tensor([0.4, 0.5, 0.6]),
            Data.TARGET: torch.tensor([1, 0, 1]),
            Data.LOSS: torch.tensor(0.3),
            Data.METRICS: {"accuracy": torch.tensor(0.9), "f1": torch.tensor(0.85)},
        },
    ]


class TestCSVWriter:
    def test_csv_writer_init(self, tmp_path):
        """Test CSVWriter initialization."""
        writer = CSVWriter(output_dir=str(tmp_path), filename="test.csv", write_interval="batch")
        assert writer.output_dir == str(tmp_path)
        assert writer.filename == "test.csv"
        assert writer.write_interval == "batch"
        assert writer.write_on_batch_end is True
        assert writer.write_on_epoch_end is False

    def test_csv_writer_write_on_batch_end(self, tmp_path, dummy_system, mock_batch_output):
        """Test CSVWriter writes on batch end."""
        output_file = tmp_path / "test.csv"
        writer = CSVWriter(output_dir=str(tmp_path), filename="test.csv", write_interval="batch")
        writer.on_train_batch_end(dummy_system.trainer, dummy_system, mock_batch_output, None, 0)

        assert output_file.exists()
        content = output_file.read_text()
        assert "pred_0,pred_1,pred_2,target_0,target_1,target_2,loss,accuracy,f1" in content
        assert "0.1,0.2,0.3,0,1,0,0.5,0.8,0.7" in content

    def test_csv_writer_write_on_epoch_end(self, tmp_path, dummy_system, mock_epoch_output):
        """Test CSVWriter writes on epoch end."""
        output_file = tmp_path / "test.csv"
        writer = CSVWriter(output_dir=str(tmp_path), filename="test.csv", write_interval="epoch")
        writer.on_train_epoch_end(dummy_system.trainer, dummy_system, mock_epoch_output)

        assert output_file.exists()
        content = output_file.read_text()
        assert "pred_0,pred_1,pred_2,target_0,target_1,target_2,loss,accuracy,f1" in content
        assert "0.1,0.2,0.3,0,1,0,0.5,0.8,0.7" in content
        assert "0.4,0.5,0.6,1,0,1,0.3,0.9,0.85" in content

    def test_csv_writer_append_mode(self, tmp_path, dummy_system, mock_batch_output):
        """Test CSVWriter appends to existing file."""
        output_file = tmp_path / "test.csv"
        writer = CSVWriter(output_dir=str(tmp_path), filename="test.csv", write_interval="batch")

        writer.on_train_batch_end(dummy_system.trainer, dummy_system, mock_batch_output, None, 0)
        writer.on_train_batch_end(dummy_system.trainer, dummy_system, mock_batch_output, None, 1)

        content = output_file.read_text()
        assert content.count("0.1,0.2,0.3,0,1,0,0.5,0.8,0.7") == 2

    def test_csv_writer_write_multi_process_rank0(self, tmp_path, dummy_system, monkeypatch):
        """Test CsvWriter in a multi-process environment from rank 0."""
        test_file = tmp_path / "test.csv"
        writer = CsvWriter(path=test_file, keys=["pred", "target"])

        # Mock distributed functions for multi-process simulation
        def mock_all_gather_object(output_list, input_object):
            output_list[0] = [{"pred": torch.tensor([0.1, 0.2]), "target": torch.tensor([0, 1])}]
            output_list[1] = [{"pred": torch.tensor([0.3, 0.4]), "target": torch.tensor([1, 0])}]

        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "all_gather_object", mock_all_gather_object)

        dummy_system.trainer.is_global_zero = True
        dummy_system.trainer.world_size = 2
        dummy_system.trainer.global_rank = 0

        # Simulate writing some data (this data will be gathered)
        writer.write({"pred": torch.tensor([0.1, 0.2]), "target": torch.tensor([0, 1])}, None, 0, 0)

        writer.on_predict_epoch_end(dummy_system.trainer, dummy_system)

        assert test_file.exists()
        content = test_file.read_text()
        assert "pred_0,pred_1,target_0,target_1" in content
        assert "0.1,0.2,0,1" in content
        assert "0.3,0.4,1,0" in content

    def test_csv_writer_write_multi_process_rank1(self, tmp_path, dummy_system, monkeypatch):
        """Test CsvWriter in a multi-process environment from rank 1."""
        test_file = tmp_path / "test.csv"
        writer = CsvWriter(path=test_file, keys=["pred", "target"])

        # Mock distributed functions for multi-process simulation
        def mock_all_gather_object(output_list, input_object):
            # In a real distributed environment, this would send input_object to rank 0
            pass

        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "all_gather_object", mock_all_gather_object)

        dummy_system.trainer.is_global_zero = False
        dummy_system.trainer.world_size = 2
        dummy_system.trainer.global_rank = 1

        # Simulate writing some data
        writer.write({"pred": torch.tensor([0.5, 0.6]), "target": torch.tensor([0, 1])}, None, 0, 0)

        writer.on_predict_epoch_end(dummy_system.trainer, dummy_system)

        # On non-global_zero rank, the file should not be created by this rank
        assert not test_file.exists()
