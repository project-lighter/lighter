"""Unit tests for distributed (DDP) functionality of FileWriter and CsvWriter callbacks.

These tests emulate DDP on a single device by mocking the distributed environment.
"""

import csv
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from lighter.callbacks.csv_writer import CsvWriter
from lighter.callbacks.file_writer import FileWriter
from lighter.model import LighterModule
from lighter.utils.types.enums import Stage


def make_mock_trainer(global_rank=0, world_size=2, is_global_zero=True):
    """Create a mock trainer configured for DDP with proper strategy mocking."""
    trainer = MagicMock()
    trainer.global_rank = global_rank
    trainer.world_size = world_size
    trainer.is_global_zero = is_global_zero

    # Mock strategy with broadcast and barrier
    strategy = MagicMock()
    strategy.broadcast = lambda x, src=0: x  # Return path as-is
    strategy.barrier = MagicMock()
    strategy.root_device = torch.device("cpu")
    trainer.strategy = strategy

    # Mock the predict_loop to avoid the private API issue
    trainer.predict_loop = MagicMock()
    trainer.predict_loop.num_dataloaders = 1
    trainer.predict_loop._predictions = [[]]

    return trainer


@pytest.fixture
def mock_trainer_ddp():
    """Create a mock trainer configured for DDP rank 0."""
    return make_mock_trainer(global_rank=0, world_size=2, is_global_zero=True)


@pytest.fixture
def mock_trainer_ddp_rank1():
    """Create a mock trainer for rank 1 in DDP."""
    return make_mock_trainer(global_rank=1, world_size=2, is_global_zero=False)


@pytest.fixture
def mock_system():
    """Create a mock LighterModule for testing."""
    return MagicMock(spec=LighterModule)


class TestCsvWriterDDP:
    """Test suite for CsvWriter in distributed settings."""

    def test_csv_writer_creates_rank_specific_temp_files(self, tmp_path, mock_trainer_ddp, mock_system):
        """Test that CsvWriter creates temporary files with rank suffix."""
        csv_path = tmp_path / "predictions.csv"
        writer = CsvWriter(path=csv_path, keys=["pred", "target"])

        writer.setup(mock_trainer_ddp, mock_system, Stage.PREDICT)

        # Check that temporary file has rank suffix
        expected_temp = tmp_path / "predictions.tmp_rank0.csv"
        assert writer._temp_path == expected_temp
        assert expected_temp.exists()
        assert writer._csv_file is not None

    def test_csv_writer_multi_rank_temp_files(self, tmp_path, mock_system):
        """Test that different ranks create different temporary files."""
        csv_path = tmp_path / "predictions.csv"

        # Rank 0
        trainer_rank0 = make_mock_trainer(global_rank=0, world_size=2, is_global_zero=True)
        writer_rank0 = CsvWriter(path=csv_path, keys=["pred"])
        writer_rank0.setup(trainer_rank0, mock_system, Stage.PREDICT)

        # Rank 1
        trainer_rank1 = make_mock_trainer(global_rank=1, world_size=2, is_global_zero=False)
        writer_rank1 = CsvWriter(path=csv_path, keys=["pred"])
        writer_rank1.setup(trainer_rank1, mock_system, Stage.PREDICT)

        # Verify different temp files
        assert writer_rank0._temp_path == tmp_path / "predictions.tmp_rank0.csv"
        assert writer_rank1._temp_path == tmp_path / "predictions.tmp_rank1.csv"
        assert writer_rank0._temp_path != writer_rank1._temp_path

    def test_csv_writer_merge_without_dist(self, tmp_path, mock_trainer_ddp, mock_system):
        """Test CSV merging when distributed is not initialized (single process)."""
        csv_path = tmp_path / "predictions.csv"
        writer = CsvWriter(path=csv_path, keys=["pred", "target"])

        writer.setup(mock_trainer_ddp, mock_system, Stage.PREDICT)

        # Save temp path before it's cleared
        temp_path = writer._temp_path

        # Write some predictions
        outputs = {"pred": torch.tensor([1, 2, 3]), "target": torch.tensor([4, 5, 6])}
        writer.write(outputs, None, 0, 0)

        # Mock dist.is_initialized() to return False (non-distributed)
        with patch("torch.distributed.is_initialized", return_value=False):
            writer.on_predict_epoch_end(mock_trainer_ddp, mock_system)

        # Verify final CSV exists
        assert csv_path.exists()

        # Verify content
        df = pd.read_csv(csv_path)
        assert len(df) == 3
        assert list(df.columns) == ["pred", "target"]
        assert df["pred"].tolist() == [1, 2, 3]
        assert df["target"].tolist() == [4, 5, 6]

        # Verify temp file cleaned up
        assert not temp_path.exists()

        # Verify writer state reset
        assert writer._temp_path is None

    def test_csv_writer_merge_with_dist(self, tmp_path, mock_trainer_ddp, mock_system):
        """Test CSV merging when distributed is initialized (simulated DDP)."""
        csv_path = tmp_path / "predictions.csv"
        writer = CsvWriter(path=csv_path, keys=["pred", "target"])

        writer.setup(mock_trainer_ddp, mock_system, Stage.PREDICT)

        # Save temp path before it's cleared
        temp_path_rank0 = writer._temp_path

        # Write predictions for rank 0
        outputs_rank0 = {"pred": torch.tensor([1, 2]), "target": torch.tensor([4, 5])}
        writer.write(outputs_rank0, None, 0, 0)

        # Close the file to simulate end of predictions
        writer._csv_file.close()

        # Create a second temporary file to simulate rank 1
        temp_path_rank1 = tmp_path / "predictions.tmp_rank1.csv"
        with open(temp_path_rank1, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["pred", "target"])
            csv_writer.writerow([3, 6])

        # Mock distributed gathering
        temp_paths = [temp_path_rank0, temp_path_rank1]

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.all_gather_object") as mock_gather,
        ):
            # Simulate all_gather_object by filling the list
            def side_effect(tensor_list, tensor):
                tensor_list[:] = temp_paths

            mock_gather.side_effect = side_effect

            writer.on_predict_epoch_end(mock_trainer_ddp, mock_system)

        # Verify final CSV exists and contains data from both ranks
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 3  # 2 from rank 0 + 1 from rank 1
        assert list(df.columns) == ["pred", "target"]
        assert sorted(df["pred"].tolist()) == [1, 2, 3]
        assert sorted(df["target"].tolist()) == [4, 5, 6]

        # Verify temp files cleaned up (rank 0 is global_zero, so it cleans)
        assert not temp_path_rank0.exists()
        assert not temp_path_rank1.exists()

        # Verify writer state reset
        assert writer._temp_path is None

    def test_csv_writer_rank1_does_not_cleanup(self, tmp_path, mock_trainer_ddp_rank1, mock_system):
        """Test that non-zero ranks don't perform cleanup."""
        csv_path = tmp_path / "predictions.csv"
        writer = CsvWriter(path=csv_path, keys=["pred"])

        writer.setup(mock_trainer_ddp_rank1, mock_system, Stage.PREDICT)

        # Save temp path before it's cleared
        temp_path_rank1 = writer._temp_path

        # Write predictions for rank 1
        outputs = {"pred": torch.tensor([7, 8])}
        writer.write(outputs, None, 0, 0)

        # Close the file
        writer._csv_file.close()

        # Mock distributed gathering
        temp_path_rank0 = tmp_path / "predictions.tmp_rank0.csv"
        temp_paths = [temp_path_rank0, temp_path_rank1]

        # Create a fake rank 0 file
        temp_path_rank0.write_text("pred\n1\n2\n")

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.all_gather_object") as mock_gather,
        ):

            def side_effect(tensor_list, tensor):
                tensor_list[:] = temp_paths

            mock_gather.side_effect = side_effect

            writer.on_predict_epoch_end(mock_trainer_ddp_rank1, mock_system)

        # Rank 1 should NOT create the final CSV (only rank 0 does)
        assert not csv_path.exists()

        # Temp files should still exist (rank 1 doesn't clean up)
        assert temp_path_rank1.exists()
        assert temp_path_rank0.exists()

        # Verify writer state reset
        assert writer._temp_path is None

        # Cleanup for test
        temp_path_rank1.unlink()
        temp_path_rank0.unlink()


class TestFileWriterDDP:
    """Test suite for FileWriter in distributed settings."""

    def test_file_writer_creates_rank_directories(self, tmp_path, mock_system):
        """Test that FileWriter can work with rank-specific directories."""
        # Create a writer that could use rank in directory structure
        writer = FileWriter(directory=tmp_path / "rank_0", value_key="pred", writer_fn="tensor")

        trainer_rank0 = make_mock_trainer(global_rank=0, world_size=2, is_global_zero=True)
        writer.setup(trainer_rank0, mock_system, Stage.PREDICT)

        # Verify directory created
        assert writer.path.exists()
        assert writer.path.is_dir()

    def test_file_writer_barrier_synchronization(self, tmp_path, mock_system):
        """Test that FileWriter calls barrier for synchronization."""
        writer = FileWriter(directory=tmp_path, value_key="pred", writer_fn="tensor")

        # Create a trainer with a mock strategy that tracks barrier calls
        trainer = make_mock_trainer(global_rank=0, world_size=2, is_global_zero=True)

        writer.setup(trainer, mock_system, Stage.PREDICT)

        # Verify barrier was called on the strategy
        trainer.strategy.barrier.assert_called_once()

    def test_file_writer_no_barrier_without_dist(self, tmp_path, mock_trainer_ddp, mock_system):
        """Test that FileWriter doesn't call barrier when distributed is not initialized."""
        writer = FileWriter(directory=tmp_path, value_key="pred", writer_fn="tensor")

        with patch("torch.distributed.is_initialized", return_value=False), patch("torch.distributed.barrier") as mock_barrier:
            writer.setup(mock_trainer_ddp, mock_system, Stage.PREDICT)

            # Barrier should not be called
            mock_barrier.assert_not_called()


class TestDistributedEdgeCases:
    """Test edge cases in distributed settings."""

    def test_csv_writer_handles_empty_rank(self, tmp_path, mock_trainer_ddp, mock_system):
        """Test that CSV writer handles ranks with no predictions."""
        csv_path = tmp_path / "predictions.csv"
        writer = CsvWriter(path=csv_path, keys=["pred"])

        writer.setup(mock_trainer_ddp, mock_system, Stage.PREDICT)

        # Don't write anything (simulating empty rank)
        writer._csv_file.close()

        # Create another rank with data
        temp_path_rank1 = tmp_path / "predictions.tmp_rank1.csv"
        with open(temp_path_rank1, "w", newline="") as f:
            import csv

            csv_writer = csv.writer(f)
            csv_writer.writerow(["pred"])
            csv_writer.writerow([1])
            csv_writer.writerow([2])

        temp_paths = [writer._temp_path, temp_path_rank1]

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.all_gather_object") as mock_gather,
        ):

            def side_effect(tensor_list, tensor):
                tensor_list[:] = temp_paths

            mock_gather.side_effect = side_effect

            writer.on_predict_epoch_end(mock_trainer_ddp, mock_system)

        # Final CSV should have only rank 1's data
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert df["pred"].tolist() == [1, 2]

    def test_csv_writer_handles_none_paths(self, tmp_path, mock_trainer_ddp, mock_system):
        """Test that CSV writer handles None paths in gathered list."""
        csv_path = tmp_path / "predictions.csv"
        writer = CsvWriter(path=csv_path, keys=["pred"])

        writer.setup(mock_trainer_ddp, mock_system, Stage.PREDICT)
        outputs = {"pred": torch.tensor([1, 2])}
        writer.write(outputs, None, 0, 0)
        writer._csv_file.close()

        # Simulate gathering with some None values (failed/missing ranks)
        temp_paths = [writer._temp_path, None]

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.all_gather_object") as mock_gather,
        ):

            def side_effect(tensor_list, tensor):
                tensor_list[:] = temp_paths

            mock_gather.side_effect = side_effect

            writer.on_predict_epoch_end(mock_trainer_ddp, mock_system)

        # Should still work with None paths filtered out
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert df["pred"].tolist() == [1, 2]
