import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from loguru import logger

from lighter.callbacks.writer.base import BaseWriter


@pytest.fixture
def target_path():
    """
    Fixture that provides a test path for the writer.

    Returns:
        Path: A Path object pointing to "test" directory
    """
    return Path("test")


class MockWriter(BaseWriter):
    """
    Mock implementation of BaseWriter for testing purposes.

    This class provides a minimal implementation of the abstract base class
    with a simple tensor writer function.
    """

    @property
    def writers(self):
        """
        Define available writers for the mock class.

        Returns:
            dict: Dictionary containing writer name and corresponding function
        """
        return {"tensor": lambda x: None}

    def write(self, tensor, identifier):
        """
        Mock implementation of the write method.

        Args:
            tensor: The tensor to write
            identifier: Identifier for the tensor
        """
        pass


def test_writer_initialization(target_path):
    """
    Test the initialization of writers.

    Tests that:
    - MockWriter initializes correctly with valid writer
    - Base class raises TypeError when instantiated directly
    - Raises ValueError when initialized with invalid writer
    - Raises TypeError when initialized with invalid path
    """
    # Test initialization with a valid writer
    writer = MockWriter(path=target_path, writer="tensor")
    assert callable(writer.writer)
    with pytest.raises(TypeError):
        BaseWriter(path=target_path, writer="tensor")

    # Test initialization with invalid writer
    with pytest.raises(ValueError, match="Writer for format invalid_writer does not exist"):
        MockWriter(path=target_path, writer="invalid_writer")

    # Test initialization with invalid path
    with pytest.raises(TypeError):
        MockWriter(path=123, writer="tensor")


def test_on_predict_batch_end(target_path):
    """
    Test the on_predict_batch_end callback functionality.

    Verifies that:
    - Prediction IDs are properly assigned
    - Prediction counter increments correctly
    - Trainer's prediction list is maintained

    Args:
        target_path (Path): Fixture providing test directory path
    """
    logging.basicConfig(level=logging.INFO)
    trainer = MagicMock()
    trainer.world_size = 1
    trainer.predict_loop.num_dataloaders = 1
    trainer.predict_loop._predictions = [[]]

    pl_module = MagicMock()

    writer = MockWriter(path=target_path, writer="tensor")
    writer._pred_counter = 0

    outputs = {"pred": [torch.tensor([1, 2, 3])], "identifier": None}
    batch = MagicMock()
    batch_idx = 0

    writer.on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    assert outputs["identifier"] == [0]
    assert trainer.predict_loop._predictions == [[]]
    assert writer._pred_counter == 1

    # Test with provided identifiers
    outputs = {"pred": [torch.tensor([1, 2, 3])], "identifier": [1, 2, 3]}
    writer.on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx)
    assert writer._pred_counter == 1


def test_writer_setup_predict(target_path, caplog):
    """
    Test writer setup for prediction stage.

    Verifies that:
    - Writer initializes correctly for prediction
    - Prediction counter is properly reset
    - Global synchronization works as expected

    Args:
        target_path (Path): Fixture providing test directory path
        caplog: Pytest fixture for capturing log output
    """
    trainer = MagicMock()
    trainer.world_size = 1
    trainer.is_global_zero = True
    trainer.strategy.broadcast.return_value = target_path
    trainer.strategy.barrier = MagicMock()

    pl_module = MagicMock()

    writer = MockWriter(path=target_path, writer="tensor")
    writer.setup(trainer, pl_module, stage="predict")
    assert writer._pred_counter == 0

    # Test setup for non-predict stage
    writer = MockWriter(path=target_path, writer="tensor")
    writer.setup(trainer, pl_module, stage="train")
    assert writer._pred_counter is None


def test_writer_setup_non_predict(target_path):
    """
    Test writer setup for non-prediction stages.

    Verifies that:
    - Writer initializes correctly for non-prediction stages (e.g., train)
    - Prediction counter remains None
    - Path is properly set

    Args:
        target_path (Path): Fixture providing test directory path
    """
    trainer = MagicMock()
    trainer.world_size = 1
    trainer.is_global_zero = True
    trainer.strategy.broadcast.return_value = target_path
    trainer.strategy.barrier = MagicMock()

    pl_module = MagicMock()

    writer = MockWriter(path=target_path, writer="tensor")
    writer.setup(trainer, pl_module, stage="train")
    assert writer._pred_counter is None
    assert writer.path == target_path

    # Test with invalid path
    with pytest.raises(ValueError, match="Writer for format invalid_writer does not exist"):
        MockWriter(path=target_path, writer="invalid_writer")

    # Test with invalid path type
    with pytest.raises(TypeError):
        MockWriter(path=123, writer="tensor")


def test_writer_setup_existing_path(target_path):
    """
    Test writer setup when the target path already exists.

    Args:
        target_path (Path): Fixture providing test directory path
    """
    trainer = MagicMock()
    trainer.world_size = 1
    trainer.is_global_zero = True
    trainer.strategy.broadcast.return_value = target_path
    trainer.strategy.barrier = MagicMock()

    pl_module = MagicMock()
    writer = MockWriter(path=target_path, writer="tensor")

    # Mock path.exists() to return True and capture loguru warning
    warning_messages = []
    logger.add(lambda msg: warning_messages.append(msg.record["message"]), level="WARNING")

    with patch.object(Path, "exists", return_value=True):
        writer.setup(trainer, pl_module, stage="predict")
        assert any("already exists" in msg for msg in warning_messages)
        assert any("existing predictions will be overwritten" in msg for msg in warning_messages)


def test_writer_setup_directory_not_shared(target_path):
    """
    Test writer setup when directory is not shared between nodes.

    Args:
        target_path (Path): Fixture providing test directory path
    """
    trainer = MagicMock()
    trainer.world_size = 2
    trainer.is_global_zero = False
    trainer.global_rank = 1
    trainer.strategy.broadcast.return_value = target_path
    trainer.strategy.barrier = MagicMock()

    pl_module = MagicMock()
    writer = MockWriter(path=target_path, writer="tensor")

    # Mock path.exists() to return False to simulate directory not being shared
    # Also mock torch.distributed.get_rank() to avoid initialization error
    with patch.object(Path, "exists", return_value=False), patch("torch.distributed.get_rank", return_value=1):
        with pytest.raises(RuntimeError, match="Rank 1 does not share storage with rank 0"):
            writer.setup(trainer, pl_module, stage="predict")
