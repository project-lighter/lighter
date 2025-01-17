import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from lighter.callbacks.writer.base import LighterBaseWriter


@pytest.fixture
def target_path():
    """
    Fixture that provides a test path for the writer.

    Returns:
        Path: A Path object pointing to "test" directory
    """
    return Path("test")


class MockWriter(LighterBaseWriter):
    """
    Mock implementation of LighterBaseWriter for testing purposes.

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

    def write(self, tensor, id):
        """
        Mock implementation of the write method.

        Args:
            tensor: The tensor to write
            id: Identifier for the tensor
        """
        pass


def test_writer_initialization(target_path):
    """
    Test the initialization of writers.

    Tests that:
    - MockWriter initializes correctly with valid writer
    - Base class raises TypeError when instantiated directly

    Args:
        target_path (Path): Fixture providing test directory path

    Raises:
        TypeError: When attempting to instantiate abstract base class
    """
    # Test initialization with a valid writer
    writer = MockWriter(path=target_path, writer="tensor")
    assert callable(writer.writer)
    with pytest.raises(TypeError):
        LighterBaseWriter(path=target_path, writer="tensor")


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

    outputs = {"pred": [torch.tensor([1, 2, 3])], "id": None}
    batch = MagicMock()
    batch_idx = 0

    writer.on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    assert outputs["id"] == [0]
    assert trainer.predict_loop._predictions == [[]]
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
    assert writer._pred_counter == 0


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
