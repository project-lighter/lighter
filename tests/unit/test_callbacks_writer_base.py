import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from lighter.callbacks.writer.base import LighterBaseWriter


@pytest.fixture
def target_path():
    return Path("test")


class MockWriter(LighterBaseWriter):
    @property
    def writers(self):
        return {"tensor": lambda x: None}

    def write(self, tensor, id):
        pass


def test_writer_initialization(target_path):
    # Test initialization with a valid writer
    writer = MockWriter(path=target_path, writer="tensor")
    assert callable(writer.writer)
    with pytest.raises(TypeError):
        LighterBaseWriter(path=target_path, writer="tensor")


def test_on_predict_batch_end(target_path):
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
