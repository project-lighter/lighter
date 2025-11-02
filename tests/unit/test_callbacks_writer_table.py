from pathlib import Path
from unittest import mock

import pandas as pd
import torch
from pytorch_lightning import Trainer

from lighter.callbacks.writer.table import TableWriter


def test_table_writer_initialization():
    """
    Test proper initialization of TableWriter.
    """
    writer = TableWriter(path="test.csv", writer="tensor")
    assert writer.path == Path("test.csv")


def test_table_writer_write(tmp_path):
    """
    Test TableWriter write functionality.
    """
    test_file = tmp_path / "test.csv"
    writer = TableWriter(path=test_file, writer="tensor")

    # Mock trainer
    trainer = mock.MagicMock(spec=Trainer)
    trainer.world_size = 1
    trainer.is_global_zero = True

    # Setup the writer
    writer.setup(trainer, mock.Mock(), "predict")

    # Write some data
    writer.write(tensor=torch.tensor([1, 2, 3]), identifier=1)
    writer.write(tensor=torch.tensor(-1), identifier="some_id")
    writer.write(tensor=torch.tensor([1.5, 2.5]), identifier=1331)

    # End of epoch
    writer.on_predict_epoch_end(trainer, mock.Mock())

    # Verify file creation and content
    assert test_file.exists()
    df = pd.read_csv(test_file)
    assert len(df) == 3
    assert df["identifier"].tolist() == [1, "some_id", 1331]
    assert df["pred"].tolist() == ["[1, 2, 3]", "-1", "[1.5, 2.5]"]


def test_table_writer_distributed(tmp_path):
    """
    Test TableWriter in a distributed setting.
    """
    # Create two writers to simulate two processes
    writer_0 = TableWriter(path=tmp_path / "test.csv", writer="tensor")
    writer_1 = TableWriter(path=tmp_path / "test.csv", writer="tensor")

    # Mock trainers for each process
    trainer_0 = mock.MagicMock(spec=Trainer)
    trainer_0.world_size = 2
    trainer_0.is_global_zero = True
    trainer_0.global_rank = 0
    trainer_0.strategy.broadcast.return_value = tmp_path / "test.csv"

    trainer_1 = mock.MagicMock(spec=Trainer)
    trainer_1.world_size = 2
    trainer_1.is_global_zero = False
    trainer_1.global_rank = 1
    trainer_1.strategy.broadcast.return_value = tmp_path / "test.csv"

    # Setup the writers
    writer_0.setup(trainer_0, mock.Mock(), "predict")
    writer_1.setup(trainer_1, mock.Mock(), "predict")

    # Write data from each process
    writer_0.write(tensor=torch.tensor([1, 2, 3]), identifier=0)
    writer_1.write(tensor=torch.tensor([4, 5, 6]), identifier=1)

    # End of epoch for both processes
    writer_0.on_predict_epoch_end(trainer_0, mock.Mock())
    writer_1.on_predict_epoch_end(trainer_1, mock.Mock())

    # Verify that only rank 0 creates the final file
    assert (tmp_path / "test.csv").exists()
    assert not (tmp_path / "_temp_1.csv").exists()

    # Verify the content of the final file
    df = pd.read_csv(tmp_path / "test.csv")
    assert len(df) == 2
    assert df["identifier"].tolist() == [0, 1]
    assert df["pred"].tolist() == ["[1, 2, 3]", "[4, 5, 6]"]
