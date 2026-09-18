"""Batch writers preserve native return requests and complete output artifacts."""

import csv

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from lighter.callbacks import CsvWriter, FileWriter


class PredictionTask(LightningModule):
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        values = batch[0]
        return {"id": [f"{dataloader_idx}-{int(value):04d}" for value in values], "pred": values * 2}


@pytest.mark.parametrize("return_predictions", [True, False, None])
@pytest.mark.parametrize("loader_count", [1, 2])
@pytest.mark.parametrize("writer_kind", ["csv", "file"])
def test_writers_preserve_native_return_policy(tmp_path, return_predictions, loader_count, writer_kind):
    loaders = [DataLoader(TensorDataset(torch.arange(5)), batch_size=2) for _ in range(loader_count)]
    if writer_kind == "csv":
        writer = CsvWriter(tmp_path / "predictions.csv", keys=["id", "pred"])
    else:
        writer = FileWriter(tmp_path / "predictions", value_key="pred", name_key="id", writer_fn="tensor")
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[writer],
        default_root_dir=tmp_path,
    )
    result = trainer.predict(
        PredictionTask(),
        dataloaders=loaders[0] if loader_count == 1 else loaders,
        return_predictions=return_predictions,
    )
    expected = {f"{loader}-{value:04d}": value * 2 for loader in range(loader_count) for value in range(5)}
    if writer_kind == "csv":
        with (tmp_path / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(expected)
        assert {row["id"]: int(row["pred"]) for row in rows} == expected
    else:
        files = list((tmp_path / "predictions").glob("*.pt"))
        assert len(files) == len(expected)
        assert {path.stem: int(torch.load(path, weights_only=True)) for path in files} == expected
    if return_predictions is False:
        assert result is None
    else:
        batches = result if loader_count == 1 else [batch for loader in result for batch in loader]
        assert len(batches) == loader_count * 3
        assert {
            name: int(value) for batch in batches for name, value in zip(batch["id"], batch["pred"], strict=True)
        } == expected
