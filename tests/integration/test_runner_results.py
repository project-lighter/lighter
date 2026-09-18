"""The public Runner preserves native stage results for real Lightning execution."""

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from lighter import LighterDataModule, Runner
from lighter.utils.types.enums import Stage


class ResultTask(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        return self.weight * batch[0].float().mean()

    def validation_step(self, batch, batch_idx):
        self.log("score", batch[0].float().mean(), batch_size=len(batch[0]))

    def test_step(self, batch, batch_idx):
        self.log("score", batch[0].float().mean(), batch_size=len(batch[0]))

    def predict_step(self, batch, batch_idx):
        return batch[0] * 2

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)

    def train_dataloader(self):
        return result_loader()

    val_dataloader = train_dataloader
    test_dataloader = train_dataloader
    predict_dataloader = train_dataloader


def result_loader():
    return DataLoader(TensorDataset(torch.arange(5)), batch_size=2)


def result_data():
    return LighterDataModule(**{f"{stage}_dataloader": result_loader() for stage in ("train", "val", "test", "predict")})


@pytest.mark.parametrize("stage", [Stage.FIT, Stage.VALIDATE, Stage.TEST, Stage.PREDICT])
@pytest.mark.parametrize("external_data", [False, True])
def test_runner_returns_native_stage_result(tmp_path, stage, external_data):
    options = dict(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=str(tmp_path),
    )
    config = {
        "model": {"_target_": f"{__name__}.ResultTask"},
        "trainer": {"_target_": "pytorch_lightning.Trainer", **options},
    }
    native_args = {}
    if external_data:
        config["data"] = {"_target_": f"{__name__}.result_data"}
        native_args["datamodule"] = result_data()
    kwargs = {"return_predictions": True} if stage == Stage.PREDICT else {}
    expected = getattr(Trainer(**options), str(stage))(ResultTask(), **native_args, **kwargs)
    actual = Runner().run(stage, [config], **kwargs)
    if stage == Stage.PREDICT:
        assert actual is not None
        assert len(actual) == len(expected) == 3
        assert torch.equal(torch.cat(actual), torch.arange(5) * 2)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    elif stage == Stage.FIT:
        assert actual is expected is None
    else:
        assert actual == expected == [{"score": 2.0}]
