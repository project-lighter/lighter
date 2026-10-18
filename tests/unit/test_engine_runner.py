"""Unit tests for building and running stages in lighter/engine/runner.py"""

import os
import subprocess
import sys
import warnings
from unittest.mock import MagicMock

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer
from sparkwheel import Config
from torch.utils.data import DataLoader, TensorDataset

from lighter import LighterDataModule, Runner, build
from lighter.engine.runner import _save_config, load_config

TRAINER = {
    "_target_": "pytorch_lightning.Trainer",
    "accelerator": "cpu",
    "max_epochs": 1,
    "logger": False,
    "enable_checkpointing": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "num_sanity_val_steps": 0,
}


class ResultTask(LightningModule):
    """A tiny module whose stage results are easy to check."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        return self.weight * batch[0].float().mean()

    def validation_step(self, batch, batch_idx):
        self.log("score", batch[0].float().mean(), batch_size=len(batch[0]))

    test_step = validation_step

    def predict_step(self, batch, batch_idx):
        return batch[0] * 2

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)

    def train_dataloader(self):
        return result_loader()

    val_dataloader = test_dataloader = predict_dataloader = train_dataloader


def result_loader():
    return DataLoader(TensorDataset(torch.arange(5)), batch_size=2)


def result_data():
    return LighterDataModule(**{f"{stage}_dataloader": result_loader() for stage in ("train", "val", "test", "predict")})


def config(**extra):
    return {"model": {"_target_": f"{__name__}.ResultTask"}, "trainer": dict(TRAINER), **extra}


def test_load_config_applies_inputs_in_order():
    loaded = load_config(config(), "trainer::max_epochs=5", {"trainer": {"devices": 1}})
    assert loaded.get("trainer::max_epochs") == 5
    assert loaded.get("trainer::devices") == 1


def test_build_returns_objects_without_running():
    model, trainer, datamodule = build(config())
    assert isinstance(model, ResultTask)
    assert isinstance(trainer, Trainer)
    assert datamodule is None
    assert trainer.global_step == 0 and model.weight.item() == 1.0


def test_build_resolves_the_data_key():
    _, _, datamodule = build(config(data={"_target_": f"{__name__}.result_data"}))
    assert isinstance(datamodule, LighterDataModule)


@pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
@pytest.mark.parametrize("external_data", [False, True])
def test_run_returns_the_native_stage_result(tmp_path, stage, external_data):
    options = {k: v for k, v in TRAINER.items() if k != "_target_"}
    native_args = {"datamodule": result_data()} if external_data else {}
    extra = {"data": {"_target_": f"{__name__}.result_data"}} if external_data else {}
    expected = getattr(Trainer(**options), stage)(ResultTask(), **native_args)
    actual = Runner().run(stage, [config(**extra)])
    if stage == "predict":
        assert torch.equal(torch.cat(actual), torch.arange(5) * 2)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    elif stage == "fit":
        assert actual is expected is None
    else:
        assert actual == expected == [{"score": 2.0}]


def test_run_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Unknown stage 'train'"):
        Runner().run("train", [config()])


def test_model_must_be_a_lightning_module():
    with pytest.raises(TypeError, match="`model` must build a LightningModule, got Linear"):
        build(config(model={"_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1}))


def test_trainer_without_target_explains_the_fix():
    with pytest.raises(TypeError, match="Add a `_target_` to `trainer:`"):
        build({"model": {"_target_": f"{__name__}.ResultTask"}, "trainer": {"max_epochs": 1}})


def test_lightning_package_modules_are_accepted():
    import lightning.pytorch as L

    class LightningTask(L.LightningModule):
        pass

    sys.modules[__name__].LightningTask = LightningTask
    model, trainer, _ = build(
        {"model": {"_target_": f"{__name__}.LightningTask"}, "trainer": dict(TRAINER, _target_="lightning.pytorch.Trainer")}
    )
    assert isinstance(model, L.LightningModule) and isinstance(trainer, L.Trainer)
    with pytest.raises(TypeError, match="mixes `pytorch_lightning` and `lightning.pytorch`"):
        build({"model": {"_target_": f"{__name__}.LightningTask"}, "trainer": dict(TRAINER)})


def test_seed_defaults_to_zero_without_a_warning(monkeypatch):
    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build(config())
    assert os.environ["PL_GLOBAL_SEED"] == "0"
    assert os.environ["PL_SEED_WORKERS"] == "1"


def test_seed_comes_from_the_config_then_the_environment(monkeypatch):
    build(config(seed=42))
    first = torch.rand(1)
    build(config(seed=42))
    assert torch.equal(first, torch.rand(1))
    monkeypatch.setenv("PL_GLOBAL_SEED", "7")
    build(config())
    assert os.environ["PL_GLOBAL_SEED"] == "7"


def test_seed_must_be_an_integer():
    with pytest.raises(TypeError, match="`seed` must be an integer"):
        build(config(seed="42"))


def test_python_dash_m_lighter_runs_the_cli():
    result = subprocess.run([sys.executable, "-m", "lighter", "--help"], capture_output=True, text=True, check=True)
    assert "usage: lighter" in result.stdout


# Project discovery


def test_auto_discover_project_with_marker(tmp_path, monkeypatch):
    """The folder holding __lighter__.py is imported as 'project'."""
    from lighter.engine.runner import _import_project

    (tmp_path / "__lighter__.py").touch()
    (tmp_path / "__init__.py").touch()
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("project", None)
    try:
        assert _import_project() is True
        assert sys.modules["project"].__path__ == [str(tmp_path)]
    finally:
        sys.modules.pop("project", None)


def test_auto_discover_project_without_marker(tmp_path, monkeypatch):
    """Nothing is imported when the marker is absent."""
    from lighter.engine.runner import _import_project

    monkeypatch.chdir(tmp_path)
    assert _import_project() is False


# Configuration saving


def test_save_config_writes_config_yaml(tmp_path):
    model, trainer = MagicMock(spec=LightningModule), MagicMock(spec=Trainer)
    trainer.log_dir = str(tmp_path / "version_0")
    _save_config(Config().update({"seed": 3}), trainer, model)
    assert (tmp_path / "version_0" / "config.yaml").read_text() == "seed: 3\n"
    trainer.logger.log_hyperparams.assert_called_once_with({"seed": 3})
    model.save_hyperparameters.assert_called_once_with({"config": {"seed": 3}})


def test_save_config_without_logger():
    model, trainer = MagicMock(spec=LightningModule), MagicMock(spec=Trainer)
    trainer.logger = None
    _save_config(Config().update({"seed": 3}), trainer, model)
    model.save_hyperparameters.assert_called_once_with({"config": {"seed": 3}})
