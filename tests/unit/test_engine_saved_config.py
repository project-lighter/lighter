"""The config saved with each run (config.yaml, checkpoints) and `lighter.load_model`."""

import sys
from unittest.mock import MagicMock

import pytest
import torch
import yaml

from lighter import build, load_model
from lighter.engine.runner import cli
from lighter.engine.saved_config import SaveConfig

TASK = """import torch
from lighter import LighterModule


class Task(LighterModule):
    def training_step(self, batch, batch_idx):
        return self(batch[0]).pow(2).mean()
"""

CONFIG = """_imports_:
  torch: torch
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1
  default_root_dir: {root}
  enable_progress_bar: false
  enable_model_summary: false
model:
  _target_: project.tasks.Task
  network: {{_target_: torch.nn.Linear, in_features: 4, out_features: 1}}
  optimizer: {{_target_: torch.optim.SGD, _mode_: callable, lr: 0.1}}
data:
  _target_: lighter.LighterDataModule
  train_dataloader:
    _target_: torch.utils.data.DataLoader
    dataset: "$torch.utils.data.TensorDataset(torch.randn(8, 4))"
    batch_size: 4
"""


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Fit a project's model once from another folder; return the project folder and the fitted model."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "__lighter__.py").touch()
    (root / "tasks.py").write_text(TASK)
    (root / "config.yaml").write_text(CONFIG.format(root=root))
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
    drop = lambda: [sys.modules.pop(name) for name in list(sys.modules) if name.split(".")[0] == "project"]  # noqa: E731
    drop()
    model, trainer, datamodule = build(str(root / "config.yaml"))
    trainer.fit(model, datamodule=datamodule)
    drop()
    yield root, model
    drop()


def test_the_run_folder_holds_the_composed_config(run):
    root, model = run
    saved = yaml.safe_load((root / "lightning_logs" / "version_0" / "config.yaml").read_text())
    assert saved["seed"] == 0 and saved["_imports_"] == {"torch": "torch"}
    assert saved["model"]["_target_"] == "project.tasks.Task"
    checkpoint = torch.load(next(root.glob("lightning_logs/version_0/checkpoints/*.ckpt")), weights_only=True)
    assert checkpoint["lighter_config"] == saved
    assert "config" not in model.hparams


def test_load_model_rebuilds_the_model_with_its_weights(run):
    root, model = run
    ckpt = next(root.glob("lightning_logs/version_0/checkpoints/*.ckpt"))
    loaded = load_model(ckpt, ["model::criterion={_target_: torch.nn.MSELoss}"])
    assert type(loaded).__module__ == "project.tasks"
    assert torch.equal(loaded.network.weight, model.network.weight)
    assert isinstance(loaded.criterion, torch.nn.MSELoss)


def test_load_model_needs_a_lighter_checkpoint(tmp_path):
    torch.save({"state_dict": {}}, tmp_path / "plain.ckpt")
    with pytest.raises(ValueError, match="holds no Lighter config"):
        load_model(tmp_path / "plain.ckpt")


def test_ckpt_path_last_without_a_last_checkpoint_explains_the_version_folders(run, monkeypatch, capsys):
    root, _ = run
    monkeypatch.setattr(sys, "argv", ["lighter", "fit", str(root / "config.yaml"), "--ckpt_path", "last"])
    with pytest.raises(SystemExit) as exit:
        cli()
    assert exit.value.code == 2
    error = capsys.readouterr().err
    assert "`--ckpt_path last` found no last checkpoint" in error and "each run logs to a new version folder" in error
    assert f"--ckpt_path {next(root.glob('lightning_logs/version_0/checkpoints/*.ckpt'))}" in error


def test_save_config_writes_only_on_global_zero(tmp_path):
    trainer = MagicMock(is_global_zero=False, log_dir=str(tmp_path))
    SaveConfig({"seed": 1}).setup(trainer, None, "fit")
    assert not (tmp_path / "config.yaml").exists()
    trainer.logger.log_hyperparams.assert_not_called()
    trainer.is_global_zero = True
    SaveConfig({"seed": 1}).setup(trainer, None, "fit")
    assert (tmp_path / "config.yaml").read_text() == "seed: 1\n"
    trainer.logger.log_hyperparams.assert_called_once_with({"seed": 1})
