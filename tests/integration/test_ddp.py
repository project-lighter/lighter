"""`lighter fit` with two CPU processes: every rank builds the run, rank zero alone saves the config and logs."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

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
  accelerator: cpu
  devices: 2
  strategy: {strategy}
  max_epochs: 1
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
    dataset: "$torch.utils.data.TensorDataset(torch.randn(16, 4))"
    batch_size: 4
"""


@pytest.mark.slow
@pytest.mark.parametrize("strategy", ["ddp", "ddp_spawn"])
def test_fit_with_two_cpu_processes(tmp_path, strategy):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "__lighter__.py").touch()
    (root / "tasks.py").write_text(TASK)
    (root / "config.yaml").write_text(CONFIG.format(strategy=strategy))
    lighter = Path(sys.executable).parent / "lighter"
    result = subprocess.run([str(lighter), "fit", "config.yaml"], cwd=root, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stderr.count("| project: ") == 1  # the banner, from rank zero only
    assert [folder.name for folder in (root / "lightning_logs").iterdir()] == ["version_0"]
    assert "strategy: " + strategy in (root / "lightning_logs" / "version_0" / "config.yaml").read_text()
    checkpoint = torch.load(next(root.glob("lightning_logs/version_0/checkpoints/*.ckpt")), weights_only=True)
    assert checkpoint["lighter_config"]["trainer"]["strategy"] == strategy
