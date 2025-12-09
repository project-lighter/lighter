"""Integration test showing Lighter works with plain PyTorch Lightning modules."""

import sys
import tempfile
from pathlib import Path

# Add tests directory to path so we can import fixtures
sys.path.insert(0, str(Path(__file__).parent.parent))


from lighter.engine.runner import Runner
from lighter.utils.types.enums import Stage


def test_lighter_with_plain_lightning_module():
    """Test that Lighter can run a plain PyTorch Lightning module."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"

        # Config using a plain Lightning module
        # Dataloaders are defined in the module itself
        config_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1
  enable_checkpointing: false
  logger: false

model:
  _target_: fixtures.plain_lightning_modules.PlainLightningModule
  input_size: 10
  hidden_size: 20
  output_size: 2
  learning_rate: 0.001
"""
        config_path.write_text(config_content)

        # Run with Lighter
        runner = Runner()
        runner.run(Stage.FIT, [str(config_path)])

        # Runner no longer stores system, just verify it ran successfully without error


def test_lighter_with_lightning_module_and_external_dataloaders():
    """Test using a Lightning module that defines its own dataloaders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"

        # Minimal config - dataloaders defined in the module itself
        config_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1
  enable_checkpointing: false
  logger: false

model:
  _target_: fixtures.plain_lightning_modules.LightningModuleWithDataloaders
"""
        config_path.write_text(config_content)

        runner = Runner()
        runner.run(Stage.FIT, [str(config_path)])

        # Runner no longer stores system, just verify it ran successfully without error


def test_mixed_lighter_system_and_plain_lightning():
    """
    Test that you can switch between Lighter System and plain Lightning
    by just changing the config.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Use LighterModule
        config_path1 = Path(tmpdir) / "lighter_module.yaml"
        config_content1 = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1
  enable_checkpointing: false
  logger: false

model:
  _target_: fixtures.plain_lightning_modules.MyLighterModule
  network:
    _target_: torch.nn.Linear
    in_features: 10
    out_features: 2
  optimizer:
    _target_: torch.optim.Adam
    params: "$@model::network.parameters()"
    lr: 0.001

data:
  _target_: lighter.LighterDataModule
  train_dataloader:
    _target_: torch.utils.data.DataLoader
    dataset:
      _target_: fixtures.plain_lightning_modules.SimpleDataset
    batch_size: 8
  val_dataloader:
    _target_: torch.utils.data.DataLoader
    dataset:
      _target_: fixtures.plain_lightning_modules.SimpleDataset
    batch_size: 8
"""
        config_path1.write_text(config_content1)

        runner1 = Runner()
        runner1.run(Stage.FIT, [str(config_path1)])

        # Runner no longer stores system, just verify it ran successfully without error

        # Test 2: Use plain Lightning module (same codebase!)
        # Plain Lightning module defines dataloaders in the module itself
        config_path2 = Path(tmpdir) / "plain_lightning.yaml"
        config_content2 = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1
  enable_checkpointing: false
  logger: false

model:
  _target_: fixtures.plain_lightning_modules.PlainLightningModule
"""
        config_path2.write_text(config_content2)

        runner2 = Runner()
        runner2.run(Stage.FIT, [str(config_path2)])

        # Runner no longer stores system, just verify it ran successfully without error
