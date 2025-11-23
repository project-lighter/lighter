"""Integration tests for configuration validation with real YAML files."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lighter.engine.runner import Runner
from lighter.utils.types.enums import Stage


class TestConfigValidation:
    """Integration tests for config validation with various YAML setups."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for test configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_minimal_valid_config_loads(self, temp_config_dir):
        """Test that minimal valid config loads."""
        config_path = temp_config_dir / "minimal.yaml"
        config_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1

model:
  _target_: lighter.LighterModule
  network:
    _target_: torch.nn.Identity

data:
  _target_: lighter.LighterDataModule
  train_dataloader: {}
"""
        config_path.write_text(config_content)

        runner = Runner()
        # Patch internal methods to avoid needing real components
        with (
            patch.object(runner, "_resolve_model"),
            patch.object(runner, "_resolve_trainer"),
            patch.object(runner, "_resolve_datamodule"),
            patch.object(runner, "_execute"),
        ):
            # Should not raise error - just verify config loads
            runner.run(Stage.FIT, [str(config_path)])

    def test_config_with_all_components(self, temp_config_dir):
        """Test config with all components."""
        config_path = temp_config_dir / "full.yaml"
        config_content = """
project: ./path/to/project

vars:
  learning_rate: 0.001

trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10

model:
  _target_: lighter.LighterModule
  network:
    _target_: torch.nn.Identity
  criterion:
    _target_: torch.nn.MSELoss
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
  scheduler:
    _target_: torch.optim.lr_scheduler.StepLR
    step_size: 10
  train_metrics: []
  val_metrics: []
  test_metrics: []

data:
  _target_: lighter.LighterDataModule
  train_dataloader: {}
  val_dataloader: {}
  test_dataloader: {}
  predict_dataloader: {}

args:
  fit:
    ckpt_path: checkpoint.ckpt
  validate: {}
  test: {}
  predict: {}
"""
        config_path.write_text(config_content)

        runner = Runner()
        # Capture config to verify values
        captured_config = None

        def capture_system(config):
            nonlocal captured_config
            captured_config = config
            return MagicMock()

        with (
            patch.object(runner, "_resolve_model", side_effect=capture_system),
            patch.object(runner, "_resolve_trainer"),
            patch.object(runner, "_resolve_datamodule"),
            patch.object(runner, "_execute"),
        ):
            runner.run(Stage.FIT, [str(config_path)])
            assert captured_config.get("project") == "./path/to/project"
            assert captured_config.get("vars::learning_rate") == 0.001
            assert captured_config.get("model::optimizer::lr") == 0.001

    def test_multi_file_config_merge(self, temp_config_dir):
        """Test that multiple config files merge correctly."""
        base_path = temp_config_dir / "base.yaml"
        base_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 100
  devices: 1

model:
  _target_: lighter.LighterModule
  network:
    _target_: torch.nn.Identity

data:
  _target_: lighter.LighterDataModule
  train_dataloader: {}
"""
        base_path.write_text(base_content)

        override_path = temp_config_dir / "override.yaml"
        override_content = """
trainer:
  max_epochs: 1
  devices: 2

model:
  network:
    _target_: torch.nn.Identity
  criterion:
    _target_: torch.nn.MSELoss

data:
  train_dataloader: {}
  val_dataloader: {}
"""
        override_path.write_text(override_content)

        runner = Runner()
        # Capture config to verify values
        captured_config = None

        def capture_system(config):
            nonlocal captured_config
            captured_config = config
            return MagicMock()

        with (
            patch.object(runner, "_resolve_model", side_effect=capture_system),
            patch.object(runner, "_resolve_trainer"),
            patch.object(runner, "_resolve_datamodule"),
            patch.object(runner, "_execute"),
        ):
            runner.run(Stage.FIT, [str(base_path), str(override_path)])
            # Override values should be applied
            assert captured_config.get("trainer::max_epochs") == 1
            assert captured_config.get("trainer::devices") == 2
            # Model should be present (from override)
            assert captured_config.get("model::network::_target_") == "torch.nn.Identity"
            # New values from override should be added
            assert captured_config.get("model::criterion::_target_") == "torch.nn.MSELoss"
            assert captured_config.get("data::val_dataloader") is not None

    def test_cli_overrides_apply(self, temp_config_dir):
        """Test that CLI overrides are applied correctly."""
        config_path = temp_config_dir / "base.yaml"
        config_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10
  devices: 1

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
  train_dataloader: {}
  val_dataloader: {}
"""
        config_path.write_text(config_content)

        runner = Runner()
        # Capture config to verify values
        captured_config = None

        def capture_system(config):
            nonlocal captured_config
            captured_config = config
            return MagicMock()

        with (
            patch.object(runner, "_resolve_model", side_effect=capture_system),
            patch.object(runner, "_resolve_trainer"),
            patch.object(runner, "_resolve_datamodule"),
            patch.object(runner, "_execute"),
        ):
            overrides = ["trainer::max_epochs=5", "model::optimizer::lr=0.1"]
            runner.run(Stage.FIT, [str(config_path)] + overrides)
            assert captured_config.get("trainer::max_epochs") == 5
            assert captured_config.get("model::optimizer::lr") == 0.1

    def test_config_with_references(self, temp_config_dir):
        """Test configuration with Sparkwheel references."""
        config_path = temp_config_dir / "references.yaml"
        config_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1

model:
  _target_: lighter.LighterModule
  network:
    _target_: torch.nn.Identity
  train_metrics:
    - _target_: torchmetrics.MeanSquaredError
  val_metrics: "%::train_metrics"

data:
  _target_: lighter.LighterDataModule
  train_dataloader: {batch_size: 32}
  val_dataloader: "%::train_dataloader"
"""
        config_path.write_text(config_content)

        runner = Runner()
        # Capture config to verify values
        captured_config = None

        def capture_system(config):
            nonlocal captured_config
            captured_config = config
            return MagicMock()

        with (
            patch.object(runner, "_resolve_model", side_effect=capture_system),
            patch.object(runner, "_resolve_trainer"),
            patch.object(runner, "_resolve_datamodule"),
            patch.object(runner, "_execute"),
        ):
            runner.run(Stage.FIT, [str(config_path)])
            # Raw references (%) remain as strings until resolution
            # They are expanded when resolved, not during config loading
            train_metrics = captured_config.get("model::train_metrics")
            val_metrics_ref = captured_config.get("model::val_metrics")

            # val_metrics should be a reference string or the expanded value
            # depending on Sparkwheel version behavior
            assert train_metrics == [{"_target_": "torchmetrics.MeanSquaredError"}]
            # The reference either stays as string or gets expanded
            assert val_metrics_ref == "%::train_metrics" or val_metrics_ref == train_metrics

    def test_config_with_vars(self, temp_config_dir):
        """Test configuration with vars section."""
        config_path = temp_config_dir / "with_vars.yaml"
        config_content = """
vars:
  learning_rate: 0.001
  batch_size: 32

trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1

model:
  _target_: lighter.LighterModule
  network:
    _target_: torch.nn.Identity
  optimizer:
    _target_: torch.optim.Adam
    lr: "@vars::learning_rate"

data:
  _target_: lighter.LighterDataModule
  train_dataloader:
    batch_size: "@vars::batch_size"
  val_dataloader:
    batch_size: "@vars::batch_size"
"""
        config_path.write_text(config_content)

        runner = Runner()
        # Capture config to verify values
        captured_config = None

        def capture_system(config):
            nonlocal captured_config
            captured_config = config
            return MagicMock()

        with (
            patch.object(runner, "_resolve_model", side_effect=capture_system),
            patch.object(runner, "_resolve_trainer"),
            patch.object(runner, "_resolve_datamodule"),
            patch.object(runner, "_execute"),
        ):
            runner.run(Stage.FIT, [str(config_path)])
            # Vars should be accessible
            assert captured_config.get("vars::learning_rate") == 0.001
            assert captured_config.get("vars::batch_size") == 32
