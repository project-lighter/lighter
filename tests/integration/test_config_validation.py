"""Integration tests for configuration validation with real YAML files."""

import tempfile
from pathlib import Path
from unittest.mock import patch

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

    def test_minimal_valid_config_loads_and_validates(self, temp_config_dir):
        """Test that minimal valid config loads and validates."""
        config_path = temp_config_dir / "minimal.yaml"
        config_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  dataloaders:
    train: {}
    val: {}
"""
        config_path.write_text(config_content)

        runner = Runner()
        # Patch _setup and _execute to avoid needing real components
        with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
            # Should not raise ValidationError
            runner.run(Stage.FIT, str(config_path))
            assert runner.config is not None

    def test_config_with_all_optional_fields_validates(self, temp_config_dir):
        """Test config with all optional fields validates."""
        config_path = temp_config_dir / "full.yaml"
        config_content = """
project: ./path/to/project

vars:
  learning_rate: 0.001

trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  criterion:
    _target_: torch.nn.MSELoss
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
  scheduler:
    _target_: torch.optim.lr_scheduler.StepLR
    step_size: 10
  inferer:
    _target_: lighter.Inferer
  metrics:
    train: []
    val: []
    test: []
  dataloaders:
    train: {}
    val: {}
    test: {}
    predict: {}
  adapters:
    train: {}
    val: {}
    test: {}
    predict: {}

args:
  fit:
    ckpt_path: checkpoint.ckpt
  validate: {}
  test: {}
  predict: {}
"""
        config_path.write_text(config_content)

        runner = Runner()
        with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
            runner.run(Stage.FIT, str(config_path))
            assert runner.config.get("project") == "./path/to/project"
            assert runner.config.get("vars::learning_rate") == 0.001
            assert runner.config.get("system::optimizer::lr") == 0.001

    def test_missing_trainer_raises_validation_error(self, temp_config_dir):
        """Test that missing trainer field raises ValidationError."""
        config_path = temp_config_dir / "invalid_no_trainer.yaml"
        config_content = """
system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  dataloaders:
    train: {}
"""
        config_path.write_text(config_content)

        runner = Runner()
        with pytest.raises(ValueError, match="validation failed"):
            runner.run(Stage.FIT, str(config_path))

    def test_missing_system_raises_validation_error(self, temp_config_dir):
        """Test that missing system field raises ValidationError."""
        config_path = temp_config_dir / "invalid_no_system.yaml"
        config_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1
"""
        config_path.write_text(config_content)

        runner = Runner()
        with pytest.raises(ValueError, match="validation failed"):
            runner.run(Stage.FIT, str(config_path))

    def test_wrong_type_for_trainer_raises_validation_error(self, temp_config_dir):
        """Test that wrong type for trainer raises ValidationError."""
        config_path = temp_config_dir / "invalid_trainer_type.yaml"
        config_content = """
trainer:
  - this_should_be_a_dict
  - not_a_list

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  dataloaders:
    train: {}
"""
        config_path.write_text(config_content)

        runner = Runner()
        with pytest.raises(ValueError, match="validation failed"):
            runner.run(Stage.FIT, str(config_path))

    def test_wrong_type_for_system_raises_validation_error(self, temp_config_dir):
        """Test that wrong type for system raises ValidationError."""
        config_path = temp_config_dir / "invalid_system_type.yaml"
        config_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1

system: "this should be a dict"
"""
        config_path.write_text(config_content)

        runner = Runner()
        with pytest.raises(ValueError, match="validation failed"):
            runner.run(Stage.FIT, str(config_path))

    def test_pruning_removes_unused_dataloaders_for_fit_stage(self, temp_config_dir):
        """Test that pruning removes test/predict dataloaders for FIT stage."""
        config_path = temp_config_dir / "all_dataloaders.yaml"
        config_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  dataloaders:
    train: {batch_size: 32}
    val: {batch_size: 32}
    test: {batch_size: 32}
    predict: {batch_size: 32}
"""
        config_path.write_text(config_content)

        runner = Runner()
        with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
            runner.run(Stage.FIT, str(config_path))
            # After pruning, only train and val should remain
            assert runner.config.get("system::dataloaders::train") is not None
            assert runner.config.get("system::dataloaders::val") is not None
            assert runner.config.get("system::dataloaders::test") is None
            assert runner.config.get("system::dataloaders::predict") is None

    def test_pruning_removes_unused_dataloaders_for_test_stage(self, temp_config_dir):
        """Test that pruning removes train/val/predict dataloaders for TEST stage."""
        config_path = temp_config_dir / "all_dataloaders.yaml"
        config_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  dataloaders:
    train: {batch_size: 32}
    val: {batch_size: 32}
    test: {batch_size: 32}
    predict: {batch_size: 32}
"""
        config_path.write_text(config_content)

        runner = Runner()
        with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
            runner.run(Stage.TEST, str(config_path))
            # After pruning, only test should remain
            assert runner.config.get("system::dataloaders::test") is not None
            assert runner.config.get("system::dataloaders::train") is None
            assert runner.config.get("system::dataloaders::val") is None
            assert runner.config.get("system::dataloaders::predict") is None

    def test_pruning_removes_unused_dataloaders_for_predict_stage(self, temp_config_dir):
        """Test that pruning removes train/val/test dataloaders for PREDICT stage."""
        config_path = temp_config_dir / "all_dataloaders.yaml"
        config_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  dataloaders:
    train: {batch_size: 32}
    val: {batch_size: 32}
    test: {batch_size: 32}
    predict: {batch_size: 32}
"""
        config_path.write_text(config_content)

        runner = Runner()
        with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
            runner.run(Stage.PREDICT, str(config_path))
            # After pruning, only predict should remain
            assert runner.config.get("system::dataloaders::predict") is not None
            assert runner.config.get("system::dataloaders::train") is None
            assert runner.config.get("system::dataloaders::val") is None
            assert runner.config.get("system::dataloaders::test") is None

    def test_pruning_removes_unused_metrics(self, temp_config_dir):
        """Test that pruning removes unused metrics for each stage."""
        config_path = temp_config_dir / "all_metrics.yaml"
        config_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  metrics:
    train: []
    val: []
    test: []
  dataloaders:
    train: {}
    val: {}
    test: {}
"""
        config_path.write_text(config_content)

        runner = Runner()
        with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
            runner.run(Stage.FIT, str(config_path))
            # FIT keeps train and val metrics
            assert runner.config.get("system::metrics::train") is not None
            assert runner.config.get("system::metrics::val") is not None
            assert runner.config.get("system::metrics::test") is None

    def test_pruning_removes_optimizer_for_non_fit_stages(self, temp_config_dir):
        """Test that pruning removes optimizer/scheduler for non-FIT stages."""
        config_path = temp_config_dir / "with_optimizer.yaml"
        config_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
  scheduler:
    _target_: torch.optim.lr_scheduler.StepLR
    step_size: 10
  criterion:
    _target_: torch.nn.MSELoss
  dataloaders:
    test: {}
"""
        config_path.write_text(config_content)

        runner = Runner()
        with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
            runner.run(Stage.TEST, str(config_path))
            # TEST stage should remove optimizer, scheduler, and criterion
            assert runner.config.get("system::optimizer") is None
            assert runner.config.get("system::scheduler") is None
            assert runner.config.get("system::criterion") is None

    def test_pruning_keeps_criterion_for_validate_stage(self, temp_config_dir):
        """Test that VALIDATE stage keeps criterion but removes optimizer."""
        config_path = temp_config_dir / "validate_config.yaml"
        config_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
  criterion:
    _target_: torch.nn.MSELoss
  dataloaders:
    val: {}
"""
        config_path.write_text(config_content)

        runner = Runner()
        with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
            runner.run(Stage.VALIDATE, str(config_path))
            # VALIDATE removes optimizer but keeps criterion
            assert runner.config.get("system::optimizer") is None
            assert runner.config.get("system::criterion") is not None

    def test_multi_file_config_merge(self, temp_config_dir):
        """Test that multiple config files merge correctly."""
        base_path = temp_config_dir / "base.yaml"
        base_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 100
  devices: 1

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  dataloaders:
    train: {}
"""
        base_path.write_text(base_content)

        override_path = temp_config_dir / "override.yaml"
        override_content = """
trainer:
  max_epochs: 1
  devices: 2

system:
  model:
    _target_: torch.nn.Identity
  criterion:
    _target_: torch.nn.MSELoss
  dataloaders:
    train: {}
    val: {}
"""
        override_path.write_text(override_content)

        runner = Runner()
        with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
            runner.run(Stage.FIT, [str(base_path), str(override_path)])
            # Override values should be applied
            assert runner.config.get("trainer::max_epochs") == 1
            assert runner.config.get("trainer::devices") == 2
            # Model should be present (from override)
            assert runner.config.get("system::model::_target_") == "torch.nn.Identity"
            # New values from override should be added
            assert runner.config.get("system::criterion::_target_") == "torch.nn.MSELoss"
            assert runner.config.get("system::dataloaders::val") is not None

    def test_comma_separated_config_files(self, temp_config_dir):
        """Test that comma-separated config file paths work."""
        base_path = temp_config_dir / "base.yaml"
        base_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 100

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  dataloaders:
    train: {}
"""
        base_path.write_text(base_content)

        override_path = temp_config_dir / "override.yaml"
        override_content = """
trainer:
  max_epochs: 1

system:
  dataloaders:
    val: {}
"""
        override_path.write_text(override_content)

        runner = Runner()
        with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
            # Use comma-separated string
            config_str = f"{base_path},{override_path}"
            runner.run(Stage.FIT, config_str)
            assert runner.config.get("trainer::max_epochs") == 1

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
  dataloaders:
    train: {}
    val: {}
"""
        config_path.write_text(config_content)

        runner = Runner()
        with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
            overrides = ["trainer::max_epochs=5", "system::optimizer::lr=0.1"]
            runner.run(Stage.FIT, str(config_path), overrides)
            assert runner.config.get("trainer::max_epochs") == 5
            assert runner.config.get("system::optimizer::lr") == 0.1

    def test_stage_specific_args_pruned(self, temp_config_dir):
        """Test that stage-specific args are preserved and others pruned."""
        config_path = temp_config_dir / "with_args.yaml"
        config_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  dataloaders:
    train: {}
    val: {}

args:
  fit:
    ckpt_path: checkpoint.ckpt
  validate:
    verbose: true
  test:
    verbose: false
  predict:
    return_predictions: true
"""
        config_path.write_text(config_content)

        runner = Runner()
        with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
            runner.run(Stage.FIT, str(config_path))
            # Fit args should be preserved
            assert runner.config.get("args::fit") is not None
            # Other args should be pruned
            assert runner.config.get("args::validate") is None
            assert runner.config.get("args::test") is None
            assert runner.config.get("args::predict") is None

    def test_config_with_references(self, temp_config_dir):
        """Test configuration with Sparkwheel references."""
        config_path = temp_config_dir / "references.yaml"
        config_content = """
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  metrics:
    train:
      - _target_: torchmetrics.MeanSquaredError
    val: "%::train"
  dataloaders:
    train: {batch_size: 32}
    val: "%::train"
"""
        config_path.write_text(config_content)

        runner = Runner()
        with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
            runner.run(Stage.FIT, str(config_path))
            # Before resolution, references should exist
            assert runner.config.get("system::metrics::val") == "%::train"
            assert runner.config.get("system::dataloaders::val") == "%::train"

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

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  optimizer:
    _target_: torch.optim.Adam
    lr: "@vars::learning_rate"
  dataloaders:
    train:
      batch_size: "@vars::batch_size"
    val:
      batch_size: "@vars::batch_size"
"""
        config_path.write_text(config_content)

        runner = Runner()
        with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
            runner.run(Stage.FIT, str(config_path))
            # Vars should be accessible
            assert runner.config.get("vars::learning_rate") == 0.001
            assert runner.config.get("vars::batch_size") == 32
