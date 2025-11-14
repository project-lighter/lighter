"""Unit tests for error handling in Runner class"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from sparkwheel import Config

from lighter.engine.runner import Runner
from lighter.utils.types.enums import Stage


class TestRunnerErrorHandling:
    """Tests for Runner error handling and edge cases."""

    def test_run_with_nonexistent_config_file_raises_error(self):
        """Test that non-existent config file raises appropriate error."""
        runner = Runner()
        with pytest.raises(FileNotFoundError):
            runner.run(Stage.FIT, "nonexistent_config.yaml")

    def test_run_with_invalid_yaml_raises_error(self):
        """Test that invalid YAML raises appropriate error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            runner = Runner()
            with pytest.raises(yaml.YAMLError):
                runner.run(Stage.FIT, config_path)
        finally:
            Path(config_path).unlink()

    def test_run_with_empty_config_raises_validation_error(self):
        """Test that empty config raises validation error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")  # Empty config
            config_path = f.name

        try:
            runner = Runner()
            with pytest.raises(ValueError, match="validation failed"):
                runner.run(Stage.FIT, config_path)
        finally:
            Path(config_path).unlink()

    def test_prune_without_loaded_config_raises_error(self):
        """Test that pruning without loaded config raises ValueError."""
        runner = Runner()
        with pytest.raises(ValueError, match="Config must be loaded"):
            runner._prune_for_stage(Stage.FIT)

    def test_setup_without_loaded_config_raises_error(self):
        """Test that setup without loaded config raises ValueError."""
        runner = Runner()
        with pytest.raises(ValueError, match="Config must be loaded"):
            runner._setup(Stage.FIT)

    def test_execute_without_loaded_config_raises_error(self):
        """Test that execute without loaded config raises ValueError."""
        runner = Runner()
        with pytest.raises(ValueError, match="Config.*must be set up"):
            runner._execute(Stage.FIT)

    def test_execute_without_trainer_raises_error(self):
        """Test that execute without trainer raises ValueError."""
        runner = Runner()
        runner.config = MagicMock()
        runner.system = MagicMock()
        runner.trainer = None
        with pytest.raises(ValueError, match="trainer.*must be set up"):
            runner._execute(Stage.FIT)

    def test_execute_without_system_raises_error(self):
        """Test that execute without system raises ValueError."""
        runner = Runner()
        runner.config = MagicMock()
        runner.trainer = MagicMock()
        runner.system = None
        with pytest.raises(ValueError, match="system.*must be set up"):
            runner._execute(Stage.FIT)

    def test_run_with_invalid_override_format_raises_error(self):
        """Test that invalid override format raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  dataloaders:
    train: {}
    val: {}
""")
            config_path = f.name

        try:
            runner = Runner()
            # Sparkwheel raises ValueError for invalid override format
            with pytest.raises(ValueError, match="Invalid override format"):
                runner.run(Stage.FIT, config_path, ["invalid_override_no_equals"])
        finally:
            Path(config_path).unlink()

    def test_run_with_invalid_project_path_raises_error(self):
        """Test that invalid project path raises FileNotFoundError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
project: /nonexistent/path/to/project

trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  dataloaders:
    train: {}
    val: {}
""")
            config_path = f.name

        try:
            runner = Runner()
            with pytest.raises(FileNotFoundError):
                runner.run(Stage.FIT, config_path)
        finally:
            Path(config_path).unlink()

    def test_run_with_conflicting_overrides(self):
        """Test behavior with conflicting CLI overrides (last one wins)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  dataloaders:
    train: {}
    val: {}
""")
            config_path = f.name

        try:
            runner = Runner()
            overrides = [
                "trainer::max_epochs=5",
                "trainer::max_epochs=20",  # Conflicting - should override previous
            ]
            with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
                runner.run(Stage.FIT, config_path, overrides)
                # Last override should win
                assert runner.config.get("trainer::max_epochs") == 20
        finally:
            Path(config_path).unlink()

    def test_prune_with_all_stages_defined(self):
        """Test pruning behavior when all stages are defined."""
        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
            "system": {
                "_target_": "lighter.System",
                "model": {"_target_": "torch.nn.Identity"},
                "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
                "scheduler": {"_target_": "torch.optim.lr_scheduler.StepLR"},
                "criterion": {"_target_": "torch.nn.MSELoss"},
                "dataloaders": {
                    "train": {},
                    "val": {},
                    "test": {},
                    "predict": {},
                },
                "metrics": {
                    "train": [],
                    "val": [],
                    "test": [],
                },
                "adapters": {
                    "train": {},
                    "val": {},
                    "test": {},
                    "predict": {},
                },
            },
            "args": {
                "fit": {},
                "validate": {},
                "test": {},
                "predict": {},
            },
        }

        runner = Runner()
        runner.config = Config.load(config_dict)

        # Test pruning for VALIDATE stage
        runner._prune_for_stage(Stage.VALIDATE)

        # VALIDATE keeps: val dataloader, val metrics, criterion
        # VALIDATE removes: train/test/predict dataloaders, train/test metrics, optimizer, scheduler
        assert runner.config.get("system::dataloaders::val") is not None
        assert runner.config.get("system::dataloaders::train") is None
        assert runner.config.get("system::dataloaders::test") is None
        assert runner.config.get("system::dataloaders::predict") is None

        assert runner.config.get("system::metrics::val") is not None
        assert runner.config.get("system::metrics::train") is None
        assert runner.config.get("system::metrics::test") is None

        assert runner.config.get("system::optimizer") is None
        assert runner.config.get("system::scheduler") is None
        assert runner.config.get("system::criterion") is not None  # Kept for VALIDATE

        assert runner.config.get("args::validate") is not None
        assert runner.config.get("args::fit") is None
        assert runner.config.get("args::test") is None
        assert runner.config.get("args::predict") is None

    def test_multiple_config_files_with_list(self):
        """Test loading multiple config files as a list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
            f1.write("""
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 100

system:
  _target_: lighter.System
  model:
    _target_: torch.nn.Identity
  dataloaders:
    train: {}
""")
            config_path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            f2.write("""
trainer:
  max_epochs: 1

system:
  model:
    _target_: torch.nn.Identity
  dataloaders:
    train: {}
    val: {}
""")
            config_path2 = f2.name

        try:
            runner = Runner()
            with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
                runner.run(Stage.FIT, [config_path1, config_path2])
                # Second file should override
                assert runner.config.get("trainer::max_epochs") == 1
                # Both dataloaders should exist (from second file)
                assert runner.config.get("system::dataloaders::train") is not None
                assert runner.config.get("system::dataloaders::val") is not None
        finally:
            Path(config_path1).unlink()
            Path(config_path2).unlink()

    def test_run_with_dict_config(self):
        """Test running with dict config instead of file path."""
        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
            "system": {
                "_target_": "lighter.System",
                "model": {"_target_": "torch.nn.Identity"},
                "dataloaders": {"train": {}, "val": {}},
            },
        }

        runner = Runner()
        with patch.object(runner, "_setup"), patch.object(runner, "_execute"):
            runner.run(Stage.FIT, config_dict)
            assert runner.config is not None
            assert runner.config.get("trainer::max_epochs") == 10

    def test_stage_modes_mapping(self):
        """Test that STAGE_MODES mapping is correct."""
        from lighter.utils.types.enums import Mode

        assert Runner.STAGE_MODES[Stage.FIT] == [Mode.TRAIN, Mode.VAL]
        assert Runner.STAGE_MODES[Stage.VALIDATE] == [Mode.VAL]
        assert Runner.STAGE_MODES[Stage.TEST] == [Mode.TEST]
        assert Runner.STAGE_MODES[Stage.PREDICT] == [Mode.PREDICT]

    def test_runner_initialization_state(self):
        """Test that Runner initializes with None state."""
        runner = Runner()
        assert runner.config is None
        assert runner.system is None
        assert runner.trainer is None

    def test_execute_with_stage_args(self):
        """Test that execute passes stage-specific args correctly."""
        runner = Runner()
        runner.config = MagicMock()
        runner.system = MagicMock()
        runner.trainer = MagicMock()

        # Mock resolve to return specific args
        runner.config.resolve.return_value = {"ckpt_path": "checkpoint.ckpt"}

        runner._execute(Stage.FIT)

        # Verify trainer.fit was called with the args
        runner.trainer.fit.assert_called_once_with(
            runner.system,
            ckpt_path="checkpoint.ckpt",
        )

    def test_execute_with_empty_stage_args(self):
        """Test that execute handles empty stage args (uses default)."""
        runner = Runner()
        runner.config = MagicMock()
        runner.system = MagicMock()
        runner.trainer = MagicMock()

        # Mock resolve to return default empty dict
        runner.config.resolve.return_value = {}

        runner._execute(Stage.TEST)

        # Verify trainer.test was called with no extra args
        runner.trainer.test.assert_called_once_with(runner.system)

    def test_setup_logs_config_to_trainer_logger(self):
        """Test that setup logs config to trainer logger if available."""
        from pytorch_lightning import Trainer

        from lighter.system import System

        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer"},
            "system": {
                "_target_": "lighter.System",
                "model": {"_target_": "torch.nn.Identity"},
                "dataloaders": {"train": {}},
            },
        }

        runner = Runner()
        runner.config = Config.load(config_dict)

        mock_system = MagicMock(spec=System)
        mock_trainer = MagicMock(spec=Trainer)
        mock_logger = MagicMock()
        mock_trainer.logger = mock_logger

        with patch.object(runner.config, "resolve") as mock_resolve:
            mock_resolve.side_effect = [mock_system, mock_trainer]
            runner._setup(Stage.FIT)

            # Verify hyperparameters were logged
            mock_system.save_hyperparameters.assert_called_once()
            mock_logger.log_hyperparams.assert_called_once()

    def test_setup_handles_no_trainer_logger(self):
        """Test that setup handles case when trainer has no logger."""
        from pytorch_lightning import Trainer

        from lighter.system import System

        config_dict = {
            "trainer": {"_target_": "pytorch_lightning.Trainer"},
            "system": {
                "_target_": "lighter.System",
                "model": {"_target_": "torch.nn.Identity"},
                "dataloaders": {"train": {}},
            },
        }

        runner = Runner()
        runner.config = Config.load(config_dict)

        mock_system = MagicMock(spec=System)
        mock_trainer = MagicMock(spec=Trainer)
        mock_trainer.logger = None  # No logger

        with patch.object(runner.config, "resolve") as mock_resolve:
            mock_resolve.side_effect = [mock_system, mock_trainer]
            # Should not raise even without logger
            runner._setup(Stage.FIT)
            mock_system.save_hyperparameters.assert_called_once()
