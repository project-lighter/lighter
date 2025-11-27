"""Unit tests for error handling in the Runner class"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from sparkwheel.utils.exceptions import ConfigKeyError

from lighter.engine.runner import Runner
from lighter.utils.types.enums import Stage


class TestRunnerErrorHandling:
    """Test class for Runner error handling scenarios."""

    def test_run_without_config_raises_error(self):
        """Test that calling run without config_paths raises ConfigKeyError."""
        runner = Runner()
        with pytest.raises(ConfigKeyError):  # Sparkwheel raises ConfigKeyError for missing keys
            runner.run(Stage.FIT, [])

    def test_run_with_nonexistent_config_raises_error(self):
        """Test that nonexistent config file raises FileNotFoundError."""
        runner = Runner()
        with pytest.raises(FileNotFoundError):
            runner.run(Stage.FIT, ["/nonexistent/path/config.yaml"])

    def test_run_with_empty_config_raises_validation_error(self):
        """Test that empty config raises ConfigKeyError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")  # Empty config
            config_path = f.name

        try:
            runner = Runner()
            # Empty config should raise ConfigKeyError when trying to resolve 'model'
            with pytest.raises(ConfigKeyError):
                runner.run(Stage.FIT, [config_path])
        finally:
            Path(config_path).unlink()

    def test_multiple_config_files_with_list(self):
        """Test loading multiple config files as a list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
            f1.write("trainer:\n  _target_: pytorch_lightning.Trainer\n  max_epochs: 1\n")
            config_path1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            f2.write("model:\n  _target_: lighter.LighterModule\n  model:\n    _target_: torch.nn.Identity\n")
            config_path2 = f2.name

        try:
            runner = Runner()
            # Should load multiple configs successfully
            # Just test that it doesn't raise during loading
            with (
                patch.object(runner, "_resolve_model"),
                patch.object(runner, "_resolve_trainer"),
                patch.object(runner, "_resolve_datamodule"),
                patch.object(runner, "_save_config"),
                patch.object(runner, "_save_hyperparameters"),
                patch.object(runner, "_execute"),
            ):
                runner.run(Stage.FIT, [config_path1, config_path2])
        finally:
            Path(config_path1).unlink()
            Path(config_path2).unlink()
