"""Unit tests for the CLI functionality in lighter/engine/runner.py"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lighter.engine.runner import cli


class TestCLI:
    """Tests for CLI argument parsing and command execution."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
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
    test: {}
    predict: {}
""")
            config_path = f.name
        yield config_path
        Path(config_path).unlink()

    def test_cli_fit_command_basic(self, temp_config_file):
        """Test fit command with basic config."""
        test_args = ["lighter", "fit", temp_config_file]

        with patch.object(sys, "argv", test_args), patch("lighter.engine.runner.Runner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            cli()

            # Verify Runner was instantiated
            mock_runner_class.assert_called_once()
            # Verify run was called with correct arguments
            mock_runner.run.assert_called_once_with("fit", temp_config_file, [])

    def test_cli_fit_command_with_overrides(self, temp_config_file):
        """Test fit command with CLI overrides."""
        test_args = [
            "lighter",
            "fit",
            temp_config_file,
            "trainer::max_epochs=5",
            "trainer::devices=2",
        ]

        with patch.object(sys, "argv", test_args), patch("lighter.engine.runner.Runner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            cli()

            # Verify run was called with overrides
            mock_runner.run.assert_called_once_with(
                "fit",
                temp_config_file,
                ["trainer::max_epochs=5", "trainer::devices=2"],
            )

    def test_cli_validate_command(self, temp_config_file):
        """Test validate command."""
        test_args = ["lighter", "validate", temp_config_file]

        with patch.object(sys, "argv", test_args), patch("lighter.engine.runner.Runner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            cli()

            mock_runner.run.assert_called_once_with("validate", temp_config_file, [])

    def test_cli_validate_command_with_overrides(self, temp_config_file):
        """Test validate command with overrides."""
        test_args = [
            "lighter",
            "validate",
            temp_config_file,
            "system::model::weights=checkpoint.ckpt",
        ]

        with patch.object(sys, "argv", test_args), patch("lighter.engine.runner.Runner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            cli()

            mock_runner.run.assert_called_once_with(
                "validate",
                temp_config_file,
                ["system::model::weights=checkpoint.ckpt"],
            )

    def test_cli_test_command(self, temp_config_file):
        """Test test command."""
        test_args = ["lighter", "test", temp_config_file]

        with patch.object(sys, "argv", test_args), patch("lighter.engine.runner.Runner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            cli()

            mock_runner.run.assert_called_once_with("test", temp_config_file, [])

    def test_cli_test_command_with_overrides(self, temp_config_file):
        """Test test command with overrides."""
        test_args = [
            "lighter",
            "test",
            temp_config_file,
            "trainer::devices=1",
            "system::model::dropout=0.5",
        ]

        with patch.object(sys, "argv", test_args), patch("lighter.engine.runner.Runner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            cli()

            mock_runner.run.assert_called_once_with(
                "test",
                temp_config_file,
                ["trainer::devices=1", "system::model::dropout=0.5"],
            )

    def test_cli_predict_command(self, temp_config_file):
        """Test predict command."""
        test_args = ["lighter", "predict", temp_config_file]

        with patch.object(sys, "argv", test_args), patch("lighter.engine.runner.Runner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            cli()

            mock_runner.run.assert_called_once_with("predict", temp_config_file, [])

    def test_cli_predict_command_with_overrides(self, temp_config_file):
        """Test predict command with overrides."""
        test_args = [
            "lighter",
            "predict",
            temp_config_file,
            "system::model::weights=best.ckpt",
            "trainer::devices=4",
        ]

        with patch.object(sys, "argv", test_args), patch("lighter.engine.runner.Runner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            cli()

            mock_runner.run.assert_called_once_with(
                "predict",
                temp_config_file,
                ["system::model::weights=best.ckpt", "trainer::devices=4"],
            )

    def test_cli_missing_command(self):
        """Test that missing command raises error."""
        test_args = ["lighter"]

        with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
            cli()

    def test_cli_invalid_command(self):
        """Test that invalid command raises error."""
        test_args = ["lighter", "invalid_command", "config.yaml"]

        with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
            cli()

    def test_cli_missing_config(self):
        """Test that missing config argument raises error."""
        test_args = ["lighter", "fit"]

        with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
            cli()

    def test_cli_error_propagation(self, temp_config_file):
        """Test that exceptions from runner.run are propagated without chaining."""
        test_args = ["lighter", "fit", temp_config_file]

        with patch.object(sys, "argv", test_args), patch("lighter.engine.runner.Runner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner
            # Make run raise an exception
            mock_runner.run.side_effect = ValueError("Configuration error")

            with pytest.raises(ValueError, match="Configuration error") as exc_info:
                cli()

            # Verify the exception has no __cause__ (raise e from None)
            assert exc_info.value.__cause__ is None

    def test_cli_multiple_overrides(self, temp_config_file):
        """Test handling multiple overrides."""
        test_args = [
            "lighter",
            "fit",
            temp_config_file,
            "trainer::max_epochs=100",
            "trainer::devices=2",
            "system::optimizer::lr=0.001",
            "system::optimizer::weight_decay=0.0001",
        ]

        with patch.object(sys, "argv", test_args), patch("lighter.engine.runner.Runner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            cli()

            # Verify all overrides are passed
            mock_runner.run.assert_called_once()
            _, args, kwargs = mock_runner.run.mock_calls[0]
            assert args[0] == "fit"
            assert args[1] == temp_config_file
            assert len(args[2]) == 4
            assert "trainer::max_epochs=100" in args[2]
            assert "trainer::devices=2" in args[2]
            assert "system::optimizer::lr=0.001" in args[2]
            assert "system::optimizer::weight_decay=0.0001" in args[2]

    def test_cli_all_stages_independent(self, temp_config_file):
        """Test that each stage command is independent."""
        stages = ["fit", "validate", "test", "predict"]

        for stage in stages:
            test_args = ["lighter", stage, temp_config_file]

            with patch.object(sys, "argv", test_args), patch("lighter.engine.runner.Runner") as mock_runner_class:
                mock_runner = MagicMock()
                mock_runner_class.return_value = mock_runner

                cli()

                # Verify correct stage is called
                mock_runner.run.assert_called_once_with(stage, temp_config_file, [])

    def test_cli_comma_separated_configs_as_single_arg(self):
        """Test that comma-separated config paths work as a single argument."""
        test_args = ["lighter", "fit", "config1.yaml,config2.yaml"]

        with patch.object(sys, "argv", test_args), patch("lighter.engine.runner.Runner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            cli()

            # Runner should receive the comma-separated string as-is
            mock_runner.run.assert_called_once_with(
                "fit",
                "config1.yaml,config2.yaml",
                [],
            )

    def test_cli_override_with_equals_in_value(self, temp_config_file):
        """Test that overrides with = in the value are handled correctly."""
        test_args = [
            "lighter",
            "fit",
            temp_config_file,
            "system::model::config=key1=value1",
        ]

        with patch.object(sys, "argv", test_args), patch("lighter.engine.runner.Runner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            cli()

            # Verify override is passed correctly
            mock_runner.run.assert_called_once()
            _, args, kwargs = mock_runner.run.mock_calls[0]
            assert "system::model::config=key1=value1" in args[2]

    def test_cli_override_with_special_characters(self, temp_config_file):
        """Test overrides with special characters."""
        test_args = [
            "lighter",
            "fit",
            temp_config_file,
            "system::model::name=my-model_v1.0",
        ]

        with patch.object(sys, "argv", test_args), patch("lighter.engine.runner.Runner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            cli()

            mock_runner.run.assert_called_once()
            _, args, kwargs = mock_runner.run.mock_calls[0]
            assert "system::model::name=my-model_v1.0" in args[2]

    def test_cli_no_overrides(self, temp_config_file):
        """Test that no overrides results in empty list."""
        test_args = ["lighter", "fit", temp_config_file]

        with patch.object(sys, "argv", test_args), patch("lighter.engine.runner.Runner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner

            cli()

            mock_runner.run.assert_called_once()
            _, args, kwargs = mock_runner.run.mock_calls[0]
            assert args[2] == []  # Empty overrides list
