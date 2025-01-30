"""Unit tests for the Runner class in lighter/engine/runner.py"""

from unittest.mock import MagicMock, patch

import pytest
from pytorch_lightning import Trainer
from pytorch_lightning.tuner import Tuner

from lighter.engine.runner import Runner, cli
from lighter.system import System
from lighter.utils.types.enums import Stage


@pytest.fixture
def mock_system():
    """Fixture providing a mock System instance."""
    system = MagicMock(spec=System)
    system.save_hyperparameters = MagicMock()
    return system


@pytest.fixture
def mock_trainer():
    """Fixture providing a mock Trainer instance."""
    trainer = MagicMock(spec=Trainer)
    trainer.logger = MagicMock()
    trainer.logger.log_hyperparams = MagicMock()
    trainer.fit = MagicMock()
    trainer.validate = MagicMock()
    trainer.test = MagicMock()
    trainer.predict = MagicMock()
    return trainer


@pytest.fixture
def mock_config():
    """Fixture providing a mock configuration."""
    return {
        "system": {
            "model": "test_model",
            "optimizer": {"name": "Adam", "lr": 0.001},
        },
        "trainer": {
            "max_epochs": 10,
            "accelerator": "auto",
        },
        "args": {
            "fit": {"some_arg": "value"},
        },
        "project": None,
    }


@pytest.fixture
def runner():
    """Fixture providing a Runner instance."""
    return Runner()


def test_runner_initialization(runner):
    """Test that Runner initializes with correct default values."""
    assert runner.config is None
    assert runner.resolver is None
    assert runner.system is None
    assert runner.trainer is None
    assert runner.args is None


@patch("lighter.engine.runner.Config")
@patch("lighter.engine.runner.Resolver")
@patch("lighter.engine.runner.seed_everything")
@patch("lighter.engine.runner.import_module_from_path")
def test_run_setup(
    mock_import, mock_seed, mock_resolver_class, mock_config_class, runner, mock_config, mock_system, mock_trainer
):
    """Test that run method sets up everything correctly."""
    # Setup mocks
    mock_config_instance = MagicMock()
    mock_config_instance.get.return_value = None  # No project path
    mock_config_class.return_value = mock_config_instance

    mock_resolver_instance = MagicMock()
    mock_stage_config = MagicMock()
    mock_stage_config.get.return_value = None  # No project path
    mock_stage_config.get_parsed_content.side_effect = [mock_system, mock_trainer, {}]
    mock_resolver_instance.get_stage_config.return_value = mock_stage_config
    mock_resolver_class.return_value = mock_resolver_instance

    # Run with config
    runner.run(Stage.FIT, mock_config)

    # Verify seed was set
    mock_seed.assert_called_once()

    # Verify config was created
    mock_config_class.assert_called_once()

    # Verify resolver was created
    mock_resolver_class.assert_called_once_with(mock_config_instance)

    # Verify system and trainer were set up
    assert runner.system == mock_system
    assert runner.trainer == mock_trainer


def test_setup_stage_with_invalid_system(runner, mock_trainer):
    """Test that _setup_stage raises error for invalid system type."""
    mock_config = MagicMock()
    mock_config.get.return_value = None  # No project path
    mock_config.get_parsed_content.side_effect = ["not a system", mock_trainer, {}]

    runner.resolver = MagicMock()
    runner.resolver.get_stage_config.return_value = mock_config

    with pytest.raises(ValueError, match="'system' must be an instance of System"):
        runner._setup_stage(Stage.FIT)


def test_setup_stage_with_invalid_trainer(runner, mock_system):
    """Test that _setup_stage raises error for invalid trainer type."""
    mock_config = MagicMock()
    mock_config.get.return_value = None  # No project path
    mock_config.get_parsed_content.side_effect = [mock_system, "not a trainer", {}]

    runner.resolver = MagicMock()
    runner.resolver.get_stage_config.return_value = mock_config

    with pytest.raises(ValueError, match="'trainer' must be an instance of Trainer"):
        runner._setup_stage(Stage.FIT)


@patch("lighter.engine.runner.import_module_from_path")
def test_setup_stage_with_project(mock_import, runner, mock_system, mock_trainer):
    """Test that _setup_stage correctly imports project module."""
    mock_config = MagicMock()
    mock_config.get.return_value = "path/to/project"
    mock_config.get_parsed_content.side_effect = [mock_system, mock_trainer, {}]

    runner.resolver = MagicMock()
    runner.resolver.get_stage_config.return_value = mock_config
    runner.config = MagicMock()  # Add config for _save_config

    runner._setup_stage(Stage.FIT)
    mock_import.assert_called_once_with("project", "path/to/project")


def test_save_config(runner, mock_system, mock_trainer):
    """Test that _save_config correctly saves configuration."""
    runner.system = mock_system
    runner.trainer = mock_trainer
    runner.config = MagicMock()

    runner._save_config()

    mock_system.save_hyperparameters.assert_called_once_with(runner.config.get())
    mock_trainer.logger.log_hyperparams.assert_called_once_with(runner.config.get())


def test_save_config_without_logger(runner, mock_system):
    """Test that _save_config works when trainer has no logger."""
    runner.system = mock_system
    runner.trainer = MagicMock(spec=Trainer, logger=None)
    runner.config = MagicMock()

    runner._save_config()

    mock_system.save_hyperparameters.assert_called_once_with(runner.config.get())


@patch("lighter.engine.runner.Tuner")
def test_run_stage_normal(mock_tuner_class, runner, mock_system, mock_trainer):
    """Test running normal stages (fit, validate, test, predict)."""
    runner.system = mock_system
    runner.trainer = mock_trainer
    runner.args = {"some_arg": "value"}

    # Test fit stage
    runner._run_stage(Stage.FIT)
    mock_trainer.fit.assert_called_once_with(mock_system, some_arg="value")
    mock_trainer.reset_mock()

    # Test validate stage
    runner._run_stage(Stage.VALIDATE)
    mock_trainer.validate.assert_called_once_with(mock_system, some_arg="value")
    mock_trainer.reset_mock()

    # Test test stage
    runner._run_stage(Stage.TEST)
    mock_trainer.test.assert_called_once_with(mock_system, some_arg="value")
    mock_trainer.reset_mock()

    # Test predict stage
    runner._run_stage(Stage.PREDICT)
    mock_trainer.predict.assert_called_once_with(mock_system, some_arg="value")


@patch("lighter.engine.runner.Tuner")
def test_run_stage_tuner(mock_tuner_class, runner, mock_system, mock_trainer):
    """Test running tuner stages (lr_find, scale_batch_size)."""
    runner.system = mock_system
    runner.trainer = mock_trainer
    runner.args = {"some_arg": "value"}

    mock_tuner = MagicMock()
    mock_tuner_class.return_value = mock_tuner

    # Test lr_find stage
    runner._run_stage(Stage.LR_FIND)
    mock_tuner.lr_find.assert_called_once_with(mock_system, some_arg="value")
    mock_tuner.reset_mock()

    # Test scale_batch_size stage
    runner._run_stage(Stage.SCALE_BATCH_SIZE)
    mock_tuner.scale_batch_size.assert_called_once_with(mock_system, some_arg="value")


def test_run_stage_invalid_stage(runner, mock_trainer):
    """Test that _run_stage raises AttributeError for invalid stage."""
    runner.trainer = mock_trainer
    runner.system = MagicMock()
    runner.args = {}

    with pytest.raises(AttributeError):
        runner._run_stage("invalid_stage")


@patch("lighter.engine.runner.Runner")
@patch("lighter.engine.runner.fire.Fire")
def test_cli_commands(mock_fire, mock_runner_class):
    """Test CLI command functions."""
    mock_runner = MagicMock()
    mock_runner_class.return_value = mock_runner

    # Call cli() to get the command dict
    cli()
    command_dict = mock_fire.call_args[0][0]

    # Test each command function
    config = "config.yaml"
    kwargs = {"arg1": "value1"}

    for command, func in command_dict.items():
        func(config, **kwargs)
        mock_runner.run.assert_called_with(getattr(Stage, command.upper()), config, **kwargs)
        mock_runner.reset_mock()


@patch("lighter.engine.runner.fire.Fire")
def test_cli_fire_interface(mock_fire):
    """Test that cli() calls fire.Fire with correct command dict."""
    cli()
    mock_fire.assert_called_once()
    command_dict = mock_fire.call_args[0][0]
    assert set(command_dict.keys()) == {
        "fit",
        "validate",
        "test",
        "predict",
        "lr_find",
        "scale_batch_size",
    }


def test_main_guard():
    """Test the __main__ guard."""
    with patch("lighter.engine.runner.cli") as mock_cli:
        # Execute the main module
        exec(
            compile(
                'if __name__ == "__main__": cli()',
                filename="lighter/engine/runner.py",
                mode="exec",
            ),
            {"__name__": "__main__", "cli": mock_cli},
        )
        mock_cli.assert_called_once()

        # Reset mock and test when not main
        mock_cli.reset_mock()
        exec(
            compile(
                'if __name__ == "__main__": cli()',
                filename="lighter/engine/runner.py",
                mode="exec",
            ),
            {"__name__": "not_main", "cli": mock_cli},
        )
        mock_cli.assert_not_called()
