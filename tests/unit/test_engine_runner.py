from unittest.mock import MagicMock, patch

import pytest
from pytorch_lightning import Trainer

from lighter.engine.runner import cli, parse_config, run
from lighter.system import LighterSystem


@pytest.fixture
def mock_system():
    """
    Creates a mock LighterSystem instance for testing.

    Returns:
        MagicMock: A mock object that simulates a LighterSystem instance.
    """
    return MagicMock(spec=LighterSystem)


@pytest.fixture
def mock_trainer():
    """
    Creates a mock PyTorch Lightning Trainer instance for testing.

    Returns:
        MagicMock: A mock object that simulates a Trainer instance.
    """
    return MagicMock(spec=Trainer)


@pytest.fixture
def mock_parse_config():
    """
    Patches the parse_config function for testing.

    Yields:
        MagicMock: A mock object that simulates the parse_config function.
    """
    with patch("lighter.engine.runner.parse_config") as mock:
        yield mock


@pytest.fixture
def mock_trainer_class():
    """
    Patches the Trainer class for testing while maintaining its specification.

    Yields:
        MagicMock: A mock object that simulates the Trainer class.
    """
    with patch("lighter.engine.runner.Trainer", new=Trainer) as mock:
        yield mock


@pytest.fixture
def mock_seed_everything():
    """
    Patches the seed_everything function for testing.

    Yields:
        MagicMock: A mock object that simulates the seed_everything function.
    """
    with patch("lighter.engine.runner.seed_everything") as mock:
        yield mock


@pytest.fixture
def base_config(mock_system, mock_trainer):
    """
    Creates a basic configuration function for testing.

    Args:
        mock_system: Mock LighterSystem instance
        mock_trainer: Mock Trainer instance

    Returns:
        callable: A function that returns configuration values based on keys.
    """
    return lambda x, default=None: {
        "project": None,
        "system": mock_system,
        "trainer": mock_trainer,
    }.get(x, default)


def test_parse_config_no_config():
    """
    Tests that parse_config raises ValueError when no configuration is provided.

    Raises:
        ValueError: Expected to be raised when parse_config is called without arguments.
    """
    with pytest.raises(ValueError):
        parse_config()


def test_cli():
    """
    Tests that the CLI function properly calls fire.Fire.
    """
    with patch("lighter.engine.runner.fire.Fire") as mock_fire:
        cli()
        mock_fire.assert_called_once()


def test_run_invalid_method():
    """
    Tests that run raises ValueError when an invalid method name is provided.

    Raises:
        ValueError: Expected to be raised when an invalid method name is passed.
    """
    with pytest.raises(ValueError):
        run("invalid_method")


def test_run_with_tuner_method(mock_system, mock_trainer, mock_parse_config, mock_seed_everything, mock_trainer_class):
    """
    Tests that a Tuner method is correctly called with appropriate arguments.

    Args:
        mock_system: Mock LighterSystem instance
        mock_trainer: Mock Trainer instance
        mock_parse_config: Mock parse_config function
        mock_seed_everything: Mock seed_everything function
        mock_trainer_class: Mock Trainer class
    """
    with patch("lighter.engine.runner.Tuner") as mock_tuner_class:
        # Configure mock parse_config to return our test configuration
        mock_parse_config.return_value.get_parsed_content.side_effect = lambda x, default=None: {
            "project": None,
            "system": mock_system,
            "trainer": mock_trainer,
        }.get(x, default)

        mock_trainer_class.return_value = mock_trainer
        mock_tuner = mock_tuner_class.return_value

        run("scale_batch_size")

        # Verify that the scale_batch_size method was called with correct arguments
        mock_tuner.scale_batch_size.assert_called_once_with(mock_system, **{})


def test_run_with_project_import(mock_system, mock_trainer, mock_parse_config, mock_seed_everything, mock_trainer_class):
    """
    Tests that the project module is correctly imported when a project path is specified.

    Args:
        mock_system: Mock LighterSystem instance
        mock_trainer: Mock Trainer instance
        mock_parse_config: Mock parse_config function
        mock_seed_everything: Mock seed_everything function
        mock_trainer_class: Mock Trainer class
    """
    with patch("lighter.engine.runner.import_module_from_path") as mock_import_module:
        # Set up configuration with a project path
        mock_parse_config.return_value.get_parsed_content.side_effect = lambda x, default=None: {
            "project": "some_project_path",
            "system": mock_system,
            "trainer": mock_trainer,
        }.get(x, default)

        mock_trainer_class.return_value = mock_trainer

        run("fit")

        # Verify that the project module was imported correctly
        mock_import_module.assert_called_once_with("project", "some_project_path")


def test_run_with_invalid_system(mock_trainer, mock_parse_config, mock_seed_everything, mock_trainer_class):
    """
    Tests that run raises ValueError when an invalid system instance is provided.

    Args:
        mock_trainer: Mock Trainer instance
        mock_parse_config: Mock parse_config function
        mock_seed_everything: Mock seed_everything function
        mock_trainer_class: Mock Trainer class

    Raises:
        ValueError: Expected to be raised when system is not a LighterSystem instance.
    """
    mock_parse_config.return_value.get_parsed_content.side_effect = lambda x, default=None: {
        "project": None,
        "system": "not_a_system_instance",  # Invalid system instance
        "trainer": mock_trainer,
    }.get(x, default)

    mock_trainer_class.return_value = mock_trainer

    with pytest.raises(ValueError, match="Expected 'system' to be an instance of 'LighterSystem'"):
        run("fit")


def test_run_with_invalid_trainer(mock_system, mock_parse_config, mock_seed_everything):
    """
    Tests that run raises ValueError when an invalid trainer instance is provided.

    Args:
        mock_system: Mock LighterSystem instance
        mock_parse_config: Mock parse_config function
        mock_seed_everything: Mock seed_everything function

    Raises:
        ValueError: Expected to be raised when trainer is not a PyTorch Lightning Trainer instance.
    """
    mock_parse_config.return_value.get_parsed_content.side_effect = lambda x, default=None: {
        "project": None,
        "system": mock_system,
        "trainer": "not_a_trainer_instance",  # Invalid trainer instance
    }.get(x, default)

    with pytest.raises(ValueError, match="Expected 'trainer' to be an instance of PyTorch Lightning 'Trainer'"):
        run("fit")


def test_run_with_trainer_logger(mock_system, mock_trainer, mock_parse_config, mock_seed_everything, mock_trainer_class):
    """
    Tests that the trainer logger correctly logs hyperparameters.

    Args:
        mock_system: Mock LighterSystem instance
        mock_trainer: Mock Trainer instance
        mock_parse_config: Mock parse_config function
        mock_seed_everything: Mock seed_everything function
        mock_trainer_class: Mock Trainer class
    """
    # Set up mock logger
    mock_logger = MagicMock()
    mock_trainer.logger = mock_logger
    mock_parse_config.return_value.get_parsed_content.side_effect = lambda x, default=None: {
        "project": None,
        "system": mock_system,
        "trainer": mock_trainer,
    }.get(x, default)

    mock_trainer_class.return_value = mock_trainer

    run("fit")

    # Verify that hyperparameters were logged
    mock_logger.log_hyperparams.assert_called_once_with(mock_parse_config.return_value.config)


def test_run_trainer_fit_called(mock_system, mock_trainer, mock_parse_config, mock_seed_everything, mock_trainer_class):
    """
    Tests that the trainer's fit method is called with correct arguments.

    Args:
        mock_system: Mock LighterSystem instance
        mock_trainer: Mock Trainer instance
        mock_parse_config: Mock parse_config function
        mock_seed_everything: Mock seed_everything function
        mock_trainer_class: Mock Trainer class
    """
    # Configure mock parse_config to return our test configuration
    mock_parse_config.return_value.get_parsed_content.side_effect = lambda x, default=None: {
        "project": None,
        "system": mock_system,
        "trainer": mock_trainer,
    }.get(x, default)

    mock_trainer_class.return_value = mock_trainer

    run("fit")

    # Verify that the fit method was called with correct arguments
    mock_trainer.fit.assert_called_once_with(mock_system, **{})
