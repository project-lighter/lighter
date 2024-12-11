from unittest.mock import MagicMock, patch

import pytest
from pytorch_lightning import Trainer

from lighter.engine.runner import cli, parse_config, run
from lighter.system import LighterSystem


def test_parse_config_no_config():
    with pytest.raises(ValueError):
        parse_config()


def test_cli():
    with patch("lighter.engine.runner.fire.Fire") as mock_fire:
        cli()
        mock_fire.assert_called_once()


def test_run_invalid_method():
    with pytest.raises(ValueError):
        run("invalid_method")


def test_run_with_tuner_method():
    """
    Test that a Tuner method is correctly called.
    """
    mock_system = MagicMock(spec=LighterSystem)
    mock_trainer = MagicMock(spec=Trainer)
    with patch("lighter.engine.runner.parse_config") as mock_parse_config, patch(
        "lighter.engine.runner.seed_everything"
    ), patch("lighter.engine.runner.Trainer", new=Trainer) as mock_trainer_class, patch(
        "lighter.engine.runner.Tuner"
    ) as mock_tuner_class:
        mock_parse_config.return_value.get_parsed_content.side_effect = lambda x, default=None: {
            "project": None,
            "system": mock_system,
            "trainer": mock_trainer,
        }.get(x, default)

        mock_trainer_class.return_value = mock_trainer
        mock_tuner = mock_tuner_class.return_value

        run("scale_batch_size")

        mock_tuner.scale_batch_size.assert_called_once_with(mock_system, **{})


def test_run_with_project_import():
    """
    Test that the project is imported when specified.
    """
    mock_system = MagicMock(spec=LighterSystem)
    mock_trainer = MagicMock(spec=Trainer)
    with patch("lighter.engine.runner.parse_config") as mock_parse_config, patch(
        "lighter.engine.runner.seed_everything"
    ), patch("lighter.engine.runner.import_module_from_path") as mock_import_module, patch(
        "lighter.engine.runner.Trainer", new=Trainer
    ) as mock_trainer_class:
        mock_parse_config.return_value.get_parsed_content.side_effect = lambda x, default=None: {
            "project": "some_project_path",
            "system": mock_system,
            "trainer": mock_trainer,
        }.get(x, default)

        mock_trainer_class.return_value = mock_trainer

        run("fit")

        mock_import_module.assert_called_once_with("project", "some_project_path")


def test_run_with_invalid_system():
    mock_trainer = MagicMock(spec=Trainer)
    with patch("lighter.engine.runner.parse_config") as mock_parse_config, patch(
        "lighter.engine.runner.seed_everything"
    ), patch("lighter.engine.runner.Trainer", new=Trainer) as mock_trainer_class:
        mock_parse_config.return_value.get_parsed_content.side_effect = lambda x, default=None: {
            "project": None,
            "system": "not_a_system_instance",
            "trainer": mock_trainer,
        }.get(x, default)

        mock_trainer_class.return_value = mock_trainer

        with pytest.raises(ValueError, match="Expected 'system' to be an instance of 'LighterSystem'"):
            run("fit")


def test_run_with_invalid_trainer():
    mock_system = MagicMock(spec=LighterSystem)
    with patch("lighter.engine.runner.parse_config") as mock_parse_config, patch("lighter.engine.runner.seed_everything"):
        mock_parse_config.return_value.get_parsed_content.side_effect = lambda x, default=None: {
            "project": None,
            "system": mock_system,
            "trainer": "not_a_trainer_instance",
        }.get(x, default)

        with pytest.raises(ValueError, match="Expected 'trainer' to be an instance of PyTorch Lightning 'Trainer'"):
            run("fit")


def test_run_with_trainer_logger():
    mock_system = MagicMock(spec=LighterSystem)
    mock_trainer = MagicMock(spec=Trainer)
    mock_logger = MagicMock()
    mock_trainer.logger = mock_logger
    with patch("lighter.engine.runner.parse_config") as mock_parse_config, patch(
        "lighter.engine.runner.seed_everything"
    ), patch("lighter.engine.runner.Trainer", new=Trainer) as mock_trainer_class:
        mock_parse_config.return_value.get_parsed_content.side_effect = lambda x, default=None: {
            "project": None,
            "system": mock_system,
            "trainer": mock_trainer,
        }.get(x, default)

        mock_trainer_class.return_value = mock_trainer

        run("fit")

        mock_logger.log_hyperparams.assert_called_once_with(mock_parse_config.return_value.config)


def test_run_trainer_fit_called():
    """
    Test that the trainer's fit method is called with the correct arguments.
    """
    mock_system = MagicMock(spec=LighterSystem)
    mock_trainer = MagicMock(spec=Trainer)
    with patch("lighter.engine.runner.parse_config") as mock_parse_config, patch(
        "lighter.engine.runner.seed_everything"
    ), patch("lighter.engine.runner.Trainer", new=Trainer) as mock_trainer_class:
        mock_parse_config.return_value.get_parsed_content.side_effect = lambda x, default=None: {
            "project": None,
            "system": mock_system,
            "trainer": mock_trainer,
        }.get(x, default)

        mock_trainer_class.return_value = mock_trainer

        run("fit")

        mock_trainer.fit.assert_called_once_with(mock_system, **{})
