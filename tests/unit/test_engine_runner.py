"""Unit tests for the Runner class."""

from unittest.mock import MagicMock, patch

import pytest
from pytorch_lightning import Trainer

from lighter.engine.config import Config
from lighter.engine.resolver import Resolver
from lighter.engine.runner import Runner
from lighter.system import System
from lighter.utils.types.enums import Stage


@pytest.fixture
def mock_config():
    """Provides a mock Config object."""
    mock_dict = {
        "trainer": {"_target_": "pytorch_lightning.Trainer"},
        "system": {
            "_target_": "lighter.System",
            "model": {"_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1},
            "optimizer": {"_target_": "torch.optim.SGD", "params": "$@system#model.parameters()", "lr": 0.01},
            "criterion": {"_target_": "torch.nn.MSELoss"},
            "dataloaders": {
                "train": {
                    "_target_": "torch.utils.data.DataLoader",
                    "dataset": {
                        "_target_": "torch.utils.data.TensorDataset",
                        "tensors": ["$torch.randn(10, 1)", "$torch.randn(10, 1)"],
                    },
                }
            },
        },
        "args": {"fit": {"max_epochs": 1}},
    }
    return Config(mock_dict, validate=False)


@pytest.fixture
def mock_resolver(mock_config):
    """Provides a mock Resolver object."""
    resolver = Resolver(mock_config)
    resolver.get_stage_config = MagicMock(return_value=mock_config)  # Mock to return the full config for simplicity
    return resolver


@pytest.fixture
def mock_system():
    """Provides a mock System object."""
    system = MagicMock(spec=System)
    system.save_hyperparameters = MagicMock()
    return system


@pytest.fixture
def mock_trainer():
    """Provides a mock Trainer object."""
    trainer = MagicMock(spec=Trainer)
    trainer.logger = MagicMock()
    trainer.fit = MagicMock()
    trainer.validate = MagicMock()
    trainer.test = MagicMock()
    trainer.predict = MagicMock()
    return trainer


@patch("lighter.engine.runner.seed_everything")
@patch("lighter.engine.runner.Config")
@patch("lighter.engine.runner.Resolver")
@patch("lighter.engine.runner.import_module_from_path")
def test_runner_run_fit_stage(
    mock_import_module,
    MockResolver,
    MockConfig,
    mock_seed_everything,
    mock_system,
    mock_trainer,
    mock_config,
):
    """Test that the run method correctly orchestrates the FIT stage."""
    MockConfig.return_value = mock_config
    MockResolver.return_value.get_stage_config.return_value = mock_config
    with patch.object(
        mock_config,
        "get_parsed_content",
        side_effect=[
            mock_system,  # For system initialization
            mock_trainer,  # For trainer initialization
            {"max_epochs": 1},  # For args#fit
        ],
    ) as mock_get_parsed_content:
        runner = Runner()
        runner.run(Stage.FIT, config="path/to/config.yaml")

        mock_seed_everything.assert_called_once()
        MockConfig.assert_called_once_with("path/to/config.yaml", validate=True)
        MockResolver.assert_called_once_with(mock_config)
        MockResolver.return_value.get_stage_config.assert_called_once_with(Stage.FIT)
        mock_get_parsed_content.assert_any_call("system")
        mock_get_parsed_content.assert_any_call("trainer")
        mock_get_parsed_content.assert_any_call("args#fit", default={})
        mock_system.save_hyperparameters.assert_called_once()
        mock_trainer.logger.log_hyperparams.assert_called_once()
        mock_trainer.fit.assert_called_once_with(mock_system, max_epochs=1)


@patch("lighter.engine.runner.seed_everything")
@patch("lighter.engine.runner.Config")
@patch("lighter.engine.runner.Resolver")
@patch("lighter.engine.runner.import_module_from_path")
def test_runner_run_validate_stage(
    mock_import_module,
    MockResolver,
    MockConfig,
    mock_seed_everything,
    mock_system,
    mock_trainer,
    mock_config,
):
    """Test that the run method correctly orchestrates the VALIDATE stage."""
    MockConfig.return_value = mock_config
    MockResolver.return_value.get_stage_config.return_value = mock_config
    with patch.object(
        mock_config,
        "get_parsed_content",
        side_effect=[
            mock_system,  # For system initialization
            mock_trainer,  # For trainer initialization
            {"verbose": True},  # For args#validate
        ],
    ) as mock_get_parsed_content:
        runner = Runner()
        runner.run(Stage.VALIDATE, config="path/to/config.yaml")

        mock_trainer.validate.assert_called_once_with(mock_system, verbose=True)


@patch("lighter.engine.runner.seed_everything")
@patch("lighter.engine.runner.Config")
@patch("lighter.engine.runner.Resolver")
@patch("lighter.engine.runner.import_module_from_path")
def test_runner_run_test_stage(
    mock_import_module,
    MockResolver,
    MockConfig,
    mock_seed_everything,
    mock_system,
    mock_trainer,
    mock_config,
):
    """Test that the run method correctly orchestrates the TEST stage."""
    MockConfig.return_value = mock_config
    MockResolver.return_value.get_stage_config.return_value = mock_config
    with patch.object(
        mock_config,
        "get_parsed_content",
        side_effect=[
            mock_system,  # For system initialization
            mock_trainer,  # For trainer initialization
            {"ckpt_path": "best"},  # For args#test
        ],
    ) as mock_get_parsed_content:
        runner = Runner()
        runner.run(Stage.TEST, config="path/to/config.yaml")

        mock_trainer.test.assert_called_once_with(mock_system, ckpt_path="best")


@patch("lighter.engine.runner.seed_everything")
@patch("lighter.engine.runner.Config")
@patch("lighter.engine.runner.Resolver")
@patch("lighter.engine.runner.import_module_from_path")
def test_runner_run_predict_stage(
    mock_import_module,
    MockResolver,
    MockConfig,
    mock_seed_everything,
    mock_system,
    mock_trainer,
    mock_config,
):
    """Test that the run method correctly orchestrates the PREDICT stage."""
    MockConfig.return_value = mock_config
    MockResolver.return_value.get_stage_config.return_value = mock_config
    with patch.object(
        mock_config,
        "get_parsed_content",
        side_effect=[
            mock_system,  # For system initialization
            mock_trainer,  # For trainer initialization
            {"return_predictions": True},  # For args#predict
        ],
    ) as mock_get_parsed_content:
        runner = Runner()
        runner.run(Stage.PREDICT, config="path/to/config.yaml")

        mock_trainer.predict.assert_called_once_with(mock_system, return_predictions=True)


@patch("lighter.engine.runner.seed_everything")
@patch("lighter.engine.runner.Config")
@patch("lighter.engine.runner.Resolver")
@patch("lighter.engine.runner.import_module_from_path")
def test_runner_setup_stage_invalid_system_raises_error(
    mock_import_module,
    MockResolver,
    MockConfig,
    mock_seed_everything,
    mock_system,
    mock_trainer,
    mock_config,
):
    """Test that _setup_stage raises ValueError if system is not an instance of System."""
    MockConfig.return_value = mock_config
    MockResolver.return_value.get_stage_config.return_value = mock_config
    with patch.object(
        mock_config,
        "get_parsed_content",
        side_effect=[
            "not_a_system_instance",  # For system initialization
            mock_trainer,  # For trainer initialization
            {},  # For args
        ],
    ) as mock_get_parsed_content:
        runner = Runner()
        with pytest.raises(ValueError, match="'system' must be an instance of System"):
            runner.run(Stage.FIT, config="path/to/config.yaml")


@patch("lighter.engine.runner.seed_everything")
@patch("lighter.engine.runner.Config")
@patch("lighter.engine.runner.Resolver")
@patch("lighter.engine.runner.import_module_from_path")
def test_runner_setup_stage_invalid_trainer_raises_error(
    mock_import_module,
    MockResolver,
    MockConfig,
    mock_seed_everything,
    mock_system,
    mock_trainer,
    mock_config,
):
    """Test that _setup_stage raises ValueError if trainer is not an instance of Trainer."""
    MockConfig.return_value = mock_config
    MockResolver.return_value.get_stage_config.return_value = mock_config
    with patch.object(
        mock_config,
        "get_parsed_content",
        side_effect=[
            mock_system,  # For system initialization
            "not_a_trainer_instance",  # For trainer initialization
            {},  # For args
        ],
    ) as mock_get_parsed_content:
        runner = Runner()
        with pytest.raises(ValueError, match="'trainer' must be an instance of Trainer"):
            runner.run(Stage.FIT, config="path/to/config.yaml")


@patch("lighter.engine.runner.seed_everything")
@patch("lighter.engine.runner.Config")
@patch("lighter.engine.runner.Resolver")
@patch("lighter.engine.runner.import_module_from_path")
def test_runner_save_config(
    mock_import_module,
    MockResolver,
    MockConfig,
    mock_seed_everything,
    mock_system,
    mock_trainer,
    mock_config,
):
    """Test that _save_config correctly calls save_hyperparameters and log_hyperparams."""
    MockConfig.return_value = mock_config
    MockResolver.return_value.get_stage_config.return_value = mock_config
    with patch.object(
        mock_config,
        "get_parsed_content",
        side_effect=[
            mock_system,  # For system initialization
            mock_trainer,  # For trainer initialization
            {},  # For args
        ],
    ) as mock_get_parsed_content:
        runner = Runner()
        runner.run(Stage.FIT, config="path/to/config.yaml")

    mock_system.save_hyperparameters.assert_called_once_with(mock_get_parsed_content.call_args_list[3].return_value)
    mock_trainer.logger.log_hyperparams.assert_called_once_with(mock_get_parsed_content.call_args_list[3].return_value)


@patch("lighter.engine.runner.fire.Fire")
@patch("lighter.engine.runner.Runner.run")
def test_cli_calls_runner_methods(mock_runner_run, mock_fire_fire):
    """Test that the cli function correctly sets up fire with runner methods."""
    from lighter.engine.runner import cli

    cli()

    mock_fire_fire.assert_called_once()
    # Verify that the 'fit' command passed to fire.Fire calls runner.run with Stage.FIT
    fire_args = mock_fire_fire.call_args[0][0]
    assert "fit" in fire_args
    # Call the fit function that fire would call
    fire_args["fit"]("test_config.yaml", some_arg=1)
    mock_runner_run.assert_called_once_with(Stage.FIT, config="test_config.yaml", some_arg=1)

    mock_runner_run.reset_mock()
    fire_args["validate"]("test_config.yaml", some_arg=2)
    mock_runner_run.assert_called_once_with(Stage.VALIDATE, config="test_config.yaml", some_arg=2)

    mock_runner_run.reset_mock()
    fire_args["test"]("test_config.yaml", some_arg=3)
    mock_runner_run.assert_called_once_with(Stage.TEST, config="test_config.yaml", some_arg=3)

    mock_runner_run.reset_mock()
    fire_args["predict"]("test_config.yaml", some_arg=4)
    mock_runner_run.assert_called_once_with(Stage.PREDICT, config="test_config.yaml", some_arg=4)
