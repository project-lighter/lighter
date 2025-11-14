"""Unit tests for the Runner class in lighter/engine/runner.py"""

from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
from pytorch_lightning import Trainer
from sparkwheel import Config

from lighter.engine.runner import Runner
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
def base_config():
    """Fixture providing a base configuration dictionary."""
    return {
        "trainer": {
            "_target_": "pytorch_lightning.Trainer",
            "max_epochs": 10,
        },
        "system": {
            "_target_": "lighter.system.System",
            "model": {"_target_": "torch.nn.Identity"},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "dataloaders": {
                "train": {"batch_size": 32},
                "val": {"batch_size": 32},
                "test": {"batch_size": 32},
                "predict": {"batch_size": 32},
            },
            "metrics": {
                "train": [],
                "val": [],
                "test": [],
            },
        },
        "args": {
            "fit": {"some_arg": "value"},
            "validate": {},
            "test": {},
            "predict": {},
        },
    }


@pytest.fixture
def runner():
    """Fixture providing a Runner instance."""
    return Runner()


def test_runner_initialization(runner):
    """Test that Runner initializes with correct default values."""
    assert runner.config is None
    assert runner.system is None
    assert runner.trainer is None


def test_runner_applies_overrides(runner, base_config):
    """Test that CLI overrides are applied correctly."""
    overrides = ["trainer::max_epochs=100"]

    # Mock the setup and execute to avoid needing real models
    with patch.object(runner, "_setup") as mock_setup, patch.object(runner, "_execute") as mock_execute:
        runner.run(Stage.FIT, base_config, overrides)

        # Verify override was applied
        assert runner.config.get("trainer::max_epochs") == 100

        # Verify methods were called
        mock_setup.assert_called_once()
        mock_execute.assert_called_once()


def test_prune_removes_unused_modes(runner, base_config):
    """Test that pruning removes unused dataloaders/metrics for each stage."""
    # Load config - use deepcopy since Config.load modifies the dict
    runner.config = Config.load(deepcopy(base_config))

    # Test FIT stage pruning
    runner._prune_for_stage(Stage.FIT)
    system = runner.config.get("system", {})
    dataloaders = system.get("dataloaders", {})
    metrics = system.get("metrics", {})

    # FIT keeps train and val
    assert "train" in dataloaders
    assert "val" in dataloaders
    assert "test" not in dataloaders
    assert "predict" not in dataloaders
    assert "train" in metrics
    assert "val" in metrics
    assert "test" not in metrics

    # Reset config and test TEST stage - use fresh copy
    runner.config = Config.load(deepcopy(base_config))
    runner._prune_for_stage(Stage.TEST)
    system = runner.config.get("system", {})
    dataloaders = system.get("dataloaders", {})
    metrics = system.get("metrics", {})

    # TEST stage should only have test dataloader
    assert "test" in dataloaders
    assert "train" not in dataloaders
    assert "val" not in dataloaders
    assert "predict" not in dataloaders


def test_prune_removes_optimizer_for_non_fit(runner, base_config):
    """Test that optimizer/scheduler are removed for non-FIT stages."""
    runner.config = Config.load(deepcopy(base_config))

    # Test VALIDATE stage
    runner._prune_for_stage(Stage.VALIDATE)
    system = runner.config.get("system", {})

    # VALIDATE removes optimizer/scheduler but keeps criterion
    assert "optimizer" not in system
    assert "scheduler" not in system

    # Test TEST stage - use fresh copy
    runner.config = Config.load(deepcopy(base_config))
    runner._prune_for_stage(Stage.TEST)
    system = runner.config.get("system", {})

    # TEST removes everything
    assert "optimizer" not in system
    assert "scheduler" not in system


def test_setup_with_invalid_system(runner, mock_trainer):
    """Test that _setup raises error for invalid system type."""
    # Create config that resolves to non-System object (just a dict)
    bad_config = {
        "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
        "system": {"_target_": "builtins.dict"},  # This will resolve to dict type, not System
    }

    runner.config = Config.load(bad_config)

    # Mock trainer resolution to avoid needing real trainer
    with patch.object(runner.config, "resolve") as mock_resolve:
        mock_resolve.side_effect = [dict(), mock_trainer]  # First call returns dict instead of System

        with pytest.raises(TypeError, match="system must be System"):
            runner._setup(Stage.FIT)


def test_setup_with_invalid_trainer(runner, mock_system):
    """Test that _setup raises error for invalid trainer type."""
    # Create simple config
    bad_config = {
        "trainer": {"_target_": "builtins.dict"},  # This will resolve to dict type
        "system": {"_target_": "lighter.system.System"},
    }

    runner.config = Config.load(bad_config)

    # Mock system and trainer resolution
    with patch.object(runner.config, "resolve") as mock_resolve:
        mock_resolve.side_effect = [mock_system, dict()]  # Second call returns dict instead of Trainer

        with pytest.raises(TypeError, match="trainer must be Trainer"):
            runner._setup(Stage.FIT)


@patch("lighter.engine.runner.import_module_from_path")
def test_setup_with_project(mock_import, runner, base_config, mock_system, mock_trainer):
    """Test that _setup correctly imports project module."""
    config_with_project = base_config.copy()
    config_with_project["project"] = "path/to/project"

    runner.config = Config.load(config_with_project)

    # Mock resolve to return our mocks
    with patch.object(runner.config, "resolve") as mock_resolve:
        mock_resolve.side_effect = [mock_system, mock_trainer]

        runner._setup(Stage.FIT)
        mock_import.assert_called_once_with("project", "path/to/project")


def test_execute_calls_stage_method(runner, mock_system, mock_trainer):
    """Test that _execute calls the correct trainer method."""
    runner.config = MagicMock()
    runner.config.resolve.return_value = {"some_arg": "value"}
    runner.system = mock_system
    runner.trainer = mock_trainer

    # Test fit
    runner._execute(Stage.FIT)
    mock_trainer.fit.assert_called_once_with(mock_system, some_arg="value")
    mock_trainer.reset_mock()

    # Test validate
    runner._execute(Stage.VALIDATE)
    mock_trainer.validate.assert_called_once_with(mock_system, some_arg="value")
    mock_trainer.reset_mock()

    # Test test
    runner._execute(Stage.TEST)
    mock_trainer.test.assert_called_once_with(mock_system, some_arg="value")
    mock_trainer.reset_mock()

    # Test predict
    runner._execute(Stage.PREDICT)
    mock_trainer.predict.assert_called_once_with(mock_system, some_arg="value")
