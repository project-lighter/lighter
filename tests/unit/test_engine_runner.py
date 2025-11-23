"""Unit tests for the Runner class in lighter/engine/runner.py"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from pytorch_lightning import Trainer
from sparkwheel import Config

from lighter.engine.runner import Runner
from lighter.model import LighterModule
from lighter.utils.types.enums import Stage


@pytest.fixture
def mock_model():
    """Fixture providing a mock LightningModule instance."""
    model = MagicMock(spec=LighterModule)
    model.save_hyperparameters = MagicMock()
    return model


@pytest.fixture
def mock_datamodule():
    """Fixture providing a mock LightningDataModule instance."""
    from pytorch_lightning import LightningDataModule

    datamodule = MagicMock(spec=LightningDataModule)
    return datamodule


@pytest.fixture
def mock_runner_trainer():
    """Fixture providing a mock Trainer instance for runner tests.

    Note: Named differently from the shared mock_trainer to avoid conflicts
    since this has runner-specific configuration.
    """
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
        "model": {
            "_target_": "lighter.LighterModule",
            "model": {"_target_": "torch.nn.Identity"},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
        },
        "data": {
            "_target_": "lighter.LighterDataModule",
            "train_dataloader": {"batch_size": 32},
            "val_dataloader": {"batch_size": 32},
            "test_dataloader": {"batch_size": 32},
            "predict_dataloader": {"batch_size": 32},
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
    """Test that Runner initializes correctly."""
    # Runner no longer stores state, just verify it exists
    assert runner is not None


def test_runner_applies_overrides(runner, base_config):
    """Test that CLI overrides are applied correctly."""
    # Combine config and overrides into single inputs list
    inputs = [base_config, "trainer::max_epochs=100"]

    # Mock the resolve and execute to avoid needing real models
    with (
        patch.object(runner, "_resolve_model") as mock_resolve_model,
        patch.object(runner, "_resolve_trainer") as mock_resolve_trainer,
        patch.object(runner, "_resolve_datamodule") as mock_resolve_datamodule,
        patch.object(runner, "_execute") as mock_execute,
    ):
        mock_model = MagicMock()
        mock_model.save_hyperparameters = MagicMock()
        mock_resolve_model.return_value = mock_model

        mock_trainer = MagicMock()
        mock_trainer.logger = None
        mock_resolve_trainer.return_value = mock_trainer

        mock_datamodule = MagicMock()
        mock_resolve_datamodule.return_value = mock_datamodule

        runner.run(Stage.FIT, inputs)

        # Verify methods were called
        mock_resolve_model.assert_called_once()
        mock_resolve_trainer.assert_called_once()
        mock_resolve_datamodule.assert_called_once()
        mock_execute.assert_called_once()


def test_setup_with_invalid_model(runner, mock_trainer):
    """Test that _resolve_model raises error for invalid model type."""
    # Create config that resolves to non-LightningModule object (just a dict)
    bad_config = {
        "trainer": {"_target_": "pytorch_lightning.Trainer", "max_epochs": 10},
        "model": {"_target_": "builtins.dict"},  # This will resolve to dict type, not LightningModule
    }

    config = Config().update(bad_config)

    # Mock resolve to return dict instead of LightningModule
    with patch.object(config, "resolve") as mock_resolve:
        mock_resolve.return_value = dict()  # Returns dict instead of LightningModule

        with pytest.raises(TypeError, match="model must be LightningModule or LighterModule"):
            runner._resolve_model(config)


def test_setup_with_invalid_trainer(runner, mock_model):
    """Test that _resolve_trainer raises error for invalid trainer type."""
    # Create simple config
    bad_config = {
        "trainer": {"_target_": "builtins.dict"},  # This will resolve to dict type
        "model": {"_target_": "lighter.LighterModule"},
    }

    config = Config().update(bad_config)

    # Mock resolve to return dict instead of Trainer
    with patch.object(config, "resolve") as mock_resolve:
        mock_resolve.return_value = dict()  # Returns dict instead of Trainer

        with pytest.raises(TypeError, match="trainer must be Trainer"):
            runner._resolve_trainer(config)


def test_execute_calls_stage_method(runner, mock_model, mock_trainer, mock_datamodule):
    """Test that _execute calls the correct trainer method."""
    config = MagicMock()

    # Test fit
    runner._execute(Stage.FIT, config, mock_model, mock_trainer, mock_datamodule)
    mock_trainer.fit.assert_called_once_with(mock_model, datamodule=mock_datamodule)
    mock_trainer.reset_mock()

    # Test validate
    runner._execute(Stage.VALIDATE, config, mock_model, mock_trainer, mock_datamodule)
    mock_trainer.validate.assert_called_once_with(mock_model, datamodule=mock_datamodule)
    mock_trainer.reset_mock()

    # Test test
    runner._execute(Stage.TEST, config, mock_model, mock_trainer, mock_datamodule)
    mock_trainer.test.assert_called_once_with(mock_model, datamodule=mock_datamodule)
    mock_trainer.reset_mock()

    # Test predict
    runner._execute(Stage.PREDICT, config, mock_model, mock_trainer, mock_datamodule)
    mock_trainer.predict.assert_called_once_with(mock_model, datamodule=mock_datamodule)


# Tests for auto-discovery feature


def test_auto_discover_project_with_marker(runner, tmp_path, monkeypatch):
    """Test that ProjectImporter finds __lighter__.py marker file."""
    from lighter.engine.runner import ProjectImporter

    # Create a project directory with marker file and __init__.py
    project_dir = tmp_path / "my_project"
    project_dir.mkdir()
    marker_file = project_dir / "__lighter__.py"
    marker_file.touch()
    init_file = project_dir / "__init__.py"
    init_file.touch()

    # Change working directory to project
    monkeypatch.chdir(project_dir)

    # Test auto-discovery
    importer = ProjectImporter()
    found = importer.auto_discover_and_import()

    assert found is True


def test_auto_discover_project_without_marker(runner, tmp_path, monkeypatch):
    """Test that ProjectImporter returns False when marker is absent."""
    from lighter.engine.runner import ProjectImporter

    # Create a directory without marker file
    project_dir = tmp_path / "not_a_project"
    project_dir.mkdir()

    # Change working directory to directory without marker
    monkeypatch.chdir(project_dir)

    # Test auto-discovery
    importer = ProjectImporter()
    found = importer.auto_discover_and_import()

    assert found is False


@patch("lighter.engine.runner.import_module_from_path")
def test_setup_with_auto_discovery(mock_import, runner, base_config, mock_model, mock_trainer, tmp_path, monkeypatch):
    """Test that ProjectImporter auto-discovers and imports project."""
    from lighter.engine.runner import ProjectImporter

    # Create project directory with marker
    project_dir = tmp_path / "auto_project"
    project_dir.mkdir()
    (project_dir / "__lighter__.py").touch()
    monkeypatch.chdir(project_dir)

    # Test auto-discovery and import
    result = ProjectImporter.auto_discover_and_import()

    # Verify auto-discovered project was imported as 'project'
    assert result is True
    mock_import.assert_called_once_with("project", project_dir)


@patch("lighter.engine.runner.import_module_from_path")
def test_setup_without_project_or_discovery(mock_import, runner, base_config, mock_model, mock_trainer, tmp_path, monkeypatch):
    """Test that ProjectImporter works without project module (plain Lightning)."""
    from lighter.engine.runner import ProjectImporter

    # Create directory without marker
    no_project_dir = tmp_path / "plain_lightning"
    no_project_dir.mkdir()
    monkeypatch.chdir(no_project_dir)

    # Test auto-discovery without project
    result = ProjectImporter.auto_discover_and_import()

    # Verify no project module was imported
    assert result is False
    mock_import.assert_not_called()


# Tests for output directory management


@patch("lighter.engine.runner.datetime")
def test_create_output_dir(mock_datetime, runner, tmp_path, monkeypatch):
    """Test that output directory is created with correct timestamp structure."""
    from lighter.engine.runner import OutputDir

    # Mock datetime
    mock_now = MagicMock()
    mock_now.strftime.side_effect = lambda fmt: {"%Y-%m-%d": "2025-11-21", "%H-%M-%S": "14-30-45"}.get(fmt, "")
    mock_datetime.now.return_value = mock_now

    # Change to temp directory
    monkeypatch.chdir(tmp_path)

    # Create output dir
    output_dir = OutputDir.create_timestamped()

    # Verify structure
    assert output_dir.path == Path("outputs/2025-11-21/14-30-45")
    assert output_dir.path.exists()
    assert output_dir.path.is_dir()


def test_configure_output_dir_sets_default_root_dir(runner, base_config):
    """Test that ConfigLoader sets default_root_dir when not specified."""
    from lighter.engine.runner import ConfigLoader, OutputDir

    config = Config().update(base_config)
    output_dir = OutputDir(Path("outputs/2025-11-21/14-30-45"))

    # Verify default_root_dir is not set initially
    assert config.get("trainer::default_root_dir") is None

    # Configure output dir
    ConfigLoader.set_default_root_dir(config, output_dir)

    # Verify default_root_dir was set
    assert config.get("trainer::default_root_dir") == "outputs/2025-11-21/14-30-45"


def test_configure_output_dir_respects_user_override(runner, base_config):
    """Test that ConfigLoader doesn't override user-specified default_root_dir."""
    from lighter.engine.runner import ConfigLoader, OutputDir

    config_with_root = base_config.copy()
    config_with_root["trainer"]["default_root_dir"] = "/custom/path"

    config = Config().update(config_with_root)
    output_dir = OutputDir(Path("outputs/2025-11-21/14-30-45"))

    # Configure output dir
    ConfigLoader.set_default_root_dir(config, output_dir)

    # Verify user's default_root_dir was preserved
    assert config.get("trainer::default_root_dir") == "/custom/path"


@patch("builtins.open", new_callable=mock_open)
@patch("lighter.engine.runner.yaml.dump")
def test_save_config_to_output(mock_yaml_dump, mock_file, runner, base_config, tmp_path):
    """Test that config is saved to output directory."""
    from lighter.engine.runner import OutputDir

    config = Config().update(base_config)
    output_dir_path = tmp_path / "outputs" / "2025-11-21" / "14-30-45"
    output_dir_path.mkdir(parents=True)
    output_dir = OutputDir(output_dir_path)

    # Save config
    output_dir.save_config(config)

    # Verify file was opened at correct path
    expected_path = output_dir_path / "config.yaml"
    mock_file.assert_called_once_with(expected_path, "w")

    # Verify yaml.dump was called with config
    mock_yaml_dump.assert_called_once()
    call_args = mock_yaml_dump.call_args
    assert call_args[0][0] == config.get()  # First positional arg
    assert call_args[1]["default_flow_style"] is False
    assert call_args[1]["sort_keys"] is False


@patch("lighter.engine.runner.OutputDir.create_timestamped")
@patch("lighter.engine.runner.ConfigLoader.set_default_root_dir")
@patch("lighter.engine.runner.ProjectImporter.auto_discover_and_import")
def test_run_creates_output_dir_before_setup(
    mock_import, mock_set_root, mock_create, runner, base_config, mock_model, mock_trainer, mock_datamodule
):
    """Test that run() creates and configures output directory."""
    from lighter.engine.runner import OutputDir

    mock_output_dir = OutputDir(Path("outputs/2025-11-21/14-30-45"))
    mock_create.return_value = mock_output_dir
    mock_import.return_value = False

    # Mock resolve and execute to avoid needing real models
    with (
        patch.object(runner, "_resolve_model") as mock_resolve_model,
        patch.object(runner, "_resolve_trainer") as mock_resolve_trainer,
        patch.object(runner, "_resolve_datamodule") as mock_resolve_datamodule,
        patch.object(runner, "_execute") as mock_execute,
        patch.object(mock_output_dir, "save_config") as mock_save,
    ):
        mock_resolve_model.return_value = mock_model
        mock_resolve_trainer.return_value = mock_trainer
        mock_resolve_datamodule.return_value = mock_datamodule

        # Run
        runner.run(Stage.FIT, [base_config])

        # Verify order of calls
        assert mock_create.called
        assert mock_set_root.called
        assert mock_save.called
        assert mock_resolve_model.called
        assert mock_resolve_trainer.called
        assert mock_resolve_datamodule.called
        assert mock_execute.called


def test_runner_initialization_includes_output_dir(runner):
    """Test that Runner initializes correctly."""
    # Runner no longer stores state, just verify it exists
    assert runner is not None


# Tests for CLI kwargs handling


def test_execute_with_cli_kwargs(runner, mock_model, mock_trainer, mock_datamodule):
    """Test that CLI kwargs are passed to trainer method."""
    config = MagicMock()

    # Test with CLI kwargs
    runner._execute(Stage.FIT, config, mock_model, mock_trainer, mock_datamodule, ckpt_path="checkpoint.ckpt")

    mock_trainer.fit.assert_called_once_with(mock_model, datamodule=mock_datamodule, ckpt_path="checkpoint.ckpt")


def test_execute_without_cli_kwargs(runner, mock_model, mock_trainer, mock_datamodule):
    """Test that execute works without CLI kwargs."""
    config = MagicMock()

    # Test with no CLI kwargs
    runner._execute(Stage.FIT, config, mock_model, mock_trainer, mock_datamodule)

    mock_trainer.fit.assert_called_once_with(mock_model, datamodule=mock_datamodule)


def test_run_passes_kwargs_to_execute(runner, base_config, mock_model, mock_trainer, mock_datamodule):
    """Test that Runner.run() passes CLI kwargs to _execute()."""
    # Mock resolve and execute to track calls
    with (
        patch.object(runner, "_resolve_model") as mock_resolve_model,
        patch.object(runner, "_resolve_trainer") as mock_resolve_trainer,
        patch.object(runner, "_resolve_datamodule") as mock_resolve_datamodule,
        patch.object(runner, "_execute") as mock_execute,
    ):
        mock_resolve_model.return_value = mock_model
        mock_resolve_trainer.return_value = mock_trainer
        mock_resolve_datamodule.return_value = mock_datamodule

        # Run with CLI kwargs
        runner.run(Stage.FIT, [base_config], ckpt_path="checkpoint.ckpt", verbose=True)

        # Verify _execute was called with kwargs
        mock_execute.assert_called_once()
        call_kwargs = mock_execute.call_args[1]
        assert "ckpt_path" in call_kwargs
        assert call_kwargs["ckpt_path"] == "checkpoint.ckpt"
        assert "verbose" in call_kwargs
        assert call_kwargs["verbose"] is True
