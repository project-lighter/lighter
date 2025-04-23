"""Unit tests for the Resolver class in lighter/engine/resolver.py"""

import pytest

from lighter.engine.config import Config
from lighter.engine.resolver import Resolver
from lighter.utils.types.enums import Stage


@pytest.fixture
def full_config():
    """Fixture providing a complete configuration for testing."""
    return {
        "system": {
            "model": "test_model",
            "optimizer": {"name": "Adam", "lr": 0.001},
            "scheduler": {"name": "StepLR", "step_size": 10},
            "criterion": "CrossEntropyLoss",
            "dataloaders": {
                "train": {"batch_size": 32},
                "val": {"batch_size": 64},
                "test": {"batch_size": 64},
                "predict": {"batch_size": 128},
            },
            "metrics": {
                "train": {"accuracy": "Accuracy"},
                "val": {"accuracy": "Accuracy"},
                "test": {"accuracy": "Accuracy"},
            },
        },
        "args": {
            "fit": {"max_epochs": 10},
            "validate": {"verbose": True},
            "test": {"verbose": True},
            "predict": {"verbose": False},
        },
    }


@pytest.fixture
def resolver(full_config):
    """Fixture providing a Resolver instance."""
    config = Config(full_config, validate=False)
    return Resolver(config)


def test_resolver_initialization(resolver):
    """Test that Resolver initializes correctly."""
    assert isinstance(resolver.config, Config)


def test_invalid_stage(resolver):
    """Test that invalid stage raises ValueError."""
    with pytest.raises(ValueError, match="Invalid stage"):
        resolver.get_stage_config("invalid_stage")


def test_fit_stage_config(resolver):
    """Test configuration for fit stage."""
    config = resolver.get_stage_config(Stage.FIT)
    system_config = config.get("system")

    # Should keep train and val dataloaders
    assert "train" in system_config["dataloaders"]
    assert "val" in system_config["dataloaders"]
    assert "test" not in system_config["dataloaders"]
    assert "predict" not in system_config["dataloaders"]

    # Should keep train and val metrics
    assert "train" in system_config["metrics"]
    assert "val" in system_config["metrics"]
    assert "test" not in system_config["metrics"]

    # Should keep optimizer, scheduler, and criterion
    assert "optimizer" in system_config
    assert "scheduler" in system_config
    assert "criterion" in system_config

    # Should keep only fit args
    assert list(config.get("args").keys()) == ["fit"]


def test_validate_stage_config(resolver):
    """Test configuration for validate stage."""
    config = resolver.get_stage_config(Stage.VALIDATE)
    system_config = config.get("system")

    # Should keep only val dataloader
    assert "val" in system_config["dataloaders"]
    assert "train" not in system_config["dataloaders"]
    assert "test" not in system_config["dataloaders"]
    assert "predict" not in system_config["dataloaders"]

    # Should keep only val metrics
    assert "val" in system_config["metrics"]
    assert "train" not in system_config["metrics"]
    assert "test" not in system_config["metrics"]

    # Should remove optimizer and scheduler but keep criterion
    assert "optimizer" not in system_config
    assert "scheduler" not in system_config
    assert "criterion" in system_config

    # Should keep only validate args
    assert list(config.get("args").keys()) == ["validate"]


def test_test_stage_config(resolver):
    """Test configuration for test stage."""
    config = resolver.get_stage_config(Stage.TEST)
    system_config = config.get("system")

    # Should keep only test dataloader
    assert "test" in system_config["dataloaders"]
    assert "train" not in system_config["dataloaders"]
    assert "val" not in system_config["dataloaders"]
    assert "predict" not in system_config["dataloaders"]

    # Should keep only test metrics
    assert "test" in system_config["metrics"]
    assert "train" not in system_config["metrics"]
    assert "val" not in system_config["metrics"]

    # Should remove optimizer, scheduler, and criterion
    assert "optimizer" not in system_config
    assert "scheduler" not in system_config
    assert "criterion" not in system_config

    # Should keep only test args
    assert list(config.get("args").keys()) == ["test"]


def test_predict_stage_config(resolver):
    """Test configuration for predict stage."""
    config = resolver.get_stage_config(Stage.PREDICT)
    system_config = config.get("system")

    # Should keep only predict dataloader
    assert "predict" in system_config["dataloaders"]
    assert "train" not in system_config["dataloaders"]
    assert "val" not in system_config["dataloaders"]
    assert "test" not in system_config["dataloaders"]

    # Should remove all metrics
    assert "metrics" in system_config
    assert not system_config["metrics"]

    # Should remove optimizer, scheduler, and criterion
    assert "optimizer" not in system_config
    assert "scheduler" not in system_config
    assert "criterion" not in system_config

    # Should keep only predict args
    assert list(config.get("args").keys()) == ["predict"]
