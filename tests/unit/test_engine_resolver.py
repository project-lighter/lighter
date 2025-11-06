"""Unit tests for the Resolver class."""

import copy

import pytest
import yaml

from lighter.engine.config import Config
from lighter.engine.resolver import Resolver
from lighter.utils.types.enums import Mode, Stage


@pytest.fixture
def full_config_dict():
    """Provides a full configuration dictionary with all stages and components."""
    return {
        "trainer": {"_target_": "pytorch_lightning.Trainer"},
        "system": {
            "_target_": "lighter.System",
            "model": {"_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1},
            "optimizer": {"_target_": "torch.optim.SGD", "params": "$@system#model.parameters()", "lr": 0.01},
            "scheduler": {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 1},
            "criterion": {"_target_": "torch.nn.MSELoss"},
            "metrics": {
                "train": {"_target_": "torchmetrics.Accuracy"},
                "val": {"_target_": "torchmetrics.Accuracy"},
                "test": {"_target_": "torchmetrics.Accuracy"},
            },
            "dataloaders": {
                "train": {
                    "_target_": "torch.utils.data.DataLoader",
                    "dataset": {
                        "_target_": "torch.utils.data.TensorDataset",
                        "tensors": ["$torch.randn(10, 1)", "$torch.randn(10, 1)"],
                    },
                },
                "val": {
                    "_target_": "torch.utils.data.DataLoader",
                    "dataset": {
                        "_target_": "torch.utils.data.TensorDataset",
                        "tensors": ["$torch.randn(10, 1)", "$torch.randn(10, 1)"],
                    },
                },
                "test": {
                    "_target_": "torch.utils.data.DataLoader",
                    "dataset": {
                        "_target_": "torch.utils.data.TensorDataset",
                        "tensors": ["$torch.randn(10, 1)", "$torch.randn(10, 1)"],
                    },
                },
                "predict": {
                    "_target_": "torch.utils.data.DataLoader",
                    "dataset": {"_target_": "torch.utils.data.TensorDataset", "tensors": ["$torch.randn(10, 1)"]},
                },
            },
        },
        "args": {
            "fit": {"max_epochs": 10},
            "validate": {"verbose": True},
            "test": {"ckpt_path": "best"},
            "predict": {"return_predictions": True},
        },
    }


def test_get_stage_config_fit(full_config_dict):
    """Test get_stage_config for the FIT stage."""
    config = Config(full_config_dict, validate=False)
    resolver = Resolver(config)
    stage_config = resolver.get_stage_config(Stage.FIT)

    # Expected components for FIT
    system_config = stage_config.get()["system"]
    assert "train" in system_config["dataloaders"]
    assert "val" in system_config["dataloaders"]
    assert "test" not in system_config["dataloaders"]
    assert "predict" not in system_config["dataloaders"]
    assert "train" in system_config["metrics"]
    assert "val" in system_config["metrics"]
    assert "test" not in system_config["metrics"]
    assert "optimizer" in system_config
    assert "scheduler" in system_config
    assert "criterion" in system_config
    assert stage_config.get()["args"] == {"fit": {"max_epochs": 10}}


def test_get_stage_config_validate(full_config_dict):
    """Test get_stage_config for the VALIDATE stage."""
    config = Config(full_config_dict, validate=False)
    resolver = Resolver(config)
    stage_config = resolver.get_stage_config(Stage.VALIDATE)

    # Expected components for VALIDATE
    system_config = stage_config.get()["system"]
    assert "train" not in system_config["dataloaders"]
    assert "val" in system_config["dataloaders"]
    assert "test" not in system_config["dataloaders"]
    assert "predict" not in system_config["dataloaders"]
    assert "train" not in system_config["metrics"]
    assert "val" in system_config["metrics"]
    assert "test" not in system_config["metrics"]
    assert "optimizer" not in system_config
    assert "scheduler" not in system_config
    assert "criterion" in system_config  # Criterion should be present for validation loss
    assert stage_config.get()["args"] == {"validate": {"verbose": True}}


def test_get_stage_config_test(full_config_dict):
    """Test get_stage_config for the TEST stage."""
    config = Config(full_config_dict, validate=False)
    resolver = Resolver(config)
    stage_config = resolver.get_stage_config(Stage.TEST)

    # Expected components for TEST
    system_config = stage_config.get()["system"]
    assert "train" not in system_config["dataloaders"]
    assert "val" not in system_config["dataloaders"]
    assert "test" in system_config["dataloaders"]
    assert "predict" not in system_config["dataloaders"]
    assert "train" not in system_config["metrics"]
    assert "val" not in system_config["metrics"]
    assert "test" in system_config["metrics"]
    assert "optimizer" not in system_config
    assert "scheduler" not in system_config
    assert "criterion" not in system_config
    assert stage_config.get()["args"] == {"test": {"ckpt_path": "best"}}


def test_get_stage_config_predict(full_config_dict):
    """Test get_stage_config for the PREDICT stage."""
    config = Config(full_config_dict, validate=False)
    resolver = Resolver(config)
    stage_config = resolver.get_stage_config(Stage.PREDICT)

    # Expected components for PREDICT
    system_config = stage_config.get()["system"]
    assert "train" not in system_config["dataloaders"]
    assert "val" not in system_config["dataloaders"]
    assert "test" not in system_config["dataloaders"]
    assert "predict" in system_config["dataloaders"]
    assert "train" not in system_config["metrics"]
    assert "val" not in system_config["metrics"]
    assert "test" not in system_config["metrics"]
    assert "optimizer" not in system_config
    assert "scheduler" not in system_config
    assert "criterion" not in system_config
    assert stage_config.get()["args"] == {"predict": {"return_predictions": True}}


def test_get_stage_config_empty_dataloaders(full_config_dict):
    """Test get_stage_config when dataloaders are empty for a stage."""
    config_copy = copy.deepcopy(full_config_dict)
    config_copy["system"]["dataloaders"] = {}
    config = Config(config_copy, validate=False)
    resolver = Resolver(config)

    stage_config = resolver.get_stage_config(Stage.FIT)
    assert "dataloaders" not in stage_config.get()["system"]


def test_get_stage_config_missing_metrics(full_config_dict):
    """Test get_stage_config when metrics are missing for a stage."""
    config_copy = copy.deepcopy(full_config_dict)
    config_copy["system"]["metrics"] = {}
    config = Config(config_copy, validate=False)
    resolver = Resolver(config)

    stage_config = resolver.get_stage_config(Stage.FIT)
    assert "metrics" not in stage_config.get()["system"]


def test_get_stage_config_invalid_stage_raises_error(full_config_dict):
    """Test that an invalid stage raises a ValueError."""
    config = Config(full_config_dict, validate=False)
    resolver = Resolver(config)
    with pytest.raises(ValueError, match="Invalid stage"):
        resolver.get_stage_config("invalid_stage")


def test_get_stage_config_no_system_section():
    """Test get_stage_config when there is no 'system' section in the config."""
    config_dict = {"trainer": {"_target_": "pytorch_lightning.Trainer"}}
    config = Config(config_dict, validate=False)
    resolver = Resolver(config)
    stage_config = resolver.get_stage_config(Stage.FIT)
    assert "system" not in stage_config.get()


def test_get_stage_config_no_dataloaders_section():
    """Test get_stage_config when there is no 'dataloaders' section in the config."""
    config_dict = {
        "trainer": {"_target_": "pytorch_lightning.Trainer"},
        "system": {
            "_target_": "lighter.System",
            "model": {"_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1},
            "optimizer": {"_target_": "torch.optim.SGD", "params": "$@system#model.parameters()", "lr": 0.01},
            "criterion": {"_target_": "torch.nn.MSELoss"},
        },
    }
    config = Config(config_dict, validate=False)
    resolver = Resolver(config)
    stage_config = resolver.get_stage_config(Stage.FIT)
    assert "dataloaders" not in stage_config.get()["system"]


def test_get_stage_config_no_metrics_section():
    """Test get_stage_config when there is no 'metrics' section in the config."""
    config_dict = {
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
                },
            },
        },
    }
    config = Config(config_dict, validate=False)
    resolver = Resolver(config)
    stage_config = resolver.get_stage_config(Stage.FIT)
    assert "metrics" not in stage_config.get()["system"]


def test_get_stage_config_no_args_section():
    """Test get_stage_config when there is no 'args' section in the config."""
    config_dict = {
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
                },
            },
        },
    }
    config = Config(config_dict, validate=False)
    resolver = Resolver(config)
    stage_config = resolver.get_stage_config(Stage.FIT)
    assert "args" not in stage_config.get()


def test_get_stage_config_args_for_other_stages_removed(full_config_dict):
    """Test that args for other stages are removed."""
    config = Config(full_config_dict, validate=False)
    resolver = Resolver(config)
    stage_config = resolver.get_stage_config(Stage.FIT)
    assert "validate" not in stage_config.get()["args"]
    assert "test" not in stage_config.get()["args"]
    assert "predict" not in stage_config.get()["args"]


def test_get_stage_config_criterion_in_validate_but_not_test_predict(full_config_dict):
    """Test that criterion is present in validate stage but not test/predict."""
    config = Config(full_config_dict, validate=False)
    resolver = Resolver(config)

    validate_config = resolver.get_stage_config(Stage.VALIDATE)
    assert "criterion" in validate_config.get()["system"]

    test_config = resolver.get_stage_config(Stage.TEST)
    assert "criterion" not in test_config.get()["system"]

    predict_config = resolver.get_stage_config(Stage.PREDICT)
    assert "criterion" not in predict_config.get()["system"]
