"""Unit tests for the Config and Schema classes."""

import pytest
import yaml

from lighter.engine.config import Config

from lighter.utils.types.enums import Stage


@pytest.fixture
def minimal_config_dict():
    """Provides a minimal valid configuration dictionary."""
    return {
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
    }


@pytest.fixture
def minimal_config_yaml(minimal_config_dict):
    """Provides a minimal valid configuration as a YAML string."""
    return yaml.dump(minimal_config_dict)


def test_config_init_from_dict(minimal_config_dict):
    """Test Config initialization from a dictionary."""
    config = Config(minimal_config_dict, validate=False)
    assert config.get() == minimal_config_dict


def test_config_init_from_yaml_string(minimal_config_yaml):
    """Test Config initialization from a YAML string."""
    config = Config(minimal_config_yaml, validate=False)
    assert yaml.dump(config.get()) == minimal_config_yaml


def test_config_init_from_file(tmp_path, minimal_config_yaml):
    """Test Config initialization from a YAML file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(minimal_config_yaml)
    config = Config(str(config_file), validate=False)
    assert yaml.dump(config.get()) == minimal_config_yaml


def test_config_init_with_overrides(minimal_config_dict):
    """Test Config initialization with overrides."""
    config = Config(minimal_config_dict, validate=False, **overrides)
    expected_config = minimal_config_dict.copy()
    expected_config["trainer"]["max_epochs"] = 5
    assert config.get() == expected_config


def test_config_init_with_nested_overrides(minimal_config_dict):
    """Test Config initialization with nested overrides."""
    overrides = {"system#model#in_features": 2}
    config = Config(minimal_config_dict, validate=False, **overrides)
    expected_config = minimal_config_dict.copy()
    expected_config["system"]["model"]["in_features"] = 2
    assert config.get() == expected_config


def test_config_init_with_invalid_config_raises_error():
    """Test that an invalid configuration raises a ValueError during validation."""
    from lighter.engine.config import ConfigurationException

    invalid_config = {}  # Assuming an empty config is invalid
    with pytest.raises(ConfigurationException, match="Config validation error"):
        Config(invalid_config, validate=True)


def test_config_get_parsed_content(minimal_config_dict):
    """Test retrieving parsed content from the config."""
    config = Config(minimal_config_dict, validate=False)
    trainer_target = config.get_parsed_content("trainer#_target_")
    assert trainer_target == "pytorch_lightning.Trainer"

    lr = config.get_parsed_content("system#optimizer#lr")
    assert lr == 0.01


def test_config_get_parsed_content_with_default(minimal_config_dict):
    """Test retrieving parsed content with a default value."""
    config = Config(minimal_config_dict, validate=False)
    non_existent = config.get_parsed_content("non_existent_key", default="default_value")
    assert non_existent == "default_value"


def test_config_get_parsed_content_with_evaluation(minimal_config_dict):
    """Test retrieving parsed content that requires evaluation."""
    config = Config(minimal_config_dict, validate=False)
    # This will actually instantiate the Linear model and get its parameters
    model_params = config.get_parsed_content("system#optimizer#params")
    assert hasattr(model_params, "__iter__")  # Should be an iterable of parameters


def test_config_get_parsed_content_non_existent_key_no_default_raises_error(minimal_config_dict):
    """Test that retrieving a non-existent key without a default raises an error."""
    config = Config(minimal_config_dict, validate=False)
    with pytest.raises(KeyError):
        config.get_parsed_content("non_existent_key")


def test_config_get_parsed_content_with_list_index(minimal_config_dict):
    """Test retrieving parsed content from a list using index."""
    config = Config(minimal_config_dict, validate=False)
    tensor_str = config.get_parsed_content("system#dataloaders#train#dataset#tensors#0")
    assert tensor_str == "$torch.randn(10, 1)"


def test_config_get_parsed_content_with_invalid_list_index_raises_error(minimal_config_dict):
    """Test that an invalid list index raises an IndexError."""
    config = Config(minimal_config_dict, validate=False)
    with pytest.raises(IndexError):
        config.get_parsed_content("system#dataloaders#train#dataset#tensors#99")


def test_config_get_parsed_content_with_invalid_path_raises_error(minimal_config_dict):
    """Test that an invalid path raises a KeyError."""
    config = Config(minimal_config_dict, validate=False)
    with pytest.raises(KeyError):
        config.get_parsed_content("system#non_existent#key")


def test_config_validation_success(minimal_config_dict):
    """Test successful validation of a minimal config."""
    Config(minimal_config_dict, validate=True)  # Should not raise an error


def test_config_validation_missing_required_field():
    """Test config validation fails for a missing required field."""
    invalid_config = {
        "trainer": {"_target_": "pytorch_lightning.Trainer"},
    }
    with pytest.raises(ConfigurationException, match="Config validation error"):
        Config(invalid_config, validate=True)


def test_config_validation_invalid_type():
    """Test config validation fails for an invalid type."""
    invalid_config = {
        "trainer": {"_target_": "pytorch_lightning.Trainer"},
        "system": "not_a_dict",  # system should be a dict
    }
    with pytest.raises(ConfigurationException, match="Config validation error"):
        Config(invalid_config, validate=True)


def test_config_validation_extra_field_allowed(minimal_config_dict):
    """Test that extra fields not in schema are allowed by default."""
    config_with_extra = minimal_config_dict.copy()
    config_with_extra["system"]["extra_field"] = "some_value"  # Extra field
    Config(config_with_extra, validate=True)  # Should not raise an error


def test_config_validation_dataloaders_modes(minimal_config_dict):
    """Test config validation for dataloaders with valid and invalid modes."""
    valid_modes_config = minimal_config_dict.copy()
    valid_modes_config["system"]["dataloaders"] = {
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
            "dataset": {
                "_target_": "torch.utils.data.TensorDataset",
                "tensors": ["$torch.randn(10, 1)", "$torch.randn(10, 1)"],
            },
        },
    }
    Config(valid_modes_config, validate=True)  # Should not raise an error

    invalid_mode_config = minimal_config_dict.copy()
    invalid_mode_config["system"]["dataloaders"] = {
        "invalid_mode": {
            "_target_": "torch.utils.data.DataLoader",
            "dataset": {
                "_target_": "torch.utils.data.TensorDataset",
                "tensors": ["$torch.randn(10, 1)", "$torch.randn(10, 1)"],
            },
        }
    }
    with pytest.raises(ConfigurationException, match="Config validation error"):
        Config(invalid_mode_config, validate=True)
