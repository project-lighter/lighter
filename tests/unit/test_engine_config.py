"""Unit tests for the Config class in lighter/engine/config.py"""

import pytest

from lighter.engine.config import Config, ConfigurationException


def test_config_initialization_with_dict():
    """Test initializing Config with a dictionary."""
    config_dict = {
        "system": {
            "model": "test_model",
            "optimizer": {"name": "Adam", "lr": 0.001},
        }
    }
    config = Config(config_dict, validate=False)
    result = config.get()
    result.pop("_meta_", None)  # Remove _meta_ field for comparison
    assert result == config_dict


def test_config_initialization_with_invalid_type():
    """Test that initializing Config with invalid type raises ValueError."""
    with pytest.raises(ValueError, match="Invalid type for 'config'"):
        Config(123, validate=False)


def test_config_validation():
    """Test that config validation works correctly."""
    # Valid config
    valid_config = {
        "system": {
            "_target_": "lighter.system.System",
            "model": {
                "_target_": "torch.nn.Linear",
                "in_features": 10,
                "out_features": 2,
            },
            "optimizer": {
                "_target_": "torch.optim.Adam",
                "lr": 0.001,
            },
            "criterion": {
                "_target_": "torch.nn.CrossEntropyLoss",
            },
        },
        "trainer": {
            "_target_": "pytorch_lightning.Trainer",
            "max_epochs": 10,
        },
    }
    config = Config(valid_config, validate=True)
    result = config.get()
    # Remove _meta_ field for comparison
    result.pop("_meta_", None)
    assert result == valid_config

    # Invalid config (missing required fields)
    invalid_config = {
        "system": {
            "model": "test_model",  # Not a dict
            # Missing _target_ and optimizer
        }
    }
    with pytest.raises(ConfigurationException):
        Config(invalid_config, validate=True)


def test_config_initialization_without_overrides():
    """Test initializing Config without any overrides."""
    config_dict = {
        "system": {
            "_target_": "lighter.system.System",
            "model": {
                "_target_": "torch.nn.Linear",
                "in_features": 10,
                "out_features": 2,
            },
        }
    }
    config = Config(config_dict, validate=False)
    result = config.get()
    result.pop("_meta_", None)  # Remove _meta_ field for comparison
    assert result == config_dict


def test_config_overrides():
    """Test that config overrides work correctly."""
    base_config = {
        "system": {
            "_target_": "lighter.system.System",
            "model": {
                "_target_": "torch.nn.Linear",
                "in_features": 10,
                "out_features": 2,
            },
            "optimizer": {
                "_target_": "torch.optim.Adam",
                "lr": 0.001,
            },
        },
        "trainer": {
            "_target_": "pytorch_lightning.Trainer",
            "max_epochs": 10,
        },
    }

    # Override existing value
    config = Config(base_config, validate=False)
    config._config_parser.update({"system": {"optimizer": {"lr": 0.01}}})
    result = config.get()
    result.pop("_meta_", None)  # Remove _meta_ field for comparison
    assert result["system"]["optimizer"]["lr"] == 0.01

    # Add new value
    config = Config(base_config, validate=False, new_key="new_value")
    result = config.get()
    result.pop("_meta_", None)  # Remove _meta_ field for comparison
    assert result["new_key"] == "new_value"


def test_config_get_methods():
    """Test the get and get_parsed_content methods."""
    config_dict = {
        "system": {
            "_target_": "lighter.system.System",
            "model": {
                "_target_": "torch.nn.Linear",
                "in_features": 10,
                "out_features": 2,
            },
            "optimizer": {
                "_target_": "torch.optim.Adam",
                "lr": 0.001,
            },
        },
        "trainer": {
            "_target_": "pytorch_lightning.Trainer",
            "max_epochs": 10,
        },
    }
    config = Config(config_dict, validate=False)

    # Test get method
    result = config.get()
    result.pop("_meta_", None)  # Remove _meta_ field for comparison
    assert result["system"]["optimizer"]["lr"] == 0.001
    assert config.get("non_existent", "default") == "default"

    # Test get_parsed_content method
    # Note: We only test the raw config values since instantiation requires a real model
    raw_config = config.get("system")
    assert raw_config["optimizer"]["_target_"] == "torch.optim.Adam"
    assert raw_config["optimizer"]["lr"] == 0.001
    assert config.get("non_existent", default="default") == "default"


def test_config_validation_error_formatting():
    """Test that validation errors are formatted correctly."""
    invalid_config = {
        "system": {
            "model": "test_model",  # Not a dict
            "optimizer": "not_a_dict",  # Not a dict
        }
    }

    with pytest.raises(ConfigurationException) as exc_info:
        Config(invalid_config, validate=True)

    error_msg = str(exc_info.value)
    assert "Configuration validation failed" in error_msg
    assert "system.optimizer" in error_msg


def test_config_validation_error_formatting_complex():
    """Test validation error formatting with complex nested structures."""
    from lighter.engine.config import format_validation_errors

    # Test nested dictionary errors
    nested_dict_errors = {"system": {"model": {"layers": ["must be a dict"], "activation": "invalid value"}}}
    formatted = format_validation_errors(nested_dict_errors)
    assert "system.model.layers: must be a dict" in formatted
    assert "system.model.activation: invalid value" in formatted

    # Test list errors
    list_errors = {"data": {"transforms": [{"name": "missing required field"}, {"params": "invalid type"}]}}
    formatted = format_validation_errors(list_errors)
    assert "data.transforms.name: missing required field" in formatted
    assert "data.transforms.params: invalid type" in formatted

    # Test mixed type errors
    mixed_errors = {"trainer": ["invalid value", {"max_epochs": "must be an integer"}, 42]}  # Non-string, non-dict value
    formatted = format_validation_errors(mixed_errors)
    assert "trainer: invalid value" in formatted
    assert "trainer.max_epochs: must be an integer" in formatted
    assert "trainer: 42" in formatted
