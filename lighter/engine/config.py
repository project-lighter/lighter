from typing import Any

import cerberus
from monai.bundle.config_parser import ConfigParser

from lighter.engine.schema import SCHEMA


class ConfigurationException(Exception):
    """Custom exception for validation errors."""

    def __init__(self, errors: str):
        super().__init__(f"Configuration validation failed:\n{errors}")


class Config:
    """
    Handles loading, overriding, validating, and normalizing configurations.
    """

    def __init__(
        self,
        config: str | dict,
        validate: bool,
        **config_overrides: Any,
    ):
        """
        Initialize the Config object.

        Args:
            config: Path to a YAML configuration file or a dictionary containing the configuration.
            validate: Whether to validate the configuration.
            config_overrides: Keyword arguments to override values in the configuration file
        """
        if not isinstance(config, (dict, str, type(None))):
            raise ValueError("Invalid type for 'config'. Must be a dictionary or (comma-separated) path(s) to YAML file(s).")

        self._config_parser = ConfigParser(globals=False)
        self._config_parser.read_config(config)
        self._config_parser.parse()

        # TODO: verify that switching from .update(config_overrides) to .set(value, name) is
        # a valid approach. The latter allows creation of currently non-existent keys.
        for name, value in config_overrides.items():
            self._config_parser.set(value, name)

        # Validate the configuration
        if validate:
            validator = cerberus.Validator(SCHEMA)
            valid = validator.validate(self.get())
            if not valid:
                errors = format_validation_errors(validator.errors)
                raise ConfigurationException(errors)

    def get(self, key: str | None = None, default: Any = None) -> Any:
        """Get raw content for the given key. If key is None, get the entire config."""
        return self._config_parser.config if key is None else self._config_parser.config.get(key, default)

    def get_parsed_content(self, key: str | None = None, default: Any = None) -> Any:
        """
        Get the parsed content for the given key. If key is None, get the entire parsed config.
        """
        return self._config_parser.get_parsed_content(key, default=default)


def format_validation_errors(errors: dict) -> str:
    """
    Recursively format validation errors into a readable string.
    """
    messages = []

    def process_error(key, value, base_path=""):
        full_key = f"{base_path}.{key}" if base_path else key

        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                process_error(sub_key, sub_value, full_key)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    messages.append(f"{full_key}: {item}")
                elif isinstance(item, dict):
                    process_error(key, item, base_path)
                else:
                    messages.append(f"{full_key}: {item}")
        else:
            messages.append(f"{full_key}: {value}")

    process_error("", errors)
    return "\n".join(messages)
