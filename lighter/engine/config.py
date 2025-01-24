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
        config: str | dict | None = None,
        schema: dict | None = SCHEMA,
        **config_overrides: Any,
    ):
        """
        Initialize the Config object.

        Args:
            config: Path to a YAML configuration file or a dictionary containing the configuration.
            schema: A Cerberus schema for validation.
            config_overrides: Keyword arguments to override values in the configuration file
        """
        self._config_parser = ConfigParser(globals=False)
        if isinstance(config, dict):
            self._config_parser.update(config)
        elif isinstance(config, str):
            # Read one or more config files separated by commas
            config = self._config_parser.load_config_files(config)
            self._config_parser.read_config(config)
        else:
            raise ValueError("Invalid type for 'config'. Must be a dictionary or path(s) to YAML file(s).")
        self._config_parser.parse()

        if schema:
            self._validator = cerberus.Validator(schema)
            self.validate()

        self._config_parser.update(config_overrides)

    def get(self, key: str | None = None, default: Any = None) -> Any:
        """Get raw content for the given key. If key is None, get the entire config."""
        return self._config_parser.config if key is None else self._config_parser.config.get(key, default)

    def get_parsed_content(self, key: str | None = None, default: Any = None) -> Any:
        """
        Get the parsed content for the given key. If key is None, get the entire parsed config.
        """
        return self._config_parser.get_parsed_content(key, default=default)

    def validate(self) -> None:  # Add normalize argument
        """Validate the configuration against the Cerberus schema."""
        if not self._validator.validate(self.get()):
            error_messages = self._format_validation_errors(self._validator.errors)
            raise ConfigurationException(error_messages)

    def _format_validation_errors(self, errors: dict) -> str:
        """
        Recursively format Cerberus validation errors into a readable string.
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
