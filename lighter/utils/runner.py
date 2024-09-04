from typing import Any

from functools import partial

import fire
from monai.bundle.config_parser import ConfigParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.tuner import Tuner

from lighter.system import LighterSystem
from lighter.utils.dynamic_imports import import_module_from_path
from lighter.utils.schema import ArgsConfigSchema, ConfigSchema


def cli() -> None:
    """Defines the command line interface for running lightning trainer's methods."""
    commands = {method: partial(run, method) for method in ArgsConfigSchema.model_fields}
    fire.Fire(commands)


def parse_config(**kwargs) -> ConfigParser:
    """
    Parses configuration files and updates the provided parser
    with given keyword arguments. Returns an updated parser object.

    Args:
        **kwargs (dict): Keyword arguments containing 'config' and, optionally, config overrides.
    Returns:
        An instance of ConfigParser with configuration and overrides merged and parsed.

    Raises:
        ValueError: If '--config' is not specified in the keyword arguments.
        ValueError: If the configuration validation against the schema fails.
    """
    if "config" not in kwargs:
        raise ValueError("'--config' not specified. Please provide a valid configuration file.")

    # Initialize the parser with the predefined structure.
    parser = ConfigParser(ConfigSchema().dict(), globals=False)
    # Update the parser with the configuration file.
    parser.update(parser.load_config_files(kwargs.pop("config")))
    # Update the parser with the provided cli arguments.
    parser.update(kwargs)
    # Validate the configuration against the schema.
    ConfigSchema(**parser.config)
    return parser


def run(*methods: str, **kwargs: Any) -> None:
    """
    Execute specified methods of the Lightning Trainer or Tuner.

    Args:
        *methods (str): Method name(s) to be executed.
        **kwargs (Any): Keyword arguments that include 'config' and specific config overrides passed to `parse_config()`.
    """
    for method in methods:
        if method not in ArgsConfigSchema.model_fields:
            valid_methods = list(ArgsConfigSchema.model_fields)
            raise ValueError(
                f"Invalid command '{method}' specified. Available commands: {valid_methods}. "
                f"If you intended to pass an argument, ensure it is prefixed with '--'."
            )

    seed_everything()

    # Parse and validate the config.
    parser = parse_config(**kwargs)

    # Project. If specified, the give path is imported as a module.
    project = parser.get_parsed_content("project")
    if project is not None:
        import_module_from_path("project", project)

    # System
    system = parser.get_parsed_content("system")
    if not isinstance(system, LighterSystem):
        raise ValueError("Expected 'system' to be an instance of 'LighterSystem'")

    # Trainer
    trainer = parser.get_parsed_content("trainer")
    if not isinstance(trainer, Trainer):
        raise ValueError("Expected 'trainer' to be an instance of PyTorch Lightning 'Trainer'")

    # Save the config to checkpoints under "hyper_parameters". Log it if a logger is defined.
    system.save_hyperparameters(parser.config)
    if trainer.logger is not None:
        trainer.logger.log_hyperparams(parser.config)

    # Run the Trainer/Tuner method(s).
    for method in methods:
        args = parser.get_parsed_content(f"args#{method}", default={})
        if hasattr(trainer, method):
            getattr(trainer, method)(system, **args)
        elif hasattr(Tuner, method):
            tuner = Tuner(trainer)
            getattr(tuner, method)(system, **args)
