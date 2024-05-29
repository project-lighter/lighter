from typing import Any

from functools import partial

import fire
from monai.bundle.config_parser import ConfigParser
from pytorch_lightning import seed_everything

from lighter.system import LighterSystem
from lighter.utils.dynamic_imports import import_module_from_path

CONFIG_STRUCTURE = {"project": None, "system": {}, "trainer": {}, "args": {}, "vars": {}}
TRAINER_METHOD_NAMES = ["fit", "validate", "test", "predict", "lr_find", "scale_batch_size"]


def cli() -> None:
    """Defines the command line interface for running lightning Trainer/Tuner's methods."""
    commands = {method: partial(run, method) for method in TRAINER_METHOD_NAMES}
    fire.Fire(commands)


def parse_config(**kwargs) -> ConfigParser:
    """
    Parses configuration files and updates the provided parser
    with given keyword arguments. Returns an updated parser object.

    Args:
        **kwargs (dict): Keyword arguments containing 'config_file' and, optionally, config overrides.
    Returns:
        An instance of ConfigParser with configuration and overrides merged and parsed.
    """
    # Ensure a config file is specified.
    config_file = kwargs.pop("config_file", None)
    if config_file is None:
        raise ValueError("--config_file not specified. Exiting.")

    # Read the config file and update it with overrides.
    parser = ConfigParser(CONFIG_STRUCTURE, globals=False)
    parser.read_config(config_file)
    parser.update(kwargs)
    return parser


def validate_config(parser: ConfigParser) -> None:
    """
    Validates the configuration parser against predefined structures and allowed method names.

    This function checks if the keys in the top-level of the configuration parser are valid according to the
    CONFIG_STRUCTURE. It also verifies that the 'args' section of the configuration only contains keys that
    correspond to valid Trainer/Tuner method names as defined in TRAINER_METHOD_NAMES.

    Args:
        parser (ConfigParser): The configuration parser instance to validate.

    Raises:
        ValueError: If there are invalid keys in the top-level configuration.
        ValueError: If there are invalid method names specified in the 'args' section.
    """
    # Validate parser keys against structure
    invalid_keys = set(parser.get().keys()) - set(CONFIG_STRUCTURE.keys()) - {"_meta_", "_requires_"}
    if invalid_keys:
        raise ValueError(f"Invalid top-level config keys: {list(invalid_keys)}. Allowed keys: {list(CONFIG_STRUCTURE.keys())}")

    # Validate that 'args' contains only valid Trainer/Tuner method names.
    invalid_keys = set(parser.get("args").keys()) - set(TRAINER_METHOD_NAMES)
    if invalid_keys:
        raise ValueError(f"Invalid method names in 'args': {invalid_keys}. Allowed methods are: {TRAINER_METHOD_NAMES}")


def run(method: str, **kwargs: Any):
    """Run the Trainer/Tuner method.

    Args:
        method (str): name of the Trainer/Tuner method to run.
        **kwargs (Any): keyword arguments that include 'config_file' and specific config overrides passed to `parse_config()`.
    """
    seed_everything()

    # Parse and validate the config.
    parser = parse_config(**kwargs)
    validate_config(parser)

    # Import the project folder as a module, if specified.
    project = parser.get_parsed_content("project")
    if project is not None:
        import_module_from_path("project", project)

    # Get the main components from the parsed config.
    system = parser.get_parsed_content("system")
    trainer = parser.get_parsed_content("trainer")
    trainer_method_args = parser.get_parsed_content(f"args#{method}")

    # Checks
    if not isinstance(system, LighterSystem):
        raise ValueError(f"Expected 'system' to be an instance of LighterSystem, got {system.__class__.__name__}.")
    if not hasattr(trainer, method):
        raise ValueError(f"{trainer.__class__.__name__} has no method named '{method}'.")
    if any("dataloaders" in key or "datamodule" in key for key in trainer_method_args):
        raise ValueError("All dataloaders should be defined as part of the LighterSystem, not passed as method arguments.")

    # Save the config to checkpoints under "hyper_parameters" and log it if a logger is defined.
    config = parser.get()
    config.pop("_meta_")  # MONAI Bundle adds this automatically, remove it.
    system.save_hyperparameters(config)
    if trainer.logger is not None:
        trainer.logger.log_hyperparams(config)

    # Run the Trainer method.
    getattr(trainer, method)(system, **trainer_method_args)
