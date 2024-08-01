from typing import Any

import copy
from functools import partial

import fire
from monai.bundle.config_parser import ConfigParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.tuner import Tuner

from lighter.system import LighterSystem
from lighter.utils.dynamic_imports import import_module_from_path

CONFIG_STRUCTURE = {
    "project": None,
    "vars": {},
    "args": {
        # Keys - names of the methods; values - arguments passed to them.
        "fit": {},
        "validate": {},
        "test": {},
        "predict": {},
        "lr_find": {},
        "scale_batch_size": {},
    },
    "system": {},
    "trainer": {},
}


def cli() -> None:
    """Defines the command line interface for running lightning trainer's methods."""
    commands = {method: partial(run, method) for method in CONFIG_STRUCTURE["args"]}
    try:
        fire.Fire(commands)
    except TypeError as e:
        if "run() takes 1 positional argument but" in str(e):
            raise ValueError(
                "Ensure that only one command is run at a time (e.g., 'lighter fit') and that "
                "other command line arguments start with '--' (e.g., '--config', '--system#batch_size=1')."
            ) from e
        raise


def parse_config(**kwargs) -> ConfigParser:
    """
    Parses configuration files and updates the provided parser
    with given keyword arguments. Returns an updated parser object.

    Args:
        **kwargs (dict): Keyword arguments containing 'config' and, optionally, config overrides.
    Returns:
        An instance of ConfigParser with configuration and overrides merged and parsed.
    """
    config = kwargs.pop("config", None)
    if config is None:
        raise ValueError("'--config' not specified. Please provide a valid configuration file.")

    # Create a deep copy to ensure the original structure remains unaltered by ConfigParser.
    structure = copy.deepcopy(CONFIG_STRUCTURE)
    # Initialize the parser with the predefined structure.
    parser = ConfigParser(structure, globals=False)
    # Update the parser with the configuration file.
    parser.update(parser.load_config_files(config))
    # Update the parser with the provided cli arguments.
    parser.update(kwargs)
    return parser


def validate_config(parser: ConfigParser) -> None:
    """
    Validates the configuration parser against predefined structure.

    Args:
        parser (ConfigParser): The configuration parser instance to validate.

    Raises:
        ValueError: If there are invalid keys in the top-level configuration.
        ValueError: If there are invalid method names specified in the 'args' section.
    """
    invalid_root_keys = set(parser.get()) - set(CONFIG_STRUCTURE)
    if invalid_root_keys:
        raise ValueError(f"Invalid top-level config keys: {invalid_root_keys}. Allowed keys: {list(CONFIG_STRUCTURE)}.")

    invalid_args_keys = set(parser.get("args")) - set(CONFIG_STRUCTURE["args"])
    if invalid_args_keys:
        raise ValueError(f"Invalid key in 'args': {invalid_args_keys}. Allowed keys: {list(CONFIG_STRUCTURE['args'])}.")

    typechecks = {
        "project": (str, type(None)),
        "vars": dict,
        "system": dict,
        "trainer": dict,
        "args": dict,
        **{f"args#{k}": dict for k in CONFIG_STRUCTURE["args"]},
    }
    for key, dtype in typechecks.items():
        if not isinstance(parser.get(key), dtype):
            raise ValueError(f"Invalid value for key '{key}'. Expected a {dtype}.")


def run(method: str, **kwargs: Any) -> None:
    """Run the trainer method.

    Args:
        method (str): name of the trainer method to run.
        **kwargs (Any): keyword arguments that include 'config' and specific config overrides passed to `parse_config()`.
    """
    seed_everything()

    # Parse and validate the config.
    parser = parse_config(**kwargs)
    validate_config(parser)

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

    # Trainer/Tuner method arguments.
    method_args = parser.get_parsed_content(f"args#{method}")
    if any("dataloaders" in key or "datamodule" in key for key in method_args):
        raise ValueError("Datasets are defined within the 'system', not passed in `args`.")

    # Save the config to checkpoints under "hyper_parameters". Log it if a logger is defined.
    system.save_hyperparameters(parser.get())
    if trainer.logger is not None:
        trainer.logger.log_hyperparams(parser.get())

    # Run the trainer/tuner method.
    if hasattr(trainer, method):
        getattr(trainer, method)(system, **method_args)
    elif hasattr(Tuner, method):
        tuner = Tuner(trainer)
        getattr(tuner, method)(system, **method_args)
    else:
        raise ValueError(f"Method '{method}' is not a valid Trainer or Tuner method [{list(CONFIG_STRUCTURE['args'])}].")
