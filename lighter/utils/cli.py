from typing import Any, Dict

import sys
from functools import partial

import fire
import yaml
from loguru import logger
from monai.bundle.config_parser import ConfigParser
from pytorch_lightning import seed_everything

from lighter.utils.dynamic_imports import import_module_from_path
from lighter.utils.misc import ensure_list


def interface():
    """Defines the command line interface for running Trainer's methods. The available methods are:
    - fit
    - validate
    - test
    - predict
    - tune
    """
    commands = {}
    # All Trainer methods' names
    for name in ["fit", "validate", "test", "predict", "tune"]:
        # Creates command for fire to run. The command is the Trainer method's name. Fire passes
        # **kwargs to the function.
        commands[name] = partial(run_trainer_method, name)

    fire.Fire(commands)


def parse_config(**kwargs):
    """
    Parses configuration files and updates the provided parser
    with given keyword arguments. Returns an updated parser object.

    Args:
        **kwargs (dict): Keyword arguments containing configuration data.
            config_file (str): Path to the main configuration file.
            args_file (str, optional): Path to secondary configuration file for additional arguments.
            Additional key-value pairs can also be provided to be added or updated in the parser.

    Returns:
        ConfigParser: An instance of ConfigParser with parsed and merged configuration data.
    """

    # Check that a config file is specified.
    if "config_file" not in kwargs:
        raise ValueError("No config file specified. Exiting.")

    # Load config from `config_file`
    config = ConfigParser.load_config_files(kwargs["config_file"])
    if "project" in config:
        import_module_from_path("project", config["project"])

    parser = ConfigParser()
    parser.read_config(kwargs["config_file"])

    if "args_file" in kwargs:
        args = ConfigParser.load_config_file(kwargs["args_file"])
        parser.update(pairs=args)

    parser.update(pairs=kwargs)

    return parser


def run_trainer_method(method: Dict, **kwargs: Any):
    """Call monai.bundle.run() on a Trainer method. If a project path
    is defined in the config file(s), import it.

    Args:
        method_name: name of the Trainer method to run. ["fit", "validate", "test", "predict", "tune"].
        **kwargs (Any): keyword arguments passed to the `monai.bundle.run` function.
    """
    # Sets the random seed to `PL_GLOBAL_SEED` env variable. If not specified, it picks a random seed.
    seed_everything()

    # Parse the config file(s).
    parser = parse_config(**kwargs)

    # Get trainer and system
    trainer = parser.get_parsed_content("trainer")
    system = parser.get_parsed_content("system")

    # Export config to be loaded by other modules, if neccessary
    parser.export_config_file(parser.get(), f"{trainer.default_root_dir}/config.yaml")

    # Run the Trainer method.
    if not hasattr(trainer, method):
        raise ValueError(f"Trainer has no method named {method}.")

    getattr(trainer, method)(system)
