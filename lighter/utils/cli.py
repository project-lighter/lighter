from typing import Any, Dict

import sys
from functools import partial

import fire
import yaml
from loguru import logger
from monai.bundle.scripts import run
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
        # Creates the configuration for the Trainer method 'name' and sets 'system' as the model to run on.
        config = {name: {"_target_": f"$@trainer.{name}", "model": "@system"}}
        # Set the method name and config. Fire collects **kwargs from the CLI and passes them to the method.
        commands[name] = partial(run_trainer_method, name, config)
    fire.Fire(commands)


def run_trainer_method(method_name, method_config: Dict, **kwargs: Any):
    """Call monai.bundle.run() on a Trainer method. If a project path
    is defined in the config file(s), import it.

    Args:
        method_name: name of the Trainer method to run. ["fit", "validate", "test", "predict", "tune"].
        method_config: config definition of the Trainer method.
        **kwargs (Any): keyword arguments passed to the `monai.bundle.run` function.
    """
    # Sets the random seed to `PL_GLOBAL_SEED` env variable. If not specified, it picks a random seed.
    seed_everything()

    # Import the project as a module.
    if "config_file" in kwargs:
        project_imported = False
        # Handle multiple configs. Start from the config file specified last as it overrides the previous ones.
        for config in reversed(ensure_list(kwargs["config_file"])):
            with open(config, encoding="utf-8") as config:
                config = yaml.safe_load(config)
                if "project" not in config:
                    continue
                # Only one config file can specify the project path
                if project_imported:
                    logger.error("`project` must be specified in one config only. Exiting.")
                    sys.exit()
                # Import it as a module named 'project'.
                import_module_from_path("project", config["project"])
                project_imported = True

    # Run the Trainer method.
    run(method_name, **method_config, **kwargs)
