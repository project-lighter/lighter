from typing import Any, Dict

from monai.bundle.config_parser import ConfigParser
from pytorch_lightning import seed_everything

from lighter.utils.dynamic_imports import import_module_from_path


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
        raise ValueError("--config_file not specified. Exiting.")

    # Parse the config file(s).
    parser = ConfigParser()
    parser.read_config(kwargs.pop("config_file"))
    parser.update(pairs=kwargs)

    # Import the project folder as a module, if specified.
    project = parser.get("project", None)
    if project is not None:
        import_module_from_path("project", project)

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

    config = parser.get()
    config.pop("_meta_")
    # Save the config to model checkpoints under the "hyper_parameters" key.
    system.save_hyperparameters(config)
    # Log the config.
    if trainer.logger is not None:
        trainer.logger.log_hyperparams(config)

    # Run the Trainer method.
    if not hasattr(trainer, method):
        raise ValueError(f"Trainer has no method named {method}.")
    getattr(trainer, method)(system)
