import importlib
import inspect
import sys
from dataclasses import field, make_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, get_origin

from loguru import logger
from omegaconf import MISSING, DictConfig, OmegaConf

from lighter.utils import import_attr

# Resolvers - functions that can be used from the config. For example, given a function with the
# name `fn` that accepts two inputs, it can be called from configs as follows: `${fn: 1, 2}`.

OmegaConf.register_new_resolver("eval", eval)
# `${import:<ATTR>}` can be used to import an attribute via config. E.g. `${import:torch.float}`
OmegaConf.register_new_resolver("import", import_attr)
# `${now:}` can be used in configs to get the current datetime string
OmegaConf.register_new_resolver("now", lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))


def init_config(config_path: str) -> DictConfig:
    """Loads and returns the config from the given path and imports
    the project module if specified.

    Args:
        config_path (str): path to the yaml file.

    Returns:
        DictConfig: configuration.
    """
    # Load the config file
    conf = OmegaConf.load(config_path)
    # Allows the framework to find user-defined, project-specific, modules
    if conf.get("project", None) is not None:
        import_project_as_module(conf.project)
    return conf


def init_config_from_cli(args: list, log: bool = False) -> DictConfig:
    """Loads a YAML config file specified with 'config' key in command line arguments,
    parses the remaining comand line arguments as config options, and type checks it
    against the config's structured dataclass.

    Args:
        args (list): list of command line arguments.
        log (bool, optional): whether to log the config. Defaults to False.

    Returns:
        DictConfig: configuration.
    """

    cli = OmegaConf.from_dotlist(args)
    if cli.get("config", None) is None:
        logger.error("Please provide the path to a YAML config using `config` option. Exiting.")
        sys.exit()

    # Filter out arguments that are passed to the Trainer.fit() or similar methods
    main_keys = ["config", "trainer", "system", "project"]
    method_args = {k: cli.pop(k) for k in list(cli) if k.split(".")[0] not in main_keys}

    # Load the config file and import the project module
    conf = init_config(cli.pop("config"))
    # Merge the params of config file and those specified in cli (e.g. `system.batch_size=8`)
    conf = OmegaConf.merge(conf, cli)

    # Merge conf and the conf dataclass for type checking
    conf = OmegaConf.merge(construct_structured_config(conf), conf)
    if log:
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(conf)}")
    return conf, method_args


def construct_structured_config(conf: DictConfig) -> DictConfig:
    """Dynamically constructs the structured config which is used as a default base for the config.
    It infers attributes' name, type and default value of any Trainer and System implementation
    and populates the structured config with it. This provides default values for the config
    when user hasn't specified or overriden them and it allows static type checking of the config.

    Args:
        conf (DictConfig): non-structured config (in OmegaConf's vocabulary).

    Returns:
        DictConfig: input config made structured.
    """
    trainer = generate_omegaconf_dataclass("TrainerConfig", import_attr(conf.trainer["_target_"]))
    system = generate_omegaconf_dataclass("SystemConfig", import_attr(conf.system["_target_"]))

    fields: List = [
        # Field name, type, default value
        ("trainer", trainer, trainer),
        ("system", system, system),
        ("project", Optional[str], field(default=None)),
        ("CONSTANTS", Optional[Dict], field(default=None)),
    ]
    return OmegaConf.structured(make_dataclass("Config", fields))


def generate_omegaconf_dataclass(dataclass_name: str, source: Callable) -> Callable:
    """Generate a dataclass compatible with OmegaConf that has attributes name, type and value
    as specified in the source's arguments. If a default value is not specified, OmegaConf's
    "MISSING" is set instead. If an attribute has no type specified, then it is set to Any.
    Furthermore, if the type is a non-builtin class, it will be changed to Dict, since
    that class will be instantiated using the '_target_' key in the instance configuration.
    Attributes that can have different types, achieved using Union, will become Any since
    OmegaConf doesn't support Union yet.

    Args:
        dataclass_name (str): desired name of the dataclass.
        source (class or function): source of attributes for the dataclass.
    Returns:
        Callable: dataclass class (not object).
    """

    fields = [("_target_", str, f"{source.__module__}.{source.__name__}")]
    for param in inspect.signature(source).parameters.values():
        # Name
        name = param.name
        if name in ["args", "kwargs"]:
            continue

        # Type
        annotation = param.annotation
        # If annotation is empty, set it to Any
        if annotation is param.empty:
            annotation = Any
        # If an annotation is a class (but not a builtin one), set it to Dict.
        # This is because, in config, we can define an instance as a dict that
        # specifies its arguments and class type (with '_target_' key).
        if inspect.isclass(annotation) and annotation.__module__ != "builtins":
            annotation = Dict
        # TODO: Get rid of this when OmegaConf supports Union,
        # https://github.com/omry/omegaconf/issues/144
        if get_origin(annotation) == Union:
            annotation = Any

        # Default value
        default_value = param.default if param.default is not param.empty else MISSING

        fields.append((name, annotation, default_value))
    return make_dataclass(dataclass_name, fields)


def import_project_as_module(project: str) -> None:
    """Given the path to the project, import it as a module with name 'project'.

    Args:
        project (str): path to the project that will be loaded as module.
    """
    assert isinstance(project, str), "project needs to be a str path"

    # Import project as module with name "project" (https://stackoverflow.com/a/41595552).
    project_path = Path(project).resolve() / "__init__.py"
    if not project_path.is_file():
        logger.error(f"No `__init__.py` in project `{project_path}`. Exiting.")
        sys.exit()
    spec = importlib.util.spec_from_file_location("project", str(project_path))
    project_module = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(project_module)  # type: ignore
    sys.modules["project"] = project_module
    logger.info(f"Project directory {project} added as 'project' module.")
