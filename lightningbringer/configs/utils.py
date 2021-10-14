import importlib
import inspect
import sys
import typing
from dataclasses import make_dataclass
from pathlib import Path

from loguru import logger
from omegaconf import MISSING, OmegaConf


def init_config(omegaconf_args, config_class):
    """Loads a YAML config file specified with 'config' key in command line arguments,
    type checks it against the config's dataclass, and parses the remaining comand line
    arguments as config options.

    Args:
        omegaconf_args (list): list of command line arguments.
        config_class (dataclass): config's dataclass, used for static type checking by OmegaConf.

    Returns:
        omegaconf.DictConfig: configuration
    """

    cli = OmegaConf.from_dotlist(omegaconf_args)
    assert "config" in cli, "Please provide path to a YAML config using `config` option."

    conf = OmegaConf.load(cli.pop("config"))
    # Merge conf and the conf dataclass for type checking
    conf = OmegaConf.merge(OmegaConf.structured(config_class), conf)
    # Merge yaml conf and cli conf
    conf = OmegaConf.merge(conf, cli)

    # Allows the framework to find user-defined, project-specific, classes and their configs
    if conf.project:
        import_project_as_module(conf.project)

    return conf


def import_project_as_module(project):
    """Given the path to the project, import it as a module with name 'project'.

    Args:
        project (str): path to the project that will be loaded as module.
    """
    assert isinstance(project, str), "project needs to be a str path"

    # Import project as module with name "project", https://stackoverflow.com/a/41595552
    project_path = Path(project).resolve() / "__init__.py"
    assert project_path.is_file(), f"No `__init__.py` in project `{project_path}`."
    spec = importlib.util.spec_from_file_location("project", str(project_path))
    project_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_module)
    sys.modules["project"] = project_module

    logger.info(f"Project directory {project} added as a module with name 'project'.")


def generate_omegaconf_dataclass(dataclass_name, source):
    """Generate a dataclass compatible with OmegaConf that has attributes name, type and value
    as specified in the source's arguments. If a default value is not specified, OmegaConf's
    "MISSING" is set instead. If an attribute has no type specified, then it is set to typing.Any.
    Furthermore, if the type is a non-builtin class, it will be changed to typing.Dict, since
    that class will be instantiated using the '_target_' key in the instance configuration.
    Attributes that can have different types, achieved through Union, will become typing.Any 
    since OmegaConf doesn't support Union yet.

    Args:
        dataclass_name (str): desired name of the dataclass.
        source (class or function): source of attributes for the dataclass.
    Returns:
        dataclass: dataclass class (not object).
    """

    # If partial fn, get the name of the base fn. Otherwise, it's an object, get the type name.
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
            annotation = typing.Any
        # If annotation is a non-builtin class, set it to Dict. This is because
        # the class is specified via '_target_' key in the config, along with other
        # arguments for its instantiation.
        if inspect.isclass(annotation) and annotation.__module__ != "builtins":
            print(annotation)
            annotation = typing.Dict
        # TODO: Get rid of this when OmegaConf supports Union
        if str(annotation).startswith("typing.Union"):
            annotation = typing.Any

        # Default value
        default_value = param.default if not param.default is param.empty else MISSING

        fields.append((name, annotation, default_value))
    return make_dataclass(dataclass_name, fields)
