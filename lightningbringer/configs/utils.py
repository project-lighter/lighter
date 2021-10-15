import inspect
import typing
from dataclasses import make_dataclass

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

    return conf


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
        # If an annotation is a class (but not a builtin one), set it to Dict.
        # This is because, in config, we can define an instance as a dict that
        # specifies its arguments and class type (with '_target_' key).
        if inspect.isclass(annotation) and annotation.__module__ != "builtins":
            annotation = typing.Dict
        # TODO: Get rid of this when OmegaConf supports Union
        if str(annotation).startswith("typing.Union"):
            annotation = typing.Any

        # Default value
        default_value = param.default if not param.default is param.empty else MISSING

        fields.append((name, annotation, default_value))
    return make_dataclass(dataclass_name, fields)
