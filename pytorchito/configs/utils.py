from loguru import logger
from omegaconf import DictConfig, OmegaConf, ListConfig

import pytorchito
from pytorchito.utils.io import import_class_from_dirs_and_modules

IMPORT_LOCATIONS = [pytorchito]


def init_config(conf, config_class):
    # Run-specific config
    conf = conf if isinstance(conf, DictConfig) else OmegaConf.load(str(conf))

    # Allows the framework to find user-defined, project-specific, classes and their configs
    if conf.project_dir:

        assert isinstance(conf.project_dir, (ListConfig, str)), \
            "project_dir needs to be a list or str"

        if isinstance(conf.project_dir, ListConfig):
            IMPORT_LOCATIONS.extend(conf.project_dir)
        else:
            IMPORT_LOCATIONS.append(conf.project_dir)

        logger.info(f"Project directories {conf.project_dir} added to the"
                    " path to allow imports of modules from it.")

    # Make yaml mergeable by instantiating the dataclasses
    conf = instantiate_dataclasses_from_yaml(conf)
    # Merge default and run-specifig config
    return OmegaConf.merge(OmegaConf.structured(config_class), conf)


def instantiate_dataclasses_from_yaml(conf):
    """Goes through a config and instantiates the fields that are dataclasses.
    A field is a dataclass if its key can be found in the keys of the IMPORT_LOCATIONS.
    Each such dataclass should have an entry "name" which is used to import its dataclass
    class using that "name" + "Config" as class name.
    Instantiates the deepest dataclasses first as otherwise OmegaConf would throw an error.
    """
    for key in get_all_conf_keys(conf):
        # Get the field for that key
        field = OmegaConf.select(conf, key)
        if is_dataclass(field):
            dataclass = init_dataclass(field)
            # Update the field for that key with the newly instantiated dataclass
            OmegaConf.update(conf, key, OmegaConf.merge(dataclass, field), merge=False)
    return conf


def init_dataclass(field):
    """Initialize a dataclass. Requires the field to have a "name" entry and a dataclass class
    whose destination can be found with IMPORT_LOCATIONS. Assumes that the class name is of
    format "name" + "Config", e.g. "MRIDatasetConfig".
    """
    dataclass_name = f'{field["name"]}Config'
    dataclass = import_class_from_dirs_and_modules(dataclass_name, IMPORT_LOCATIONS)
    return OmegaConf.structured(dataclass)


def is_dataclass(field):
    """If a field contains `name` key, it is a dataclass."""
    return bool(isinstance(field, DictConfig) and "name" in field)


def get_all_conf_keys(conf):
    """Get all keys from a conf and order from them the deepest to the shallowest."""
    conf = OmegaConf.to_container(conf)
    keys = list(iterate_nested_dict_keys(conf))
    # Order deeper to shallower
    return keys[::-1]


def iterate_nested_dict_keys(dictionary):
    """Returns an iterator that returns all keys of a nested dictionary ordered
    from the shallowest to the deepest key. The nested keys are in the dot-list format,
    e.g. "gan.discriminator.name".
    """
    if isinstance(dictionary, dict):
        current_level_keys = []
        for key in dictionary.keys():
            current_level_keys.append(key)
            yield key
        for key in current_level_keys:
            value = dictionary[key]
            for ret in iterate_nested_dict_keys(value):
                yield f"{key}.{ret}"
