from omegaconf import OmegaConf, ListConfig, DictConfig


def import_attr(module_attr):
    """Imports an attribute (functions, classes, etc.).

    Args:
        module_attr (str): Full path to an attribute, e.g., `torchvision.datasets.CIFAR10`.

    Returns:
        any: returns the caller of the imported attribute
    """
    module, attr = module_attr.rsplit(".", 1)
    module = __import__(module, fromlist=[attr])
    return getattr(module, attr)


def instantiate(conf, *args, **kwargs):
    """Instantiates a class defined in config. Config's `_target_` specifies which class to
    construct. Remaining options are passed to the object's constructor.

    Args:
        conf (omegaconf.DictConfig): Object's config.
        *args (any, optional): Positional arguments to be passed to the object's constructor.
        **kwargs (any, optional): Keyword arguments to be passed to the object's constructor.

    Returns:
        object: Object of type specified with `_target_`, instantiated with config, and,
            optionally, positional or keyword arguments.
    """
    conf = OmegaConf.to_container(conf)
    klass = import_attr(conf.pop("_target_"))
    return klass(*args, **conf, **kwargs)


def instantiate_dict_list_union(conf, *args, to_dict=False, **kwargs):
    """In config, some options like `criteria` can be either a single
    criterion (a DictConfig), or multiple criteria (a ListConfig of
    DictConfigs). This function instantiates objects from such options.

    Args:
        conf (omegaconf.DictConfig or omegaconf.ListConfig): Object's config.
        to_dict (bool, optional): If True, the function will return a dictionary of objects
            with their names as dict keys instead of a list of objects. Defaults to False.
        *args (any, optional): Positional arguments to be passed to the object's constructor.
        **kwargs (any, optional): Keyword arguments to be passed to the object's constructor.

    Returns:
        list or dict: list of instantiated objects or, if `to_dict=True`, a dict
            of instantiated objects with objects' class names as keys.
    """
    assert isinstance(conf, (DictConfig, ListConfig))

    conf = [conf] if isinstance(conf, (DictConfig)) else conf

    instantiated = [instantiate(c, *args, **kwargs) for c in conf]
    if to_dict:
        instantiated = {type(instance).__name__: instance for instance in instantiated}

    return instantiated
