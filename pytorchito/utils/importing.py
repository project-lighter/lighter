import inspect
from functools import partial
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


def instantiate(conf, **kwargs):
    """Instantiates a class or function defined in config. Config's `_target_` specifies what
    to construct. Additional keyword arguments can be passed too.

    Args:
        conf (omegaconf.DictConfig): Object's config.
        **kwargs (any, optional): Keyword arguments to be passed to the class or function.

    Returns:
        object or partial function: If '_target_' is a class, then it returns an object of
        that class, instantiated with config, and, optionally, keyword arguments. 
        Otherwise, the '_target_' is a function and it returns a partial function options
        fixed as specified in conf and keyword arguments.
    """
    conf = OmegaConf.to_container(conf)
    attr = import_attr(conf.pop("_target_"))
    if inspect.isclass(attr):
        return attr(**conf, **kwargs)
    # Set the specified config and keyword arguments to the function by creating a partial function
    return partial(attr, **conf, **kwargs)


def instantiate_dict_list_union(conf, to_dict=False, **kwargs):
    """In config, some options like `criteria` can be either a single
    criterion (a DictConfig), or multiple criteria (a ListConfig of
    DictConfigs). This function instantiates objects from such options.

    Args:
        conf (omegaconf.DictConfig or omegaconf.ListConfig): Object's config.
        to_dict (bool, optional): If True, the function will return a dictionary of objects
            with their names as dict keys instead of a list of objects. Defaults to False.
        **kwargs (any, optional): Keyword arguments to be passed to the object's constructor.

    Returns:
        list or dict: list of instantiated objects or, if `to_dict=True`, a dict
            of instantiated objects with objects' class names as keys.
    """
    assert isinstance(conf, (DictConfig, ListConfig))

    conf = [conf] if isinstance(conf, (DictConfig)) else conf

    instantiated = [instantiate(c, **kwargs) for c in conf]

    if to_dict:
        # If partial fn, get the name of the base fn. Otherwise, it's an object, get the type name.
        get_name = lambda x: x.func.__name__ if isinstance(x, partial) else type(x).__name__
        instantiated = {get_name(instance): instance for instance in instantiated}

    return instantiated
