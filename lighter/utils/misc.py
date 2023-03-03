from typing import Any, Callable, List

import inspect
import sys

from loguru import logger


def ensure_list(vals: Any) -> List:
    """Wrap the input into a list if it is not a list. If it is a None, return an empty list.

    Args:
        vals (Any): input to wrap into a list.

    Returns:
        List: output list.
    """
    if isinstance(vals, list):
        return vals
    if isinstance(vals, tuple):
        return list(vals)
    if vals is None:
        return []
    return [vals]


def setattr_dot_notation(obj: Callable, attr: str, value: Any):
    """Set object's attribute. Supports dot notation.

    Args:
        obj (Callable): object.
        attr (str): attribute name of the object.
        value (Any): value to set the attribute to.
    """
    if "." not in attr:
        if not hasattr(obj, attr):
            raise AttributeError(f"`{get_name(obj, True)}` has no attribute `{attr}`.")
        setattr(obj, attr, value)
    # Solve recursively if the attribute is defined in dot-notation
    else:
        obj_name, attr = attr.split(".", maxsplit=1)
        setattr_dot_notation(getattr(obj, obj_name), attr, value)


def hasarg(_callable: Callable, arg_name: str) -> bool:
    """Check if a function, class, or method has an argument with the specified name.

    Args:
        _callable (Callable): function, class, or method to inspect.
        arg_name (str): argument name to check for.

    Returns:
        bool: `True` if the argument if the specified name exists.
    """

    args = inspect.signature(_callable).parameters.keys()
    return arg_name in args


def countargs(_callable: Callable) -> bool:
    """Count the number of arguments that a function, class, or method accepts.
    Will not count the `self` argument.

    Args:
        callable (Callable): function, class, or method to inspect.

    Returns:
        int: number of arguments that it accepts.
    """
    return len([arg for arg in inspect.signature(_callable).parameters.keys() if arg != "self"])


def get_name(_callable: Callable, include_module_name: bool = False) -> str:
    """Get the name of an object, class or function.

    Args:
        _callable (Callable): object, class or function.
        include_module_name (bool, optional): whether to include the name of the module from
            which it comes. Defaults to False.

    Returns:
        str: name
    """
    if isinstance(_callable, object) and type(_callable).__module__ != "builtins":
        name = _callable.__class__.__name__
        if include_module_name:
            name = f"{_callable.__class__.__module__}.{name}"
    else:
        name = _callable.__name__
        if include_module_name:
            name = f"{_callable.__module__}.{name}"
    return name
