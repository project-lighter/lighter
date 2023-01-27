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
            logger.info(f"`{get_name(obj, True)}` has no attribute `{attr}`. Exiting.")
            sys.exit()
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


def get_name(_callable: Callable, include_module_name: bool = False) -> str:
    """Get the name of an object, class or function.

    Args:
        _callable (Callable): object, class or function.
        include_module_name (bool, optional): whether to include the name of the module from
            which it comes. Defaults to False.

    Returns:
        str: name
    """
    name = type(_callable).__name__ if isinstance(_callable, object) else _callable.__name__
    if include_module_name:
        module = type(_callable).__module__ if isinstance(_callable, object) else _callable.__module__
        name = f"{module}.{name}"
    return name
