from typing import Any, Callable, List

import inspect
import sys

from loguru import logger


def ensure_list(x: Any) -> List:
    """Wrap the input into a list if it is not a list. If it is a None, return an empty list.

    Args:
        x (Any): input to wrap into a list.

    Returns:
        List: output list.
    """
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if x is None:
        return []
    return [x]


def dot_notation_setattr(obj: Callable, attr: str, value: Any):
    """Set object's attribute. May use dot notation.

    Args:
        obj (Callable): object.
        attr (str): attribute name of the object.
        value (Any): attribute value to be set.
    """
    if "." not in attr:
        if not hasattr(obj, attr):
            logger.info(f"`{get_name(obj, True)}` has no attribute `{attr}`. Exiting.")
            sys.exit()
        setattr(obj, attr, value)
    # Solve recursively if the attribute is defined in dot-notation
    else:
        obj_name, attr = attr.split(".", maxsplit=1)
        dot_notation_setattr(getattr(obj, obj_name), attr, value)


def hasarg(_callable: Callable, arg_name: str) -> bool:
    """Check if a function, class, or method has an argument with the specified name.

    Args:
        callable (Callable): function, class, or method to inspect.
        arg_name (str): argument name to check for.

    Returns:
        bool: `True` if the argument if the specified name exists.
    """

    args = inspect.signature(_callable).parameters.keys()
    return arg_name in args


def get_name(x: Callable, include_module_name: bool = False) -> str:
    """Get the name of an object, class or function.

    Args:
        x (Callable): object, class or function.
        include_module_name (bool, optional): whether to include the name of the module from
            which it comes. Defaults to False.

    Returns:
        str: name
    """
    name = type(x).__name__ if isinstance(x, object) else x.__name__
    if include_module_name:
        module = type(x).__module__ if isinstance(x, object) else x.__module__
        name = f"{module}.{name}"
    return name
