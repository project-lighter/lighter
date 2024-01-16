from typing import Any, Callable, Dict, List, Union

import inspect

from torch.optim.optimizer import Optimizer


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


def ensure_dict_schema(input_dict: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure that the input dict has the specified schema. If no value is set
    for a key in the input dict, the default value from the schema is used.
    This function supports nested dictionaries.

    Args:
        input_dict (Dict[str, Any]): The input dictionary to merge with the schema.
        schema (Dict[str, Any]): A schema dictionary with default values specified.

    Returns:
        Dict[str, Any]: The merged dictionary. If input_dict is None, returns the schema dictionary.

    Raises:
        ValueError: If the input dictionary has other keys than the specified schema keys.

    """
    output_dict = schema.copy()
    if input_dict is not None:
        for key, value in input_dict.items():
            if key not in schema:
                raise ValueError(f"Key {key} is not defined in the schema.")
            if isinstance(value, dict) and isinstance(schema[key], dict):
                output_dict[key] = ensure_dict_schema(value, schema[key])
            else:
                output_dict[key] = value
    return output_dict


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


def apply_fns(data: Any, fns: Union[Callable, List[Callable]]) -> Any:
    """Apply a function or a list of functions on the input.

    Args:
        data (Any): input to apply the function(s) on.
        fns (Union[Callable, List[Callable]]): function or list of functions to apply on the input.

    Returns:
        Any: output of the function(s).
    """
    for fn in ensure_list(fns):
        data = fn(data)
    return data


def get_optimizer_stats(optimizer: Optimizer) -> Dict[str, float]:
    """
    Extract learning rates and momentum values from each parameter group of the optimizer.

    Args:
        optimizer (Optimizer): A PyTorch optimizer.

    Returns:
        Dict[str, float]: Dictionary with formatted keys and values for learning rates and momentum.
    """
    stats_dict = {}
    for group_idx, group in enumerate(optimizer.param_groups):
        lr_key = f"optimizer/{optimizer.__class__.__name__}/lr"
        momentum_key = f"optimizer/{optimizer.__class__.__name__}/momentum"

        # Add group index to the key if there are multiple parameter groups
        if len(optimizer.param_groups) > 1:
            lr_key += f"/group{group_idx+1}"
            momentum_key += f"/group{group_idx+1}"

        # Extracting learning rate
        stats_dict[lr_key] = group["lr"]

        # Extracting momentum or betas[0] if available
        if "momentum" in group:
            stats_dict[momentum_key] = group["momentum"]
        if "betas" in group:
            stats_dict[momentum_key] = group["betas"][0]

    return stats_dict
