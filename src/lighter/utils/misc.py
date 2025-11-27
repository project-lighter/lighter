"""
This module contains miscellaneous utility functions for handling lists, attributes, and function arguments.
"""

import inspect
from typing import Any, Callable

from torch.optim.optimizer import Optimizer


def ensure_list(input: Any) -> list:
    """
    Ensures that the input is wrapped in a list. If the input is None, returns an empty list.

    Args:
        input: The input to wrap in a list.

    Returns:
        List: The input wrapped in a list, or an empty list if input is None.
    """
    if isinstance(input, list):
        return input
    if isinstance(input, tuple):
        return list(input)
    if input is None:
        return []
    return [input]


def hasarg(fn: Callable, arg_name: str) -> bool:
    """
    Checks if a callable (function, method, or class) has a specific argument.

    Args:
        fn: The callable to inspect.
        arg_name: The name of the argument to check for.

    Returns:
        bool: True if the argument exists, False otherwise.
    """
    args = inspect.signature(fn).parameters.keys()
    return arg_name in args


def get_name(_callable: Callable, include_module_name: bool = False) -> str:
    """
    Retrieves the name of a callable, optionally including the module name.

    Args:
        _callable: The callable whose name to retrieve.
        include_module_name: Whether to include the module name in the result.

    Returns:
        str: The name of the callable, optionally prefixed with the module name.
    """
    # Get the name directly from the callable's __name__ attribute
    name = getattr(_callable, "__name__", type(_callable).__name__)

    if include_module_name:
        # Get the module name directly from the callable's __module__ attribute
        module = getattr(_callable, "__module__", type(_callable).__module__)
        name = f"{module}.{name}"

    return name


def get_optimizer_stats(optimizer: Optimizer) -> dict[str, float]:
    """
    Extract hyperparameters from a PyTorch optimizer.

    Collects learning rate and other key hyperparameters from each parameter group
    in the optimizer and returns them in a dictionary. Keys are formatted to show
    the optimizer type and group number (if multiple groups exist).

    Args:
        optimizer: The PyTorch optimizer to extract values from.

    Returns:
        dict[str, float]: dictionary containing optimizer hyperparameters:
            - Learning rate: "optimizer/{name}/lr[/group{N}]"
            - Momentum: "optimizer/{name}/momentum[/group{N}]" (SGD, RMSprop)
            - Beta1: "optimizer/{name}/beta1[/group{N}]" (Adam variants)
            - Beta2: "optimizer/{name}/beta2[/group{N}]" (Adam variants)
            - Weight decay: "optimizer/{name}/weight_decay[/group{N}]"

            Where [/group{N}] is only added for optimizers with multiple groups.
    """
    stats_dict = {}
    for group_idx, group in enumerate(optimizer.param_groups):
        base_key = f"optimizer/{optimizer.__class__.__name__}"

        # Add group index suffix if there are multiple parameter groups
        suffix = f"/group{group_idx + 1}" if len(optimizer.param_groups) > 1 else ""

        # Always extract learning rate (present in all optimizers)
        stats_dict[f"{base_key}/lr{suffix}"] = group["lr"]

        # Extract momentum (SGD, RMSprop)
        if "momentum" in group:
            stats_dict[f"{base_key}/momentum{suffix}"] = group["momentum"]

        # Extract betas (Adam, AdamW, NAdam, RAdam, etc.)
        if "betas" in group:
            stats_dict[f"{base_key}/beta1{suffix}"] = group["betas"][0]
            if len(group["betas"]) > 1:
                stats_dict[f"{base_key}/beta2{suffix}"] = group["betas"][1]

        # Extract weight decay if non-zero
        if "weight_decay" in group and group["weight_decay"] != 0:
            stats_dict[f"{base_key}/weight_decay{suffix}"] = group["weight_decay"]

    return stats_dict
