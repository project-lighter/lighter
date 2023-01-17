import importlib
import inspect
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
import torchvision
from loguru import logger
from torch.nn import Identity, Module, Sequential
from torch.utils.data import DataLoader

class ImportProject:
    """Given the path to a module, import it, and name it as specified.

    """
    def __init__(self, module_path: str) -> None:
        """
        Args:
        module_path (str): path to the module to load.
        """
   
        # Based on https://stackoverflow.com/a/41595552.
        module_path = Path(module_path).resolve() / "__init__.py"
        if not module_path.is_file():
            logger.error(f"No `__init__.py` in `{module_path}`. Exiting.")
            sys.exit()
        spec = importlib.util.spec_from_file_location("project", str(module_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["project"] = module
        logger.info(f"{module_path.parent} imported as 'project' module.")


def import_attr(module_attr: str) -> Any:
    """Import using dot-notation string, e.g., 'torch.nn.Module'.

    Args:
        module_attr (str): dot-notation path to the attribute.

    Returns:
        Any: imported attribute.
    """
    # Split module from attribute name
    module, attr = module_attr.rsplit(".", 1)
    # Import the module
    module = __import__(module, fromlist=[attr])
    # Get the attribute from the module
    return getattr(module, attr)


def hasarg(callable: Callable, arg_name: str) -> bool:
    """Check if a function, class, or method has an argument with the specified name.

    Args:
        callable (Callable): function, class, or method to inspect.
        arg_name (str): argument name to check for.

    Returns:
        bool: `True` if the argument if the specified name exists.
    """

    args = inspect.signature(callable).parameters.keys()
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


def wrap_into_list(x: Any) -> List:
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


def collate_fn_replace_corrupted(batch: torch.Tensor, dataset: DataLoader) -> torch.Tensor:
    """Collate function that allows to replace corrupted examples in the batch.
    The dataloader should return `None` when that occurs.
    The `None`s in the batch are replaced with other, randomly-selected, examples.

    Args:
        batch (torch.Tensor): batch from the DataLoader.
        dataset (Dataset): dataset that the DataLoader is passing through. Needs to be fixed
            in place with functools.partial before passing it to DataLoader's 'collate_fn' option
            as 'collate_fn' should only have a single argument - batch. Example:
                ```
                collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=dataset)`
                loader = DataLoader(dataset, ..., collate_fn=collate_fn).
                ```
    Returns:
        torch.Tensor: batch with new examples instead of corrupted ones.
    """
    # Idea from https://stackoverflow.com/a/57882783
    original_batch_len = len(batch)
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    filtered_batch_len = len(batch)
    # Num of corrupted examples
    num_corrupted = original_batch_len - filtered_batch_len
    if num_corrupted > 0:
        # Replace a corrupted example with another randomly selected example
        batch.extend([dataset[random.randint(0, len(dataset))] for _ in range(num_corrupted)])
        # Recursive call to replace the replacements if they are corrupted
        return collate_fn_replace_corrupted(batch, dataset)
    # Finally, when the whole batch is fine, return it
    return torch.utils.data.dataloader.default_collate(batch)


def preprocess_image(image: torch.Tensor) -> torch.Tensor:
    """Preprocess the image before logging it. If it is a batch of multiple images,
    it will create a grid image of them. In case of 3D, a single image is displayed
    with slices stacked vertically, while a batch as a grid where each column is
    a different 3D image.

    Args:
        image (torch.Tensor): 2D or 3D image tensor.

    Returns:
        torch.Tensor: image ready for logging.
    """
    image = image.detach().cpu()
    # 3D image (NCDHW)
    has_three_dims = image.ndim == 5
    if has_three_dims:
        shape = image.shape
        image = image.view(shape[0], shape[1], shape[2] * shape[3], shape[4])
    # If more than one image, create a grid
    if image.shape[0] > 1:
        nrow = image.shape[0] if has_three_dims else 8
        image = torchvision.utils.make_grid(image, nrow=nrow)
    return image


def debug_message(mode: str, input: torch.Tensor, target: torch.Tensor, pred: torch.Tensor,
                  metrics: Dict, loss: torch.Tensor) -> None:
    """Logs the debug message.

    Args:
        mode (str): the mode of the system.
        input (torch.Tensor): input.
        target (torch.Tensor): target.
        pred (torch.Tensor): prediction.
        metrics (Dict): a dict where keys are the metric names and values the measured values.
        loss (torch.Tensor): calculated loss.
    """
    msg = f"\n----------- Debugging Output -----------\nMode: {mode}"
    for name, data in {"input": input, "target": target, "pred": pred}.items():
        if isinstance(data, list):
            msg += f"\n\n{name.capitalize()} is a {type(data).__name__} of {len(data)} elements."
            for idx, tensor in enumerate(data):
                if is_tensor_debug_loggable(tensor):
                    msg += f"\nTensor {idx} shape and value:\n{tensor.shape}\n{tensor}"
                else:
                    msg += f"\n*Tensor {idx} is too big to log."
        else:
            if is_tensor_debug_loggable(data):
                msg += f"\n\n{name.capitalize()} tensor shape and value:\n{data.shape}\n{data}"
            else:
                msg += f"\n\n*{name.capitalize()} tensor is too big to log."
    msg += f"\n\nLoss:\n{loss}"
    msg += f"\n\nMetrics:\n{metrics}"
    logger.debug(msg)


def is_tensor_debug_loggable(tensor):
    """A tensor is loggable for debugging if its shape is smaller than 16 in each axis."""
    return (torch.tensor(tensor.shape[1:]) < 16).all()


def reshape_pred_if_single_value_prediction(pred: torch.Tensor,
                                            target: torch.Tensor) -> torch.Tensor:
    """When the task is to predict a single value, pred and target dimensions often
    mismatch - dataloader returns the value in the (B) shape, while the network
    returns predictions in the (B, 1) shape, where the second dim is redundant.

    Args:
        pred (torch.Tensor): predicted tensor.
        target (torch.Tensor): target tensor.

    Returns:
        torch.Tensor: reshaped predicted tensor if that was necessary.
    """
    if isinstance(pred, torch.Tensor) and target is not None:
        if len(pred.shape) == 2 and len(target.shape) == 1 == pred.shape[1]:
            return pred.flatten()
    return pred


def dot_notation_setattr(obj: Callable, attr: str, value: Any):
    """Set object's attribute. May use dot notation.

    Args:
        obj (Callable): object.
        attr (str): attribute name of the object.
        value (Any): attribute value to be set.
    """
    if '.' not in attr:
        if not hasattr(obj, attr):
            logger.info(f"`{get_name(obj, True)}` has no attribute `{attr}`.")
            sys.exit()
        setattr(obj, attr, value)
    # Solve recursively if the attribute is defined in dot-notation
    else:
        obj_name, attr = attr.split('.', maxsplit=1)
        dot_notation_setattr(getattr(obj, obj_name), attr, value)


def replace_layer_with(model: Module, layer_name: str, new_layer: Module) -> Module:
    """Replaces the specified layer of the network with another layer.

    Args:
        model (Module): PyTorch model to be edited
        layer_name (string): Name of the layer which will be replaced.
            Dot-notation supported, e.g. "layer10.fc".

    Returns:
        Module: PyTorch model with the new layer set at the specified location.
    """
    dot_notation_setattr(model, layer_name, new_layer)
    return model


def replace_layer_with_identity(model: Module, layer_name: str) -> Module:
    """Replaces any layer of the network with an Identity layer.
    Useful for removing the last layer of a network to be used as a backbone
    of an SSL model.

    Args:
        model (Module): PyTorch model to be edited
        layer_name (string): Name of the layer which will be replaced with an
            Identity function. Dot-notation supported, e.g. "layer10.fc".

    Returns:
        Module: PyTorch model with Identity layer at the specified location.
    """
    return replace_layer_with(model, layer_name, Identity())


def remove_last_layer_sequentially(model: Module()) -> Sequential:
    """Removes the last layer of a network and returns it as an nn.Sequential model.
    Useful when a network is to be used as a backbone of an SSL model.

    Args:
        model (Module): PyTorch model object.

    Returns:
        Sequential: PyTorch Sequential model with the last layer removed.
    """
    return Sequential(*list(model.children())[:-1])
