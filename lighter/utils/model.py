import torch
from torch.nn import Identity, Module, Sequential

from lighter.utils.misc import dot_notation_setattr


def reshape_pred_if_single_value_prediction(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
