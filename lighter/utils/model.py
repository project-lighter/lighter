"""
This module provides utility functions for manipulating PyTorch models, such as replacing layers or loading state_dicts.
"""

from typing import List

import torch
from loguru import logger
from torch.nn import Identity, Module, Sequential

from lighter.utils.misc import setattr_dot_notation


def replace_layer_with(model: Module, layer_name: str, new_layer: Module) -> Module:
    """
    Replaces a specified layer in a PyTorch model with a new layer.

    Args:
        model: The model to modify.
        layer_name: The name of the layer to replace,
            using dot notation if necessary (e.g. "layer10.fc.weights").
        new_layer: The new layer to insert.

    Returns:
        Module: The modified model with the new layer.
    """
    setattr_dot_notation(model, layer_name, new_layer)
    return model


def replace_layer_with_identity(model: Module, layer_name: str) -> Module:
    """
    Replaces a specified layer in a PyTorch model with an Identity layer.

    Args:
        model: The model to modify.
        layer_name: The name of the layer to replace with an Identity layer,
            using dot notation if necessary (e.g. "layer10.fc.weights").

    Returns:
        Module: The modified model with the Identity layer.
    """
    return replace_layer_with(model, layer_name, Identity())


def remove_n_last_layers_sequentially(model: Module(), num_layers=1) -> Sequential:
    """
    Removes a specified number of layers from the end of a model and returns it as a Sequential model.

    Args:
        model: The model to modify.
        num_layers: The number of layers to remove from the end.

    Returns:
        Sequential: The modified model as a Sequential container.
    """
    return Sequential(*list(model.children())[:-num_layers])


def adjust_prefix_and_load_state_dict(
    model: Module,
    ckpt_path: str,
    ckpt_to_model_prefix: dict[str, str] | None = None,
    layers_to_ignore: List[str] | None = None,
) -> Module:
    """
    This function loads a state dictionary from a checkpoint file into a model using `torch.load(strict=False)`.
    It supports remapping layer names between the checkpoint and model through the `ckpt_to_model_prefix` parameter.

    This is useful when loading weights from a model that was trained as part of a larger architecture,
    where the layer names may not match the standalone version of the model.

    Before using `ckpt_to_model_prefix`, it's recommended to:
    1. Check the layer names in both the checkpoint and target model
    2. Map the mismatched prefixes accordingly

    Args:
        model: The model to load the state_dict into.
        ckpt_path: The path to the checkpoint file.
        ckpt_to_model_prefix: Mapping of checkpoint prefixes to model prefixes.
        layers_to_ignore: Layers to ignore when loading the state_dict.

    Returns:
        Module: The model with the loaded state_dict.

    Raises:
        ValueError: If there is no overlap between the checkpoint's and model's state_dict.
    """
    # Load checkpoint and handle if state_dict is nested.
    ckpt = torch.load(ckpt_path)  # nosec B614
    if "state_dict" in ckpt:
        # System has a model attribute that contains the actual model, remove the "model." prefix
        ckpt = {key.replace("model.", ""): value for key, value in ckpt["state_dict"].items()}

    # Adjust checkpoint keys based on prefix mapping
    adjusted_ckpt = {}
    if ckpt_to_model_prefix:
        for ckpt_prefix, model_prefix in ckpt_to_model_prefix.items():
            ckpt_prefix = f"{ckpt_prefix}." if ckpt_prefix and not ckpt_prefix.endswith(".") else ckpt_prefix
            model_prefix = f"{model_prefix}." if model_prefix and not model_prefix.endswith(".") else model_prefix

            if ckpt_prefix:
                adjusted_ckpt.update(
                    {key.replace(ckpt_prefix, model_prefix): value for key, value in ckpt.items() if ckpt_prefix in key}
                )
            else:
                adjusted_ckpt.update({f"{model_prefix}{key}": value for key, value in ckpt.items()})

        if not adjusted_ckpt:
            adjusted_ckpt = ckpt
    else:
        adjusted_ckpt = ckpt

    # Remove ignored layers if specified
    if layers_to_ignore:
        for layer in layers_to_ignore:
            adjusted_ckpt.pop(layer)

    # Verify overlap between model and checkpoint keys
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(adjusted_ckpt.keys())
    if not set(model_keys) & set(ckpt_keys):
        raise ValueError(
            "There is no overlap between checkpoint's and model's state_dict."
            f"\nModel keys: {model_keys[0] + ', ..., ' + model_keys[-1] if model_keys else '[]'}"
            f"\nCheckpoint keys: {ckpt_keys[0] + ', ..., ' + ckpt_keys[-1] if ckpt_keys else '[]'}"
        )
    # Load state dict and handle incompatible keys
    incompatible_keys = model.load_state_dict(adjusted_ckpt, strict=False)
    if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
        logger.info(f"Encountered incompatible keys during checkpoint loading. If intended, ignore.\n{incompatible_keys}")
    else:
        logger.info("Checkpoint loaded successfully.")

    return model
