"""
This module provides utility functions for manipulating PyTorch models, such as replacing layers or loading state_dicts.
"""

from typing import Dict, List

import torch
from loguru import logger
from torch.nn import Identity, Module, Sequential

from lighter.utils.misc import setattr_dot_notation


def replace_layer_with(model: Module, layer_name: str, new_layer: Module) -> Module:
    """
    Replaces a specified layer in a PyTorch model with a new layer.

    Args:
        model (Module): The model to modify.
        layer_name (str): The name of the layer to replace,
            using dot notation if necessary (e.g. "layer10.fc.weights").
        new_layer (Module): The new layer to insert.

    Returns:
        Module: The modified model with the new layer.
    """
    setattr_dot_notation(model, layer_name, new_layer)
    return model


def replace_layer_with_identity(model: Module, layer_name: str) -> Module:
    """
    Replaces a specified layer in a PyTorch model with an Identity layer.

    Args:
        model (Module): The model to modify.
        layer_name (str): The name of the layer to replace with an Identity layer,
            using dot notation if necessary (e.g. "layer10.fc.weights").

    Returns:
        Module: The modified model with the Identity layer.
    """
    return replace_layer_with(model, layer_name, Identity())


def remove_n_last_layers_sequentially(model: Module(), num_layers=1) -> Sequential:
    """
    Removes a specified number of layers from the end of a model and returns it as a Sequential model.

    Args:
        model (Module): The model to modify.
        num_layers (int): The number of layers to remove from the end.

    Returns:
        Sequential: The modified model as a Sequential container.
    """
    return Sequential(*list(model.children())[:-num_layers])


def adjust_prefix_and_load_state_dict(
    model: Module, ckpt_path: str, ckpt_to_model_prefix: Dict[str, str] = None, layers_to_ignore: List[str] = None
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
        model (Module): The model to load the state_dict into.
        ckpt_path (str): The path to the checkpoint file.
        ckpt_to_model_prefix (Dict[str, str]): Mapping of checkpoint prefixes to model prefixes.
        layers_to_ignore (List[str]): Layers to ignore when loading the state_dict.

    Returns:
        Module: The model with the loaded state_dict.

    Raises:
        ValueError: If there is no overlap between the checkpoint's and model's state_dict.
    """

    # Load checkpoint
    ckpt = torch.load(ckpt_path)

    # Check if the checkpoint is a model's state_dict or a LighterSystem checkpoint.
    # A LighterSystem checkpoint contains the model’s entire internal state, we only need its state_dict.
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
        # Remove the "model." prefix from the checkpoint's state_dict keys. This is characteristic to LighterSystem.
        ckpt = {key.replace("model.", ""): value for key, value in ckpt.items()}

    # Adjust the keys in the checkpoint's state_dict to match the the model's state_dict's keys.
    if ckpt_to_model_prefix is not None:
        for ckpt_prefix, model_prefix in ckpt_to_model_prefix.items():
            # Add a dot at the end of the prefix if necessary.
            ckpt_prefix = ckpt_prefix if ckpt_prefix == "" or ckpt_prefix.endswith(".") else f"{ckpt_prefix}."
            model_prefix = model_prefix if model_prefix == "" or model_prefix.endswith(".") else f"{model_prefix}."
            if ckpt_prefix != "":
                # Replace ckpt_prefix with model_prefix in the checkpoint state_dict
                ckpt = {key.replace(ckpt_prefix, model_prefix): value for key, value in ckpt.items() if ckpt_prefix in key}
            else:
                # Add the model_prefix before the current key name if there's no specific ckpt_prefix
                ckpt = {f"{model_prefix}{key}": value for key, value in ckpt.items() if ckpt_prefix in key}

    # Check if the checkpoint's and model's state_dicts have no overlap.
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(ckpt.keys())
    if not set(ckpt_keys) & set(model_keys):
        raise ValueError(
            "There is no overlap between checkpoint's and model's state_dict."
            f"\nModel keys: '{model_keys[0]}', ..., '{model_keys[-1]}', "
            f"\nCheckpoint keys: '{ckpt_keys[0]}', ..., '{ckpt_keys[-1]}'"
        )

    # Remove the layers that are not to be loaded.
    if layers_to_ignore is not None:
        for layer in layers_to_ignore:
            ckpt.pop(layer)

    # Load the adjusted state_dict into the model instance.
    incompatible_keys = model.load_state_dict(ckpt, strict=False)

    # Log the incompatible keys during checkpoint loading.
    if len(incompatible_keys.missing_keys) > 0 or len(incompatible_keys.unexpected_keys) > 0:
        logger.info(f"Encountered incompatible keys during checkpoint loading. If intended, ignore.\n{incompatible_keys}")
    else:
        logger.info("Checkpoint loaded successfully.")

    return model
