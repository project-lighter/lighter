from typing import Any, Dict, List, Optional, Tuple, Union

import sys

import torch
import torchvision
from loguru import logger

LIGHTNING_TO_LIGHTER_STAGE = {"train": "train", "validate": "val", "test": "test"}


def parse_data(
    data: Union[
        Any,
        List[Any],
        Dict[str, Any],
        Dict[str, List[Any]],
        Dict[str, Tuple[Any]],
    ]
) -> Dict[Optional[str], Any]:
    """Parse the input data as follows:
        - If dict, go over all keys and values, unpacking list and tuples, and assigning them all
          a unique identifier based on the original key and their position if they were a list/tuple.
        - If list/tuple, enumerate them and use their position as key for each value of the list/tuple.
        - If any other type, return it as-is with the key set to 'None'. A 'None' key indicates that no
          identifier is needed because no parsing ocurred.

    Args:
        data (Union[Any, List[Any], Dict[str, Any], Dict[str, List[Any]], Dict[str, Tuple[Any]]]):
            input data to parse.

    Returns:
        Dict[Optional[str], Any]: a dict where key is either a string
            identifier or `None`, and value the parsed output.
    """
    result = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                for idx, singular in enumerate(value):
                    result[key] = f"{key}_{idx}", singular if len(value > 1) else key, singular
            else:
                result[key] = value
    elif isinstance(data, (list, tuple)):
        for idx, singular in enumerate(data):
            result[str(idx)] = singular
    else:
        result[None] = data
    return result


def check_supported_data_type(data: Any, name: str) -> None:
    """Check the input data for its type. Valid data types are:
        - torch.Tensor
        - List[torch.Tensor]
        - Tuple[torch.Tensor]
        - Dict[str, torch.Tensor]
        - Dict[str, List[torch.Tensor]]
        - Dict[str, Tuple[torch.Tensor]]

    Args:
        data (Any): input data to check
        name (str): name of the data, for identification purposes.
    """
    if isinstance(data, dict):
        is_valid = all(check_supported_data_type(elem, name) for elem in data.values())
    elif isinstance(data, (list, tuple)):
        is_valid = all(check_supported_data_type(elem, name) for elem in data)
    elif isinstance(data, torch.Tensor):
        is_valid = True
    else:
        is_valid = False

    if not is_valid:
        logger.error(
            f"`{name}` has to be a Tensor, List[Tensor], Tuple[Tensor],  Dict[str, Tensor], "
            f"Dict[str, List[Tensor]], or Dict[str, Tuple[Tensor]]. `{type(data)}` is not supported."
        )
        sys.exit()


def concatenate(outputs: Union[List[Any], Tuple[Any]]) -> Union[torch.Tensor, List[Union[str, int, float]]]:
    # List of dicts.
    if isinstance(outputs[0], dict):
        # Go over dictionaries and concatenate tensors by key.
        result = {key: concatenate([output[key] for output in outputs]) for key in outputs[0]}
    # List of lists or tuples.
    elif isinstance(outputs[0], (list, tuple)):
        # Go over lists/tuples and concatenate tensors by their position.
        result = [concatenate([output[idx] for output in outputs]) for idx in range(len(outputs[0]))]
    # List of tensors.
    elif isinstance(outputs[0], torch.Tensor):
        result = torch.cat(outputs)
    else:
        logger.error(f"Type `{type(outputs[0])}` not supported.")
        sys.exit()
    return result


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
        # Reshape 3D image from NCDHW to NC(D*H)W format
        shape = image.shape
        image = image.view(shape[0], shape[1], shape[2] * shape[3], shape[4])
    if image.shape[0] == 1:
        image = image[0]
    else:
        # If more than one image, create a grid
        nrow = image.shape[0] if has_three_dims else 8
        image = torchvision.utils.make_grid(image, nrow=nrow)
    return image
