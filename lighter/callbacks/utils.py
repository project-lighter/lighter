from typing import Any, Dict, List, Optional, Tuple, Union

import sys

import torch
import torchvision
from loguru import logger


def get_lighter_mode(lightning_stage: str) -> str:
    """Converts the name of a PyTorch Lightnig stage to the name of its corresponding Lighter mode.

    Args:
        lightning_stage (str): stage in which PyTorch Lightning Trainer is. Can be accessed using `trainer.state.stage`.

    Returns:
        str: name of the Lighter mode.
    """
    lightning_to_lighter = {"train": "train", "validate": "val", "test": "test"}
    return lightning_to_lighter[lightning_stage]


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
                for idx, element in enumerate(value):
                    result[f"{key}_{idx}" if len(value) > 1 else key] = element
            else:
                result[key] = value
    elif isinstance(data, (list, tuple)):
        for idx, element in enumerate(data):
            result[str(idx)] = element
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

    return is_valid


def structure_preserving_concatenate(
    inputs: Union[List[Any], Tuple[Any]]
) -> Union[torch.Tensor, List[Union[str, int, float]]]:
    """Recursively concatenate tensors that are either on their own or inside of other data structures (list/tuple/dict).
    An input list of tensors is reduced to a single concatenated tensor, while an input list of data structures with tensors
    will be reduced to a single data structure with its tensors concatenated along the key or position.

    Assumes that all elements of the input list have the same type and structure.

    Args:
        inputs (Union[List[Any], Tuple[Any]]): A list or tuple of either:
            - Dictionaries, each containing tensors to be concatenated by key.
            - Lists/tuples, each containing tensors to be concatenated by their position.
            - Tensors, which are concatenated along the first dimension.

    Returns:
        Union[torch.Tensor, List[Union[str, int, float]]]: The concatenated result in the same format as the input's elements.
    """
    # List of dicts.
    if isinstance(inputs[0], dict):
        # Go over dictionaries and concatenate tensors by key.
        keys = inputs[0].keys()
        result = {key: structure_preserving_concatenate([input[key] for input in inputs]) for key in keys}
    # List of lists or tuples.
    elif isinstance(inputs[0], (list, tuple)):
        # Go over lists/tuples and concatenate tensors by their position.
        result = [structure_preserving_concatenate([input[idx] for input in inputs]) for idx in range(len(inputs[0]))]
    # List of tensors.
    elif isinstance(inputs[0], torch.Tensor):
        result = torch.cat(inputs)
    else:
        logger.error(f"Type `{type(inputs[0])}` not supported.")
        sys.exit()
    return result


def preprocess_image(image: torch.Tensor) -> torch.Tensor:
    """Preprocess the image before logging it. If it is a batch of multiple images,
    it will create a grid image of them. In case of 3D, a single image is displayed
    with slices stacked vertically, while a batch of 3D images as a grid where each
    column is a different 3D image.
    Args:
        image (torch.Tensor): 2D or 3D image tensor.
    Returns:
        torch.Tensor: image ready for logging.
    """
    image = image.detach().cpu()
    # If 3D (BCDHW), concat the images vertically and horizontally.
    if image.ndim == 5:
        # BCDHW -> BC(D*H)W. Combine slices of a 3D images vertically into a single 2D image.
        shape = image.shape
        image = image.view(shape[0], shape[1], shape[2] * shape[3], shape[4])
        # BCDHW -> 1CDH(B*W). Concat images in the batch horizontally, and unsqueeze to add back the B dim.
        image = torch.cat([*image], dim=-1).unsqueeze(0)
    # If only one image in the batch, select it and return it. Same happens when the images are 3D as they
    # are combined into a single image. `make_grid` is called when a batch of multiple 2D image is provided.
    return image[0] if image.shape[0] == 1 else torchvision.utils.make_grid(image, nrow=8)
