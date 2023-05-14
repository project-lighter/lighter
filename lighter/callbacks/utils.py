from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchvision


def get_lighter_mode(lightning_stage: str) -> str:
    """Converts the name of a PyTorch Lightnig stage to the name of its corresponding Lighter mode.

    Args:
        lightning_stage (str): stage in which PyTorch Lightning Trainer is. Can be accessed using `trainer.state.stage`.

    Returns:
        str: name of the Lighter mode.
    """
    lightning_to_lighter = {"train": "train", "validate": "val", "test": "test"}
    return lightning_to_lighter[lightning_stage]


def is_data_type_supported(data: Union[Any, List[Any], Dict[str, Union[Any, List[Any], Tuple[Any]]]]) -> bool:
    """
    Check the input data recursively for its type. Valid data types are:
        - torch.Tensor
        - List[torch.Tensor]
        - Tuple[torch.Tensor]
        - Dict[str, torch.Tensor]
        - Dict[str, List[torch.Tensor]]
        - Dict[str, Tuple[torch.Tensor]]
        - Nested combinations of the above

    Args:
        data (Union[Any, List[Any], Dict[str, Union[Any, List[Any], Tuple[Any]]]]): Input data to check.

    Returns:
        bool: True if the data type is supported, False otherwise.
    """
    if isinstance(data, dict):
        is_valid = all(is_data_type_supported(elem) for elem in data.values())
    elif isinstance(data, (list, tuple)):
        is_valid = all(is_data_type_supported(elem) for elem in data)
    elif isinstance(data, torch.Tensor):
        is_valid = True
    else:
        is_valid = False
    return is_valid


def parse_data(
    data: Union[Any, List[Any], Dict[str, Union[Any, List[Any], Tuple[Any]]]], prefix: Optional[str] = None
) -> Dict[Optional[str], Any]:
    """
    Parse the input data recursively, handling nested dictionaries, lists, and tuples.

    This function will recursively parse the input data, unpacking nested dictionaries, lists, and tuples. The result
    will be a dictionary where each key is a unique identifier reflecting the data's original structure (dict keys
    or list/tuple positions) and each value is a non-container data type from the input data.

    Args:
        data (Union[Any, List[Any], Dict[str, Union[Any, List[Any], Tuple[Any]]]]): Input data to parse.
        prefix (Optional[str]): Current prefix for keys in the result dictionary. Defaults to None.

    Returns:
        Dict[Optional[str], Any]: A dictionary where key is either a string identifier or `None`, and value is the parsed output.

    Example:
        input_data = {
            "a": [1, 2],
            "b": {"c": (3, 4), "d": 5}
        }
        output_data = parse_data(input_data)
        # Output:
        # {
        #     'a_0': 1,
        #     'a_1': 2,
        #     'b_c_0': 3,
        #     'b_c_1': 4,
        #     'b_d': 5
        # }
    """
    result = {}
    if isinstance(data, dict):
        for key, value in data.items():
            # Recursively parse the value with an updated prefix
            sub_result = parse_data(value, prefix=f"{prefix}_{key}" if prefix else key)
            result.update(sub_result)
    elif isinstance(data, (list, tuple)):
        for idx, element in enumerate(data):
            # Recursively parse the element with an updated prefix
            sub_result = parse_data(element, prefix=f"{prefix}_{idx}" if prefix else str(idx))
            result.update(sub_result)
    else:
        # Assign the value to the result dictionary using the current prefix as its key
        result[prefix] = data
    return result


def gather_tensors(
    inputs: Union[List[Union[torch.Tensor, List, Tuple, Dict]], Tuple[Union[torch.Tensor, List, Tuple, Dict]]]
) -> Union[List, Dict]:
    """Recursively gather tensors. Tensors can be standalone or inside of other data structures (list/tuple/dict).
    An input list of tensors is returned as-is. Given an input list of data structures with tensors, this function
    will gather all tensors into a list and save it under a single data structure. Assumes that all elements of
    the input list have the same type and structure.

    Args:
        inputs (List[Union[torch.Tensor, List, Tuple, Dict]], Tuple[Union[torch.Tensor, List, Tuple, Dict]]):
            They can be:
            - List/Tuples of Dictionaries, each containing tensors to be gathered by their key.
            - List/Tuples of Lists/tuples, each containing tensors to be gathered by their position.
            - List/Tuples of Tensors, returned as-is.
            - Nested versions of the above.
            The input data structure must be the same for all elements of the list. They can be arbitrarily nested.

    Returns:
        Union[List, Dict]: The gathered tensors.
    """
    # List of dicts.
    if isinstance(inputs[0], dict):
        keys = inputs[0].keys()
        return {key: gather_tensors([input[key] for input in inputs]) for key in keys}
    # List of lists or tuples.
    elif isinstance(inputs[0], (list, tuple)):
        return [gather_tensors([input[idx] for input in inputs]) for idx in range(len(inputs[0]))]
    # List of tensors.
    elif isinstance(inputs[0], torch.Tensor):
        return inputs
    else:
        raise TypeError(f"Type `{type(inputs[0])}` not supported.")


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
        shape = image.shape
        # BCDHW -> BC(D*H)W. Combine slices of a 3D images vertically into a single 2D image.
        image = image.view(shape[0], shape[1], shape[2] * shape[3], shape[4])
        # BCDHW -> 1CDH(B*W). Concat images in the batch horizontally, and unsqueeze to add back the B dim.
        image = torch.cat([*image], dim=-1).unsqueeze(0)
    # If only one image in the batch, select it and return it. Same happens when the images are 3D as they
    # are combined into a single image. `make_grid` is called when a batch of multiple 2D image is provided.
    return image[0] if image.shape[0] == 1 else torchvision.utils.make_grid(image, nrow=8)
