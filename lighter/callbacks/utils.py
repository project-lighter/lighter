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


def flatten_structure(
    data: Union[Any, List[Any], Dict[str, Union[Any, List[Any], Tuple[Any]]]], prefix: Optional[str] = None
) -> Dict[Optional[str], Any]:
    """
    Recursively parse nested data structures into a flat dictionary.

    This function flattens dictionaries, lists, and tuples, returning a dictionary where each key is constructed
    from the original structure's keys or list/tuple indices. The values in the output dictionary are non-container
    data types extracted from the input.

    Args:
        data (Union[Any, List[Any], Dict[str, Union[Any, List[Any], Tuple[Any]]]]):
            The input data to parse. Can be of any data type but the function is optimized
            to handle dictionaries, lists, and tuples. Nested structures are also supported.

        prefix (Optional[str]):
            A prefix used when constructing keys for the output dictionary. Useful for recursive
            calls to maintain context. Defaults to None.

    Returns:
        Dict[Optional[str], Any]:
            A flattened dictionary where keys are unique identifiers built from the original data structure,
            and values are non-container data extracted from the input.

    Example:
        input_data = {
            "a": [1, 2],
            "b": {"c": (3, 4), "d": 5}
        }
        output_data = flatten_structure(input_data)

        Expected output:
        {
            'a_0': 1,
            'a_1': 2,
            'b_c_0': 3,
            'b_c_1': 4,
            'b_d': 5
        }
    """
    result = {}
    if isinstance(data, dict):
        for key, value in data.items():
            # Recursively parse the value with an updated prefix
            sub_result = flatten_structure(value, prefix=f"{prefix}_{key}" if prefix else key)
            result.update(sub_result)
    elif isinstance(data, (list, tuple)):
        for idx, element in enumerate(data):
            # Recursively parse the element with an updated prefix
            sub_result = flatten_structure(element, prefix=f"{prefix}_{idx}" if prefix else str(idx))
            result.update(sub_result)
    else:
        # Assign the value to the result dictionary using the current prefix as its key
        result[prefix] = data
    return result


def preprocess_image(image: torch.Tensor, add_batch_dim=False) -> torch.Tensor:
    """Preprocess the image before logging it. If it is a batch of multiple images,
    it will create a grid image of them. In case of 3D, a single image is displayed
    with slices stacked vertically, while a batch of 3D images as a grid where each
    column is a different 3D image.
    Args:
        image (torch.Tensor): 2D or 3D image tensor.
        add_batch_dim (bool, optional): Whether to add a batch dimension to the input image.
            Use only when the input image does not have a batch dimension. Defaults to False.
    Returns:
        torch.Tensor: image ready for logging.
    """
    if add_batch_dim:
        image = image.unsqueeze(0)
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
