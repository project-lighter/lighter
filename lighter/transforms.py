from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple, Union

import SimpleITK as sitk
import numpy as np
import torch

# https://github.com/SimpleITK/SlicerSimpleFilters/blob/master/SimpleFilters/SimpleFilters.py
SITK_INTERPOLATOR_DICT = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "gaussian": sitk.sitkGaussian,
    "label_gaussian": sitk.sitkLabelGaussian,
    "bspline": sitk.sitkBSpline,
    "hamming_sinc": sitk.sitkHammingWindowedSinc,
    "cosine_windowed_sinc": sitk.sitkCosineWindowedSinc,
    "welch_windowed_sinc": sitk.sitkWelchWindowedSinc,
    "lanczos_windowed_sinc": sitk.sitkLanczosWindowedSinc,
}


class Duplicate:
    """Duplicate an input and apply two different transforms. Used for SimCLR primarily."""

    def __init__(self, transforms1: Optional[Callable] = None, transforms2: Optional[Callable] = None):
        """Duplicates an input and applies the given transformations to each copy separately.

        Args:
            transforms1 (Optional[Callable], optional): _description_. Defaults to None.
            transforms2 (Optional[Callable], optional): _description_. Defaults to None.
        """
        # Wrapped into a list if it isn't one already to allow both a
        # list of transforms as well as `torchvision.transform.Compose` transforms.
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, input: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input (torch.Tensor or any other type supported by the given transforms): Input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: a tuple of two tensors.
        """
        out1, out2 = input, deepcopy(input)
        if self.transforms1 is not None:
            out1 = self.transforms1(out1)
        if self.transforms2 is not None:
            out2 = self.transforms2(out2)
        return (out1, out2)


class MultiCrop:
    """SwaV Multi-Crop augmentation."""

    def __init__(self, high_resolution_transforms: List[Callable], low_resolution_transforms: List[Callable]):
        self.high_resolution_transforms = high_resolution_transforms
        self.low_resolution_transforms = low_resolution_transforms

    def __call__(self, input):
        high_resolution_crops = [transform(input) for transform in self.high_resolution_transforms]
        low_resolution_crops = [transform(input) for transform in self.low_resolution_transforms]
        return (high_resolution_crops, low_resolution_crops)


class SitkToTensor:
    def __init__(self, add_channel_dim: bool):
        """_summary_

        Args:
            add_channel_dim (bool):  add channel dimension to the tensor. (D)HW -> C(D)HW.
        """
        self.add_channel_dim = add_channel_dim

    def __call__(self, sitk_image):
        tensor = torch.tensor(sitk.GetArrayFromImage(sitk_image))
        return tensor.unsqueeze(0) if self.add_channel_dim else tensor


class SitkRandomSpacing:
    def __init__(
        self,
        prob: float,
        min_spacing: Union[int, List[int], Tuple[int]],
        max_spacing: Union[int, List[int], Tuple[int]],
        default_value: Union[int, float],
        tolerance: Optional[float] = None,
        interpolator: str = "linear",
    ):

        self.prob = prob
        self.min_spacing = np.array(min_spacing)
        self.max_spacing = np.array(max_spacing)
        self.default_value = default_value
        self.tolerance = None if tolerance is None else np.array(tolerance)
        self.interpolator = SITK_INTERPOLATOR_DICT[interpolator]

    def __call__(self, sitk_image):
        if torch.rand(1) > self.prob:
            return sitk_image

        current_spacing = sitk_image.GetSpacing()
        current_size = sitk_image.GetSize()

        if self.tolerance is not None:
            tolerated_min_spacing = np.array(self.min_spacing) * (1 - self.tolerance)
            tolerated_max_spacing = np.array(self.max_spacing) * (1 - self.tolerance)
        else:
            tolerated_min_spacing = self.min_spacing
            tolerated_max_spacing = self.max_spacing

        min_spacing = [max(spacings) for spacings in zip(self.min_spacing, tolerated_min_spacing)]
        max_spacing = [min(spacings) for spacings in zip(self.max_spacing, tolerated_max_spacing)]

        new_spacing = [np.random.uniform(mn, mx) for mn, mx in zip(min_spacing, max_spacing)]
        new_size = []
        for size, spacing, n_spacing in zip(current_size, current_spacing, new_spacing):
            new_size.append(int(round(size * spacing / n_spacing)))

        return sitk.Resample(
            sitk_image,
            new_size,  # size
            sitk.Transform(),  # transform
            self.interpolator,  # interpolator
            sitk_image.GetOrigin(),  # outputOrigin
            new_spacing,  # outputSpacing
            sitk_image.GetDirection(),  # outputDirection
            self.default_value,  # defaultPixelValue
            sitk_image.GetPixelID(),
        )  # outputPixelType
