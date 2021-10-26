"""
Taken over from GANslate
"""

import SimpleITK as sitk
import torch
import numpy as np


def load(file_path):
    return sitk.ReadImage(str(file_path))


def write(sitk_image, file_path):
    sitk.WriteImage(sitk_image, str(file_path), True)  # True is for useCompression flag


def tensor_to_sitk_image(tensor, origin=None, spacing=None, direction=None, dtype='int16'):
    array = tensor.cpu().numpy().astype(dtype)
    sitk_image = sitk.GetImageFromArray(array)

    if origin is not None:
        sitk_image.SetOrigin(origin)

    if spacing is not None:
        sitk_image.SetSpacing(spacing)

    if direction is not None:
        sitk_image.SetDirection(direction)

    return sitk_image


def get_size(sitk_image):
    return sitk_image.GetSize()


def get_torch_like_size(sitk_image):
    """Get volume size in torch format - (D)HW instead of WH(D)."""
    size = get_size(sitk_image)
    if len(size) in [2, 3]:
        return size[::-1]
    else:
        raise NotImplementedError(f"Not implemented for {len(size)} dimensions.")


def get_npy(sitk_image):
    return sitk.GetArrayFromImage(sitk_image)


def get_tensor(sitk_image):
    return torch.Tensor(get_npy(sitk_image))


def is_image_smaller_than(sitk_image, target_size):
    image_size = get_torch_like_size(sitk_image)
    image_size = np.array(image_size)

    # When checking if a volume is big enough in xy only, discard the z dim.
    if len(image_size) == 3 and len(target_size) == 2:
        image_size = image_size[1:]

    if (image_size < target_size).any():
        return True
    return False


def get_npy_dtype(sitk_image):
    return str(sitk.GetArrayFromImage(sitk_image).dtype)


def slice_image(sitk_image, start=(0, 0, 0), end=(-1, -1, -1)):
    """"Returns the `sitk_image` sliced from the `start` index (x,y,z) to the `end` index.
    """
    size = sitk_image.GetSize()
    assert len(start) == len(end) == len(size)

    # replace -1 dim index placeholders with the size of that dimension
    end = [size[i] if end[i] == -1 else end[i] for i in range(len(end))]

    slice_filter = sitk.SliceImageFilter()
    slice_filter.SetStart(start)
    slice_filter.SetStop(end)
    return slice_filter.Execute(sitk_image)


def apply_mask(sitk_image,
               sitk_mask,
               masking_value,
               outside_value=0,
               set_same_origin=False,
               negated_mask=False):
    """
    Apply SimpleITK mask on a SimpleITK image.

    Parameters
    ------------------
    sitk_image: Input SimpleITK image
    sitk_mask: Input SimpleTIK mask image
    masking_value: int or float - Value to apply outside the mask
    outside_value: int or float - Value where mask is negative, like 0 
        in [0,1] or [0,255] masks. (Default: 0).
    set_same_origin: bool - Sets the mask origin to be equal to the image origin. 
        To be used only when the origin is roughly the same except for some 
        minor differences in the position of the origin. (Default: False)
    negated_mask: bool - if True, only the values where mask is 0 are masked out. 

    Returns
    ------------------
    SimpleITK image: image with the masked values replaced with `masking value`.

    """
    if set_same_origin:
        sitk_mask.SetOrigin(sitk_image.GetOrigin())

    mask_filter = sitk.MaskNegatedImageFilter() if negated_mask else sitk.MaskImageFilter()

    mask_filter.SetMaskingValue(masking_value)
    mask_filter.SetOutsideValue(outside_value)

    masked_sitk_image = mask_filter.Execute(sitk_image, sitk_mask)
    return masked_sitk_image
