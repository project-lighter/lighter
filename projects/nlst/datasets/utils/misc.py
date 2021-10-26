import SimpleITK as sitk
import torch


def get_bounding_box_of_mask(mask):
    """Calculates the bounding box over a mask and returns the start and end index of it."""
    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(mask)
    bounding_box = lsif.GetBoundingBox(1)
    # First three values are the starting index, the other three are the size of the box
    start = bounding_box[:3]
    # Obtain the end index by summing up the start index and the size of the bounding box for each coordinate
    end = bounding_box[3:]
    end = tuple([sum(pair) for pair in zip(start, end)])
    return start, end


def pad(volume, target_shape):
    # Squeeze the channels
    volume = volume.squeeze(0)
    assert len(target_shape) == len(volume.shape)
    # By default no padding
    pad_width = []

    for dim in range(len(target_shape)):
        if target_shape[dim] > volume.shape[dim]:
            pad_total = target_shape[dim] - volume.shape[dim]
            pad_per_side = pad_total // 2
            pad_width.append(pad_per_side)
            pad_width.append(pad_total % 2 + pad_per_side)
        else:
            pad_width.extend([0, 0])
    # functional.pad() pads from last dimension to first, reversing the order of padding
    pad_width = pad_width[::-1]
    volume = torch.nn.functional.pad(volume, pad_width, 'constant', value=volume.min())
    return volume.unsqueeze(0)
