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
