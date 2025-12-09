"""Custom writer functions for video recognition outputs."""

from pathlib import Path

import torch


def write_video_mp4(path: Path, tensor: torch.Tensor, *, suffix: str = ".mp4", fps: int = 15) -> None:
    """Write a video tensor to MP4 format.

    Args:
        path: Output path (suffix will be replaced).
        tensor: Video tensor of shape (T, H, W, C) with values in [0, 255] as uint8,
                or (C, T, H, W) / (T, C, H, W) with float values in [0, 1].
        suffix: File suffix (default: .mp4).
        fps: Frames per second for output video.
    """
    import torchvision.io

    path = path.with_suffix(suffix)

    # Handle different tensor layouts
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()

        # Determine layout and convert to (T, H, W, C) for torchvision
        if tensor.ndim == 5:
            # (B, C, T, H, W) -> take first batch
            tensor = tensor[0]

        if tensor.ndim == 4:
            # Could be (C, T, H, W) or (T, H, W, C) or (T, C, H, W)
            if tensor.shape[0] == 3:
                # (C, T, H, W) -> (T, H, W, C)
                tensor = tensor.permute(1, 2, 3, 0)
            elif tensor.shape[1] == 3:
                # (T, C, H, W) -> (T, H, W, C)
                tensor = tensor.permute(0, 2, 3, 1)
            # else assume already (T, H, W, C)

        # Convert to uint8 if float
        if tensor.dtype in (torch.float32, torch.float64, torch.float16):
            tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)

    torchvision.io.write_video(str(path), tensor, fps=fps)
