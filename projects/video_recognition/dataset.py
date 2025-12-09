"""Video datasets for action recognition."""

import torch
from torch.utils.data import Dataset


class VideoClipDataset(Dataset):
    """Wrapper for video datasets to ensure consistent output format.

    Handles video data format conversion and resizing for different backends.

    Args:
        dataset: Base video dataset (UCF101, Kinetics, etc.).
        size: Target size (height, width) for resizing frames. If None, no resizing.
    """

    def __init__(
        self,
        dataset: Dataset,
        size: tuple[int, int] | None = None,
    ) -> None:
        self.dataset = dataset
        self.size = size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # UCF101/Kinetics return (video, audio, label)
        video, audio, label = self.dataset[idx]

        # video shape from UCF101: [T, H, W, C]
        # Convert to float and normalize to [0, 1]
        video = video.float()
        if video.max() > 1.0:
            video = video / 255.0

        # Permute to [T, C, H, W] for per-frame processing
        video = video.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]

        # Resize if size is specified
        if self.size is not None:
            video = torch.nn.functional.interpolate(
                video,
                size=self.size,
                mode="bilinear",
                align_corners=False,
            )

        # Permute to [C, T, H, W] for 3D CNN
        video = video.permute(1, 0, 2, 3)  # [T, C, H, W] -> [C, T, H, W]

        return video, label


def video_collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for video batches.

    Handles variable-length videos by padding or truncating.
    """
    videos, labels = zip(*batch, strict=True)

    # Find max temporal dimension
    max_t = max(v.shape[1] for v in videos)

    # Pad videos to same length
    padded_videos = []
    for v in videos:
        if v.shape[1] < max_t:
            pad_size = max_t - v.shape[1]
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_size))
        elif v.shape[1] > max_t:
            v = v[:, :max_t]
        padded_videos.append(v)

    return torch.stack(padded_videos), torch.tensor(labels)
