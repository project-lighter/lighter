import torch


class RandomData(torch.utils.data.TensorDataset):
    """Random inputs x and targets y = sum(x), which a small network learns in seconds."""

    def __init__(self, size: int = 1024):
        x = torch.randn(size, 8)
        super().__init__(x, x.sum(dim=1, keepdim=True))
