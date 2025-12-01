"""EEG neural network architectures using braindecode.

This module re-exports braindecode models for use in configs.

Models:
- EEGNeX: State-of-the-art architecture (recommended baseline)
- EEGNetv4: Compact CNN designed for BCI
- ShallowFBCSPNet: Inspired by Filter Bank CSP
- EEGConformer: CNN + Transformer hybrid

Reference:
- EEG 2025 Challenge: https://eeg2025.github.io/
- Braindecode: https://braindecode.org/
"""

from braindecode.models import EEGConformer, EEGNetv4, EEGNeX, ShallowFBCSPNet

__all__ = [
    "EEGNeX",
    "EEGNetv4",
    "ShallowFBCSPNet",
    "EEGConformer",
]
