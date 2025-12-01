"""Neural network architectures for EEG 2025 Challenge.

Re-exports braindecode models for direct use in configs.
"""

from .eeg_networks import EEGConformer, EEGNetv4, EEGNeX, ShallowFBCSPNet

__all__ = [
    "EEGNeX",
    "EEGNetv4",
    "ShallowFBCSPNet",
    "EEGConformer",
]
