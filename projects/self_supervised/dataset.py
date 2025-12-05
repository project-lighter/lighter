"""Datasets for self-supervised learning using lightly library.

Lightly provides LightlyDataset and transforms optimized for SSL methods.

All dataset/transform composition is defined directly in the YAML config.
"""

from lightly.data import LightlyDataset as LightlyDataset
from lightly.transforms import SimCLRTransform as SimCLRTransform
