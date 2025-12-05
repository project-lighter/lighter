"""Medical imaging datasets using MONAI.

MONAI provides ready-to-use medical imaging datasets including:
- DecathlonDataset: Medical Segmentation Decathlon (10 tasks)
- MedNISTDataset: Medical version of MNIST
- TciaDataset: The Cancer Imaging Archive datasets

All transforms are defined directly in the YAML config using MONAI's transform classes.
"""

from monai.apps import DecathlonDataset as DecathlonDataset
