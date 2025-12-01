"""EEG 2025 Challenge project for Lighter framework.

This project implements solutions for both challenges of the NeurIPS 2025
EEG Foundation Challenge using the HBN-EEG dataset.

Challenge 1: Cross-Task Transfer Learning
- Predict response time from Contrast Change Detection (CCD) EEG

Challenge 2: Externalizing Factor Prediction
- Predict psychopathology scores from EEG recordings

References:
- Challenge website: https://eeg2025.github.io/
- Starter kit: https://github.com/eeg2025/startkit
- Paper: https://arxiv.org/abs/2506.19141
"""

from . import data, models, networks

__all__ = ["data", "models", "networks"]
