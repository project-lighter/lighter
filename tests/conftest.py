"""Shared test fixtures for the Lighter test suite.

This module provides common fixtures used across multiple test files to reduce
duplication and maintain consistency.
"""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_trainer():
    """Create a mock PyTorch Lightning Trainer for testing.

    Returns:
        MagicMock: A configured mock trainer with common attributes.
    """
    trainer = MagicMock()
    trainer.world_size = 1
    trainer.global_rank = 0
    trainer.is_global_zero = True
    trainer.logger = MagicMock()
    trainer.strategy.broadcast = lambda x, src: x
    trainer.strategy.barrier = MagicMock()
    trainer.predict_loop._predictions = [[]]
    trainer.predict_loop.num_dataloaders = 1

    # State flags
    trainer.training = False
    trainer.validating = False
    trainer.testing = False
    trainer.predicting = False
    trainer.sanity_checking = False

    return trainer


# ============================================================================
# Dataset Fixtures
# ============================================================================


class DummyDataset(Dataset):
    """Simple dataset returning random tensors and integer labels.

    Args:
        size: Number of samples in the dataset
        input_dim: Dimension of input tensors
        num_classes: Number of classes for labels
    """

    def __init__(self, size=32, input_dim=4, num_classes=2):
        super().__init__()
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.data = []

        for _ in range(self.size):
            x = torch.randn(input_dim)
            y = torch.randint(0, num_classes, size=()).item()
            self.data.append((x, y))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.size


@pytest.fixture
def dummy_dataset():
    """Create a simple dataset for testing.

    Returns:
        DummyDataset: Dataset with 32 samples, 4D inputs, 2 classes
    """
    return DummyDataset(size=32, input_dim=4, num_classes=2)


# ============================================================================
# Model Fixtures
# ============================================================================


class SimpleModel(nn.Module):
    """Simple feedforward model with a single linear layer.

    Args:
        in_features: Input dimension
        out_features: Output dimension
    """

    def __init__(self, in_features=4, out_features=2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def simple_model():
    """Create a simple model for testing.

    Returns:
        SimpleModel: Model with 4 input features, 2 output features
    """
    return SimpleModel(in_features=4, out_features=2)


# ============================================================================
# Helper Functions
# ============================================================================


def mock_trainer_state(trainer, training=False, validating=False, testing=False, predicting=False, sanity_checking=False):
    """Helper function to set trainer state flags for testing mode detection.

    Args:
        trainer: Mock trainer object
        training: Set training flag
        validating: Set validating flag
        testing: Set testing flag
        predicting: Set predicting flag
        sanity_checking: Set sanity_checking flag
    """
    trainer.training = training
    trainer.validating = validating
    trainer.testing = testing
    trainer.predicting = predicting
    trainer.sanity_checking = sanity_checking
