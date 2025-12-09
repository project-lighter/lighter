"""Self-supervised learning models using lightly library.

Lightly provides state-of-the-art SSL methods including SimCLR, BYOL, DINO, etc.
This module wraps lightly components for use with Lighter.
"""

import torch
import torch.nn as nn
from torchvision import models


def create_simclr_model(
    backbone: str = "resnet18",
    projection_dim: int = 128,
    hidden_dim: int = 2048,
    pretrained: bool = False,
) -> nn.Module:
    """Create a SimCLR model using lightly components.

    Args:
        backbone: Backbone architecture name.
        projection_dim: Output dimension of projection head.
        hidden_dim: Hidden dimension of projection head.
        pretrained: Use pretrained backbone weights.

    Returns:
        SimCLR model with backbone and projection head.
    """
    from lightly.models.modules import SimCLRProjectionHead

    # Create backbone
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)
        backbone_dim = 512
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet50(weights=weights)
        backbone_dim = 2048
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Remove classification head
    backbone_model = nn.Sequential(*list(base.children())[:-1], nn.Flatten())

    # Create projection head
    projection_head = SimCLRProjectionHead(backbone_dim, hidden_dim, projection_dim)

    return SimCLRNetwork(backbone_model, projection_head)


class SimCLRNetwork(nn.Module):
    """SimCLR network combining backbone and projection head."""

    def __init__(self, backbone: nn.Module, projection_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both features and projections.

        Args:
            x: Input images [B, C, H, W].

        Returns:
            Tuple of (features, projections).
        """
        features = self.backbone(x)
        projections = self.projection_head(features)
        return features, projections


def create_byol_model(
    backbone: str = "resnet18",
    projection_dim: int = 256,
    hidden_dim: int = 4096,
    pretrained: bool = False,
) -> nn.Module:
    """Create a BYOL model using lightly components.

    BYOL (Bootstrap Your Own Latent) doesn't require negative pairs.

    Args:
        backbone: Backbone architecture name.
        projection_dim: Output dimension of projection head.
        hidden_dim: Hidden dimension.
        pretrained: Use pretrained backbone weights.

    Returns:
        BYOL model.
    """
    from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead

    # Create backbone
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)
        backbone_dim = 512
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet50(weights=weights)
        backbone_dim = 2048
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    backbone_model = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
    projection_head = BYOLProjectionHead(backbone_dim, hidden_dim, projection_dim)
    prediction_head = BYOLPredictionHead(projection_dim, hidden_dim, projection_dim)

    return BYOLNetwork(backbone_model, projection_head, prediction_head)


class BYOLNetwork(nn.Module):
    """BYOL network with backbone, projection, and prediction heads."""

    def __init__(
        self,
        backbone: nn.Module,
        projection_head: nn.Module,
        prediction_head: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head
        self.prediction_head = prediction_head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            Tuple of (features, projections, predictions).
        """
        features = self.backbone(x)
        projections = self.projection_head(features)
        predictions = self.prediction_head(projections)
        return features, projections, predictions
