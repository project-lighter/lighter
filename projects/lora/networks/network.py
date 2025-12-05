"""Base networks for efficient fine-tuning examples."""

import torch
import torch.nn as nn
from torchvision import models


class ImageClassifier(nn.Module):
    """Image classifier using pretrained backbone with custom head.

    This model is designed to be adapted with LoRA - the backbone is
    pretrained and the classification head matches the target task.

    Args:
        backbone: Name of torchvision model.
        num_classes: Number of output classes.
        pretrained: Use pretrained weights.
        freeze_backbone: Whether to freeze backbone initially.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 100,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        # Load backbone
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet50(weights=weights)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            backbone_dim = 2048
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet18(weights=weights)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            backbone_dim = 512
        elif backbone == "vit_b_16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
            self.backbone.heads = nn.Identity()
            backbone_dim = 768
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(backbone_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class VisionTransformerForClassification(nn.Module):
    """Vision Transformer optimized for LoRA fine-tuning.

    ViT is particularly well-suited for LoRA because:
    - Attention layers have many linear projections
    - MLP blocks are linear layers
    - LoRA can target q_proj, k_proj, v_proj, out_proj

    Args:
        num_classes: Number of output classes.
        pretrained: Use pretrained weights.
        img_size: Input image size.
    """

    def __init__(
        self,
        num_classes: int = 100,
        pretrained: bool = True,
        img_size: int = 224,
    ) -> None:
        super().__init__()

        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = models.vit_b_16(weights=weights, image_size=img_size)

        # Replace classification head
        self.vit.heads = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)
