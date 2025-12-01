"""CLIP-style Vision-Language model.

Dual-encoder architecture that learns aligned image-text representations
through contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImageEncoder(nn.Module):
    """Vision encoder using ResNet or ViT backbone.

    Projects images to a shared embedding space.

    Args:
        backbone: Name of torchvision model ('resnet50', 'resnet18', 'vit_b_16').
        embed_dim: Output embedding dimension.
        pretrained: Use pretrained weights.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        embed_dim: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet50(weights=weights)
            self.backbone = nn.Sequential(*list(base.children())[:-1])  # Remove FC
            backbone_dim = 2048
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet18(weights=weights)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            backbone_dim = 512
        elif backbone == "vit_b_16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
            self.backbone.heads = nn.Identity()  # Remove classification head
            backbone_dim = 768
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Project to shared embedding space
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to embedding space.

        Args:
            x: Images [B, C, H, W].

        Returns:
            Image embeddings [B, embed_dim].
        """
        features = self.backbone(x)
        features = features.flatten(1)
        return self.projection(features)


class TextEncoder(nn.Module):
    """Text encoder using Transformer.

    Simple transformer encoder for text. In production, use
    HuggingFace transformers (BERT, RoBERTa, etc.) for better results.

    Args:
        vocab_size: Size of vocabulary.
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        vocab_size: int = 30522,  # BERT vocab size
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        max_length: int = 77,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode text to embedding space.

        Args:
            input_ids: Token IDs [B, L].
            attention_mask: Attention mask [B, L].

        Returns:
            Text embeddings [B, embed_dim].
        """
        B, L = input_ids.shape

        # Token + positional embeddings
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to additive mask (0 = attend, -inf = ignore)
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Pool: use [CLS] token (first token) or mean pooling
        x = x[:, 0]  # CLS token pooling

        return self.norm(x)


class CLIPModel(nn.Module):
    """CLIP-style dual encoder for vision-language learning.

    Learns to align image and text representations through contrastive learning.

    Args:
        image_encoder: Image encoder module.
        text_encoder: Text encoder module.
        embed_dim: Shared embedding dimension.
        temperature: Initial temperature for contrastive loss.
        learnable_temperature: Whether temperature is learnable.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        embed_dim: int = 512,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
    ) -> None:
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        if learnable_temperature:
            # Learnable log temperature (more stable optimization)
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("log_temperature", torch.log(torch.tensor(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode images and L2-normalize."""
        features = self.image_encoder(image)
        return F.normalize(features, dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode text and L2-normalize."""
        features = self.text_encoder(input_ids, attention_mask)
        return F.normalize(features, dim=-1)

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass computing image and text embeddings.

        Args:
            image: Images [B, C, H, W].
            input_ids: Token IDs [B, L].
            attention_mask: Attention mask [B, L].

        Returns:
            Tuple of (image_features, text_features, temperature).
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(input_ids, attention_mask)

        return image_features, text_features, self.temperature

    def compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity matrix scaled by temperature."""
        return image_features @ text_features.t() / self.temperature
