"""Video models for action recognition.

Includes 3D CNN (R3D) and Video Transformer architectures.
"""

import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    """3D convolution block with batch norm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class R3DBlock(nn.Module):
    """Residual 3D block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = Conv3DBlock(in_channels, out_channels, stride=(stride, stride, stride))
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, (stride, stride, stride), bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class R3D(nn.Module):
    """R3D (3D ResNet) for video classification.

    A simplified R3D-18 style architecture.

    Args:
        num_classes: Number of action classes.
        in_channels: Number of input channels (3 for RGB).
        num_frames: Expected number of frames (for documentation).
    """

    def __init__(
        self,
        num_classes: int = 101,
        in_channels: int = 3,
        num_frames: int = 16,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [R3DBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(R3DBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Video tensor of shape [B, C, T, H, W].

        Returns:
            Class logits [B, num_classes].
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)


class PatchEmbed3D(nn.Module):
    """3D patch embedding for video."""

    def __init__(
        self,
        img_size: int = 224,
        num_frames: int = 16,
        patch_size: tuple[int, int, int] = (2, 16, 16),
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (num_frames // patch_size[0]) * (img_size // patch_size[1]) * (img_size // patch_size[2])
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = self.proj(x)  # [B, embed_dim, T', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class VideoTransformer(nn.Module):
    """Simple Video Transformer (ViViT-style) for action recognition.

    Uses tubelet embedding and standard transformer encoder.

    Args:
        num_classes: Number of action classes.
        img_size: Spatial size of input frames.
        num_frames: Number of input frames.
        patch_size: Spatiotemporal patch size (T, H, W).
        embed_dim: Transformer embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        mlp_ratio: MLP hidden dimension ratio.
    """

    def __init__(
        self,
        num_classes: int = 101,
        img_size: int = 224,
        num_frames: int = 16,
        patch_size: tuple[int, int, int] = (2, 16, 16),
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbed3D(img_size, num_frames, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional embedding + CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Video tensor [B, C, T, H, W].

        Returns:
            Class logits [B, num_classes].
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Classification from CLS token
        x = self.norm(x[:, 0])
        return self.head(x)
