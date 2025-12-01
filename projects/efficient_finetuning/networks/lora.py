"""Low-Rank Adaptation (LoRA) implementation.

LoRA enables efficient fine-tuning by adding trainable low-rank matrices
to frozen pretrained weights. This reduces trainable parameters by 100-1000x
while maintaining performance comparable to full fine-tuning.

Reference: Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation.

    Args:
        rank: Rank of the low-rank decomposition. Higher = more expressive but more params.
        alpha: Scaling factor. The adaptation is scaled by alpha/rank.
        dropout: Dropout probability for LoRA layers.
        target_modules: Names of modules to apply LoRA to (e.g., ['q_proj', 'v_proj']).
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    target_modules: list[str] | None = None

    @property
    def scaling(self) -> float:
        """Compute LoRA scaling factor."""
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    Implements: h = Wx + (BA)x * (alpha/r)
    Where W is frozen, and B (down-projection) and A (up-projection) are trainable.

    Args:
        original_layer: The original nn.Linear layer to adapt.
        rank: LoRA rank.
        alpha: LoRA alpha (scaling factor).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Freeze original weights
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        # LoRA matrices: A (down-projection) and B (up-projection)
        # W' = W + BA, where B: out x rank, A: rank x in
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Original output
        result = self.original_layer(x)

        # LoRA adaptation: (x @ A^T) @ B^T * scaling
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        result = result + lora_out * self.scaling

        return result

    def merge_weights(self) -> None:
        """Merge LoRA weights into original layer for inference efficiency."""
        with torch.no_grad():
            self.original_layer.weight.add_((self.lora_B @ self.lora_A) * self.scaling)

    def unmerge_weights(self) -> None:
        """Unmerge LoRA weights (reverse of merge)."""
        with torch.no_grad():
            self.original_layer.weight.sub_((self.lora_B @ self.lora_A) * self.scaling)


def apply_lora(
    model: nn.Module,
    config: LoRAConfig | None = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Apply LoRA to specified modules in a model.

    Args:
        model: The model to adapt.
        config: LoRA configuration (alternative to individual args).
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA dropout.
        target_modules: Module names to apply LoRA. If None, applies to all Linear layers.

    Returns:
        Model with LoRA applied.
    """
    if config is not None:
        rank = config.rank
        alpha = config.alpha
        dropout = config.dropout
        target_modules = config.target_modules

    # Find and replace target modules
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Check if this module should be adapted
            if target_modules is not None:
                if not any(t in name for t in target_modules):
                    continue

            # Get parent module and attribute name
            parts = name.rsplit(".", 1)
            if len(parts) == 1:
                parent = model
                attr_name = parts[0]
            else:
                parent = model.get_submodule(parts[0])
                attr_name = parts[1]

            # Replace with LoRA version
            lora_layer = LoRALinear(module, rank, alpha, dropout)
            setattr(parent, attr_name, lora_layer)

    return model


def get_trainable_params(model: nn.Module) -> tuple[int, int, float]:
    """Count trainable vs total parameters.

    Returns:
        Tuple of (trainable_params, total_params, percentage).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable / total if total > 0 else 0

    return trainable, total, percentage


class LoRAWrapper(nn.Module):
    """Wrapper that applies LoRA to a base model.

    This wrapper can be instantiated from config, making it easy
    to configure LoRA through YAML.

    Args:
        base_model: The pretrained model to adapt.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha (scaling factor).
        lora_dropout: LoRA dropout.
        target_modules: Which modules to apply LoRA to.
        freeze_base: Whether to freeze base model parameters.
    """

    def __init__(
        self,
        base_model: nn.Module,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        target_modules: list[str] | None = None,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()

        self.base_model = base_model

        # Freeze base model if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Apply LoRA
        apply_lora(
            self.base_model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules,
        )

        # Log parameter counts
        trainable, total, pct = get_trainable_params(self.base_model)
        print(f"LoRA applied: {trainable:,} / {total:,} params trainable ({pct:.2f}%)")

    def forward(self, *args, **kwargs):
        """Forward through adapted model."""
        return self.base_model(*args, **kwargs)

    def merge_lora(self) -> None:
        """Merge all LoRA weights for efficient inference."""
        for module in self.base_model.modules():
            if isinstance(module, LoRALinear):
                module.merge_weights()

    def unmerge_lora(self) -> None:
        """Unmerge all LoRA weights."""
        for module in self.base_model.modules():
            if isinstance(module, LoRALinear):
                module.unmerge_weights()
