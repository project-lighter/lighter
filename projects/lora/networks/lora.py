"""LoRA (Low-Rank Adaptation) using HuggingFace PEFT.

This module provides integration with the standard PEFT library for parameter-efficient
fine-tuning. PEFT supports LoRA, QLoRA, Prefix Tuning, Prompt Tuning, and more.

LoRA enables efficient fine-tuning by adding trainable low-rank matrices to frozen
pretrained weights, reducing trainable parameters by 10-100x while maintaining
performance comparable to full fine-tuning.

Reference:
    Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
    https://arxiv.org/abs/2106.09685

Requirements:
    pip install peft
"""

from typing import Any

import torch.nn as nn


class LoRAWrapper(nn.Module):
    """Wrapper that applies LoRA to a base model using HuggingFace PEFT.

    This wrapper uses the industry-standard PEFT library for parameter-efficient
    fine-tuning. Benefits include:
    - Actively maintained by HuggingFace
    - Supports quantization (QLoRA) for memory efficiency
    - Built-in adapter saving, loading, and merging
    - Extensive model and method support

    Args:
        base_model: The pretrained model to adapt.
        lora_rank: Rank of the low-rank decomposition. Higher = more expressive
            but more parameters. Typical values: 4-64. Default: 8.
        lora_alpha: Scaling factor. The adaptation is scaled by alpha/rank.
            Typically 2x the rank. Default: 16.
        lora_dropout: Dropout probability for LoRA layers. Default: 0.1.
        target_modules: Which modules to apply LoRA to. If None, PEFT uses
            sensible defaults based on the model architecture.
        modules_to_save: Modules to save in addition to LoRA adapters (e.g., classifier head).
        task_type: Task type for PEFT config. Options: 'SEQ_CLS', 'SEQ_2_SEQ_LM',
            'CAUSAL_LM', 'TOKEN_CLS', 'QUESTION_ANS', 'FEATURE_EXTRACTION'.
        bias: Bias training strategy. Options: 'none', 'all', 'lora_only'. Default: 'none'.
        **peft_kwargs: Additional arguments passed to LoraConfig.

    Example:
        ```yaml
        model:
          network:
            _target_: project.networks.lora.LoRAWrapper
            lora_rank: 8
            lora_alpha: 16
            lora_dropout: 0.1
            target_modules: ["query", "value"]
            base_model:
              _target_: torchvision.models.resnet50
              weights: IMAGENET1K_V2
        ```

    Note:
        - For vision models, target_modules might need to be set explicitly
        - Use modules_to_save for classifier heads that need full fine-tuning
        - QLoRA requires additional setup (BitsAndBytesConfig)
    """

    def __init__(
        self,
        base_model: nn.Module,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: list[str] | None = None,
        modules_to_save: list[str] | None = None,
        task_type: str | None = None,
        bias: str = "none",
        **peft_kwargs: Any,
    ) -> None:
        super().__init__()

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as e:
            raise ImportError(
                "PEFT library required for LoRA. Install with:\n"
                "  pip install peft\n\n"
                "For QLoRA (quantized), also install:\n"
                "  pip install bitsandbytes"
            ) from e

        # Map task_type string to TaskType enum if provided
        peft_task_type = None
        if task_type:
            peft_task_type = getattr(TaskType, task_type.upper(), None)
            if peft_task_type is None:
                valid_types = [t.name for t in TaskType]
                raise ValueError(f"Invalid task_type '{task_type}'. Valid options: {valid_types}")

        # Create LoRA config
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            task_type=peft_task_type,
            bias=bias,
            **peft_kwargs,
        )

        # Apply PEFT
        self.model = get_peft_model(base_model, config)

        # Print trainable parameters summary
        self.model.print_trainable_parameters()

    def forward(self, *args, **kwargs):
        """Forward through LoRA-adapted model."""
        return self.model(*args, **kwargs)

    def save_adapter(self, path: str) -> None:
        """Save LoRA adapter weights to disk.

        Only saves the adapter weights (small), not the full model.

        Args:
            path: Directory to save adapter weights.
        """
        self.model.save_pretrained(path)
        print(f"LoRA adapter saved to: {path}")

    def load_adapter(self, path: str, adapter_name: str = "default") -> None:
        """Load a LoRA adapter from disk.

        Args:
            path: Directory containing adapter weights.
            adapter_name: Name for the adapter. Default: "default".
        """
        self.model.load_adapter(path, adapter_name)
        print(f"LoRA adapter loaded from: {path}")

    def merge_and_unload(self) -> nn.Module:
        """Merge LoRA weights into base model for efficient inference.

        After merging, the model no longer has separate adapter weights.
        This is useful for deployment where you want a single model file.

        Returns:
            Base model with LoRA weights merged in.
        """
        return self.model.merge_and_unload()

    def get_nb_trainable_parameters(self) -> tuple[int, int, float]:
        """Get trainable parameter statistics.

        Returns:
            Tuple of (trainable_params, all_params, percentage).
        """
        return self.model.get_nb_trainable_parameters()

    @property
    def base_model(self) -> nn.Module:
        """Access the underlying base model."""
        return self.model.get_base_model()
