# Efficient Fine-tuning with LoRA

Parameter-efficient fine-tuning using Low-Rank Adaptation.

## Dataset

**CIFAR-100** - 100 classes, 60k images. Auto-downloaded.

## Method

**LoRA** - freezes pretrained weights and adds small trainable low-rank matrices (~1% of parameters).

## Lighter Features Demonstrated

- **Built-in LoRA wrapper** - configurable rank, alpha, dropout
- **Freezer callback** - freeze backbone during training
- **`$` expressions** - filter trainable parameters: `$[p for p in @model::network.parameters() if p.requires_grad]`
- **CsvWriter** - prediction logging

## Requirements

No extra dependencies (built-in LoRA). For production, consider [PEFT](https://github.com/huggingface/peft).

## Usage

```bash
cd projects/efficient_finetuning

# Quick test
uv run --project ../.. lighter fit configs/lora.yaml

# Full training
lighter fit configs/lora.yaml trainer::fast_dev_run=false

# Different rank
lighter fit configs/lora.yaml vars::lora_rank=16 vars::lora_alpha=32
```

## LoRA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_rank` | 8 | Low-rank dimension |
| `lora_alpha` | 16 | Scaling factor (typically 2x rank) |
| `lora_dropout` | 0.1 | Regularization |

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
