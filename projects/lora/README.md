# LoRA (Low-Rank Adaptation)

Parameter-efficient fine-tuning using [HuggingFace PEFT](https://github.com/huggingface/peft).

## Overview

LoRA (Low-Rank Adaptation) enables efficient fine-tuning by adding small trainable low-rank matrices to frozen pretrained weights. This reduces trainable parameters by 10-100x while maintaining performance comparable to full fine-tuning.

This project uses the industry-standard **PEFT library** from HuggingFace, which provides:
- Actively maintained implementation
- Support for quantization (QLoRA)
- Built-in adapter saving, loading, and merging
- Extensive model support

## Dataset

**CIFAR-100** - 100 classes, 60k images. Auto-downloaded.

## Lighter Features Demonstrated

- **PEFT integration** - Standard LoRA via HuggingFace
- **`$` expressions** - Filter trainable parameters: `$[p for p in @model::network.parameters() if p.requires_grad]`
- **CsvWriter** - Prediction logging
- **Adapter management** - Save/load/merge adapters

## Usage

```bash
pip install lighter peft
cd projects/lora

# Train
lighter fit configs/lora.yaml

# Quick test
lighter fit configs/lora.yaml trainer::fast_dev_run=true

# Different rank (more expressive)
lighter fit configs/lora.yaml vars::lora_rank=16 vars::lora_alpha=32

# Higher dropout for regularization
lighter fit configs/lora.yaml vars::lora_dropout=0.2
```

## LoRA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_rank` | 8 | Low-rank dimension. Higher = more expressive but more params. Typical: 4-64 |
| `lora_alpha` | 16 | Scaling factor. Typically 2x rank. Controls adaptation strength |
| `lora_dropout` | 0.1 | Dropout for regularization. Increase if overfitting |

## How LoRA Works

Standard fine-tuning updates all weights W:
```
h = Wx
```

LoRA adds low-rank decomposition BA:
```
h = Wx + (BA)x * (alpha/rank)
```

Where:
- **W**: Frozen pretrained weights
- **B**: Trainable down-projection (out_features × rank)
- **A**: Trainable up-projection (rank × in_features)
- **alpha/rank**: Scaling factor

Benefits:
- 10-100x fewer trainable parameters
- Reduced memory footprint
- Faster training
- Can store multiple task-specific adapters
- Merge adapters into base model for inference

## Targeting Specific Modules

By default, PEFT applies LoRA based on model architecture. For explicit control:

```yaml
network:
  _target_: project.networks.lora.LoRAWrapper
  target_modules: ["fc"]  # ResNet: FC layers only

  # For Vision Transformers:
  # target_modules: ["query", "value"]

  # For Transformers:
  # target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

## Saving and Loading Adapters

The LoRAWrapper provides methods for adapter management:

```python
# Save adapter (small file, just LoRA weights)
model.network.save_adapter("./adapters/my_task")

# Load adapter
model.network.load_adapter("./adapters/my_task")

# Merge for deployment (single model file)
merged_model = model.network.merge_and_unload()
```

## Advanced: QLoRA

For even more memory efficiency, use quantized LoRA (QLoRA):

```bash
pip install bitsandbytes
```

Then configure quantization in your base model. See [PEFT QLoRA docs](https://huggingface.co/docs/peft/main/en/developer_guides/quantization).

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [PEFT Library](https://github.com/huggingface/peft) - HuggingFace
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Quantized LoRA
