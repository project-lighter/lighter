# Example Projects

Example projects demonstrating Lighter across domains. All use real datasets and production libraries.

> **Note:** Projects in this folder are not tracked by Git. To track one, add `!projects/<name>` to `.gitignore`.

## Overview

| Project | Domain | Dataset | Libraries |
|---------|--------|---------|-----------|
| [cifar10](./cifar10/) | Image Classification | CIFAR-10 | torchvision |
| [huggingface_llm](./huggingface_llm/) | NLP | IMDB | transformers, datasets |
| [medical_segmentation](./medical_segmentation/) | Medical Imaging | Spleen CT (Decathlon) | MONAI |
| [self_supervised](./self_supervised/) | Self-Supervised | CIFAR-10 | lightly |
| [video_recognition](./video_recognition/) | Video Understanding | UCF101 | pytorchvideo, av |
| [eeg](./eeg/) | EEG Analysis | HBN-EEG (EEG 2025 Challenge) | braindecode, eegdash |
| [vision_language](./vision_language/) | Vision-Language | Flickr8k / Flickr30k | transformers |
| [efficient_finetuning](./efficient_finetuning/) | Transfer Learning | CIFAR-100 | (built-in LoRA) |

---

## Requirements

```bash
# Medical Segmentation
pip install monai

# Self-Supervised Learning
pip install lightly

# EEG (NeurIPS 2025 Challenge)
pip install eegdash braindecode mne

# Video Recognition
pip install pytorchvideo av

# HuggingFace / Vision-Language
pip install transformers datasets
```

---

## Project Details

### cifar10
Reference example - simple CNN for CIFAR-10. No extra dependencies.

### huggingface_llm
DistilBERT sentiment classification on IMDB. Uses HuggingFace transformers.

### medical_segmentation
3D U-Net spleen segmentation using MONAI. Dataset auto-downloads (~1.5 GB).

### self_supervised
SimCLR contrastive learning with lightly. Uses NT-Xent loss on augmented views.

### video_recognition
R3D and Video Transformer for UCF101 action recognition. Manual dataset download required.

### eeg
NeurIPS 2025 EEG Foundation Challenge implementation. Two regression tasks on HBN-EEG data using EEGNeX/EEGNet models.

### vision_language
CLIP-style dual-encoder for image-text alignment. Flickr8k (easy) or Flickr30k dataset.

### efficient_finetuning
LoRA parameter-efficient fine-tuning on CIFAR-100. ~1% trainable parameters.

---

## Features Demonstrated

### Callbacks
| Callback | Projects |
|----------|----------|
| FileWriter | cifar10, medical_segmentation, video_recognition |
| CsvWriter | huggingface_llm, video_recognition, efficient_finetuning |
| Freezer | vision_language, efficient_finetuning |

### Configuration Patterns
| Pattern | Projects |
|---------|----------|
| `%` raw references | cifar10, eeg, medical_segmentation, video_recognition, vision_language, efficient_finetuning |
| `$` expressions | All projects |
| `_mode_: callable` | medical_segmentation, video_recognition, vision_language |
| Config composition | video_recognition |
| Differential LRs | vision_language |

---

## Running Projects

All projects include `fast_dev_run: true` for quick testing.

```bash
cd projects/<project_name>

# Quick test (from repo root with uv)
uv run --project ../.. lighter fit configs/<config>.yaml

# Full training
lighter fit configs/<config>.yaml trainer::fast_dev_run=false

# Multi-GPU
lighter fit configs/<config>.yaml trainer::devices=4 trainer::strategy=ddp
```

## Adding Your Own Project

1. Create directory: `projects/my_project/`
2. Add `__lighter__.py` (enables `project.` imports)
3. Add `models/`, `networks/`, `configs/` as needed
4. Run: `cd projects/my_project && lighter fit configs/config.yaml`
