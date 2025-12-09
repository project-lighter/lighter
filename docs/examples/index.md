---
title: Example Projects
---

# Example Projects

Lighter includes example projects demonstrating real-world applications across various domains. Each project is self-contained with its own config files and documentation.

## Available Projects

| Project | Domain | Description | Extra Dependencies |
|---------|--------|-------------|-------------------|
| [cifar10](https://github.com/project-lighter/lighter/tree/main/projects/cifar10) | Image Classification | Basic image classification with ResNet | - |
| [eeg](https://github.com/project-lighter/lighter/tree/main/projects/eeg) | EEG Analysis | Brain signal classification | braindecode, eegdash, mne |
| [huggingface_llm](https://github.com/project-lighter/lighter/tree/main/projects/huggingface_llm) | Text Classification | Sentiment analysis with HuggingFace Transformers | transformers, datasets |
| [lora](https://github.com/project-lighter/lighter/tree/main/projects/lora) | Parameter-Efficient Fine-Tuning | LoRA fine-tuning of language models | peft |
| [medical_segmentation](https://github.com/project-lighter/lighter/tree/main/projects/medical_segmentation) | Medical Imaging | 3D medical image segmentation | monai, itk |
| [self_supervised](https://github.com/project-lighter/lighter/tree/main/projects/self_supervised) | Self-Supervised Learning | SimCLR contrastive learning | lightly |
| [video_recognition](https://github.com/project-lighter/lighter/tree/main/projects/video_recognition) | Video Understanding | Video classification with SlowFast | pytorchvideo, av |
| [vision_language](https://github.com/project-lighter/lighter/tree/main/projects/vision_language) | Vision-Language | BLIP-2 image captioning | transformers |

## Running an Example

Each project follows the same structure:

```
projects/<name>/
├── __lighter__.py      # Project marker (enables project.* imports)
├── __init__.py
├── *.py                # Custom modules
├── configs/
│   └── *.yaml          # Experiment configs
└── README.md           # Project-specific documentation
```

To run any example:

```bash
# Clone the repo
git clone https://github.com/project-lighter/lighter.git
cd lighter

# Install Lighter
pip install -e .

# Navigate to a project
cd projects/cifar10

# Install extra dependencies if needed (check the README)
pip install <extra-deps>

# Run training
lighter fit configs/example.yaml
```

## Project Highlights

### cifar10

The simplest starting point. Demonstrates:

- Basic `LighterModule` usage
- Data augmentation in config
- Standard training workflow

### medical_segmentation

Shows how to use MONAI with Lighter for 3D medical imaging:

- 3D UNet architecture
- Medical imaging transforms
- Dice loss and metrics

### self_supervised

Contrastive learning with SimCLR:

- Custom projection heads
- Multi-view data augmentation
- NT-Xent loss

### huggingface_llm

Integrates HuggingFace Transformers:

- Tokenizer configuration
- Pre-trained model loading
- Text classification

### lora

Parameter-efficient fine-tuning:

- LoRA adapters via PEFT
- Freezing base model layers
- Memory-efficient training

## Creating Your Own Project

Use any example as a template:

1. Copy the project directory
2. Add a `__lighter__.py` marker file
3. Modify configs to point to your data and model
4. Run with `lighter fit configs/your_config.yaml`

See the [Custom Code Guide](../guides/custom-code.md) for details on project structure.
