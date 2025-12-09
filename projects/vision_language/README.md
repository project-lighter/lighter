# Vision-Language Learning (CLIP-style)

Dual-encoder for learning aligned image-text representations.

## Task

Learn joint embeddings for images and text for retrieval and zero-shot classification.

## Dataset

**Flickr8k** (default) - 8,000 images with 5 captions each
- Download from Kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k
- Extract to `.datasets/flickr8k/`

**Flickr30k** (larger) - 31,000 images with 5 captions each
- Request access: https://shannon.cs.illinois.edu/DenotationGraph/

## Architecture

- **Image Encoder**: ResNet-50 with projection head
- **Text Encoder**: Transformer with learned embeddings
- **Loss**: Symmetric contrastive loss with learnable temperature

## Lighter Features Demonstrated

- **Freezer callback** - freeze image encoder backbone during warmup
- **Differential learning rates** - lower LR for pretrained, higher for new layers
- **`_mode_: callable`** for custom collate function
- **`$` expressions** for parameter group filtering

## Usage

```bash
pip install lighter transformers
cd projects/vision_language

# Train
lighter fit configs/clip.yaml

# Quick test
lighter fit configs/clip.yaml trainer::fast_dev_run=true
```

## References

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
