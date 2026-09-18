# Vision-Language Learning (CLIP-style)

> **Status: Specialist reference.** This integration is outside the maintained onboarding path. A qualified replacement needs fixed-gallery, multi-positive retrieval with stable image/caption identities and a declared model/dependency profile. Start with [Compare and Continue](../experiment_comparison/README.md), or use the [download-free diagnostic](../tabular_regression/README.md). See the [project index](../README.md) for qualification scope.

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


## Required inputs and evaluation protocol

The default recipe explicitly constructs `bert-base-uncased` tokenization to match the text encoder's 30,522-token vocabulary. Install the project's `transformers` dependency and make that tokenizer available (download/cache it before an offline run). A missing tokenizer is an error; captions are never silently replaced with zero tokens.

Place three image-level manifests beside `captions.txt`: `Flickr_8k.trainImages.txt`, `Flickr_8k.devImages.txt`, and `Flickr_8k.testImages.txt`, one image filename per line. Some dataset distributions do not include these files. Supply an established split or create and retain a documented disjoint split before running; the loader will not silently reuse the entire dataset for each stage. All captions for an image stay in the same population. Available manifests are checked for overlap. Missing selected images/captions, unknown split names and corrupt images are errors.

Flickr30k's loader reads its supplied corpus; supply independently partitioned roots for separate evaluation populations. COCO uses the explicitly selected annotation file. The Flickr8k fixture checks use tiny real image files, quoted captions and distinct populations; they do not establish full training quality, retrieval benchmarks, or equivalence to the original CLIP training protocol.
