# HuggingFace Text Classification

> **Status: Legacy text integration.** This source is not a qualified text research workflow. A replacement needs held-out evaluation, explicit token/mask and fitting policies, and an actual dependency/model profile. The current pilot does not certify pretrained model acquisition or this integration. Start with [Compare and Continue](../experiment_comparison/README.md), or use the [download-free diagnostic](../tabular_regression/README.md). See the [project index](../README.md) for qualification scope.

DistilBERT sentiment classification on IMDB reviews.

## Dataset

**IMDB** - 50,000 movie reviews, binary sentiment. Auto-downloaded via HuggingFace `datasets`.

## Architecture

**DistilBERT** - 66M parameters, 6 layers. Pre-trained on Wikipedia + BookCorpus.

## Lighter Features Demonstrated

- **HuggingFace integration** - `transformers` and `datasets`
- **Model-computed loss** - no criterion config needed
- **`vars::` section** - reusable config values
- **CsvWriter callback** - prediction logging

## Usage

```bash
pip install lighter transformers datasets
cd projects/huggingface_llm

# Train
lighter fit configs/imdb.yaml

# Quick test
lighter fit configs/imdb.yaml trainer::fast_dev_run=true

# Different model
lighter fit configs/imdb.yaml vars::pretrained_model=bert-base-uncased
```

## References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
