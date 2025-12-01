# HuggingFace Text Classification

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

## Requirements

```bash
pip install transformers datasets
```

## Usage

```bash
cd projects/huggingface_llm

# Quick test (limited batches)
uv run --project ../.. lighter fit configs/imdb.yaml

# Full training
lighter fit configs/imdb.yaml trainer::limit_train_batches=null trainer::limit_val_batches=null

# Different model
lighter fit configs/imdb.yaml vars::pretrained_model=bert-base-uncased
```

## References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
