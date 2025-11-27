<div align="center">
  <img alt="Lighter logo" src="assets/images/lighter.png" width="80%">
</div>
<br/><br/>
<p align="center">
  <a href="https://github.com/project-lighter/lighter/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/project-lighter/lighter/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://codecov.io/gh/project-lighter/lighter"><img alt="Coverage" src="https://codecov.io/gh/project-lighter/lighter/branch/main/graph/badge.svg"></a>
  <a href="https://pypi.org/project/lighter/"><img alt="PyPI" src="https://img.shields.io/pypi/v/lighter"></a>
  <a href="https://github.com/project-lighter/lighter/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <a href="https://project-lighter.github.io/lighter"><img alt="Documentation" src="https://img.shields.io/badge/docs-latest-olive"></a>
  <a href="https://discord.gg/zJcnp6KrUp"><img alt="Discord" src="https://dcbadge.limes.pink/api/server/https://discord.gg/zJcnp6KrUp?style=flat"></a>
</p>
<br/>

**Lighter** makes PyTorch Lightning experiments reproducible and composable through YAML configuration. Stop hardcoding hyperparameters—configure everything from the command line.

## Why Lighter?

You're already using PyTorch Lightning. But every experiment requires editing Python code to change hyperparameters:

```python
# Want to try a different learning rate? Edit the code.
optimizer = Adam(params, lr=0.001)  # Change this line

# Want to use a different batch size? Edit the code.
train_loader = DataLoader(dataset, batch_size=32)  # And this one

# Want to train longer? Edit the code again.
trainer = Trainer(max_epochs=10)  # And this one too
```

**With Lighter, configure everything in YAML and override from the CLI:**

```bash
# Try different learning rates without touching code
lighter fit config.yaml model::optimizer::lr=0.001
lighter fit config.yaml model::optimizer::lr=0.01
lighter fit config.yaml model::optimizer::lr=0.1

# Every experiment is reproducible - just version control your configs
```

## Quick Start

```bash
pip install lighter
```

**Use your existing PyTorch Lightning code:**

```python
# model.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

class MyModel(pl.LightningModule):
    def __init__(self, network, learning_rate=0.001):
        super().__init__()
        self.network = network
        self.lr = learning_rate

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.network(x), y)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
```

**Configure in YAML instead of hardcoding:**

```yaml
# config.yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10

model:
  _target_: model.MyModel
  network:
    _target_: torchvision.models.resnet18
    num_classes: 10
  learning_rate: 0.001

data:
  _target_: lighter.LighterDataModule
  train_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 32
    dataset:
      _target_: torchvision.datasets.CIFAR10
      root: ./data
      train: true
      download: true
```

**Run and iterate fast:**

```bash
# Run your experiment
lighter fit config.yaml

# Try different hyperparameters - no code editing needed
lighter fit config.yaml model::learning_rate=0.01
lighter fit config.yaml trainer::max_epochs=50
lighter fit config.yaml data::train_dataloader::batch_size=64

# Use multiple GPUs
lighter fit config.yaml trainer::devices=4

# Every run creates timestamped outputs with saved configs
# outputs/2025-11-21/14-30-45/config.yaml  # Fully reproducible
```

## Key Benefits

- **Reproducible**: Every experiment = one YAML file. Version control configs like code.
- **Fast iteration**: Override any parameter from CLI without editing code.
- **Zero lock-in**: Works with any PyTorch Lightning module. Your code, your logic.
- **Composable**: Merge configs, create recipes, share experiments as files.
- **Organized**: Automatic timestamped output directories with saved configs.
- **Simple**: ~500 lines of code. Read the framework in 30 minutes.

## Optional: Use LighterModule for Less Boilerplate

If you want automatic optimizer configuration and dual logging (step + epoch), use `LighterModule`:

```python
from lighter import LighterModule

class MyModel(LighterModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)

        if self.train_metrics:
            self.train_metrics(pred, y)

        return {"loss": loss}  # Framework logs automatically

    # validation_step, test_step, predict_step...
```

```yaml
model:
  _target_: model.MyModel
  network:
    _target_: torchvision.models.resnet18
    num_classes: 10
  criterion:
    _target_: torch.nn.CrossEntropyLoss
  optimizer:
    _target_: torch.optim.Adam
    params: "$@model::network.parameters()"
    lr: 0.001
  train_metrics:
    - _target_: torchmetrics.Accuracy
      task: multiclass
      num_classes: 10
```

**LighterModule gives you:**
- Automatic `configure_optimizers()` handling
- Automatic dual logging (step + epoch)
- Config-driven criterion and metrics

**But you still control:**
- All step implementations
- Loss computation logic
- When to call metrics

## Example: Running a Hyperparameter Sweep

```bash
# Run grid search without editing code
for lr in 0.001 0.01 0.1; do
  for bs in 32 64 128; do
    lighter fit config.yaml \
      model::optimizer::lr=$lr \
      data::train_dataloader::batch_size=$bs
  done
done

# Each run saved in outputs/YYYY-MM-DD/HH-MM-SS/ with config.yaml
# Compare experiments by diffing configs
```

## Documentation

- 📚 [Get Started Tutorial](https://project-lighter.github.io/lighter/tutorials/get-started/) - 15 min walkthrough
- ⚙️ [Configuration Guide](https://project-lighter.github.io/lighter/how-to/configuration/) - Master the syntax
- 🎯 [LighterModule Design](https://project-lighter.github.io/lighter/design/model/) - Understand the internals
- 🏗️ [Architecture Overview](https://project-lighter.github.io/lighter/design/overview/) - How it all works

## Real-World Usage

- 🏥 [Foundation Models for Cancer Imaging](https://aim.hms.harvard.edu/foundation-cancer-image-biomarker)
- 🧠 [Vision Foundation Models for CT](https://arxiv.org/abs/2501.09001)

## Community

- 💬 [Discord](https://discord.gg/zJcnp6KrUp) - Chat with users
- 🐛 [GitHub Issues](https://github.com/project-lighter/lighter/issues) - Report bugs
- 📺 [YouTube](https://www.youtube.com/channel/UCef1oTpv2QEBrD2pZtrdk1Q) - Video tutorials
- 🤝 [Contributing](CONTRIBUTING.md) - Help improve Lighter

## Citation

If Lighter helps your research, please cite our [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.08101):

```bibtex
@article{lighter,
    doi = {10.21105/joss.08101},
    year = {2025}, publisher = {The Open Journal}, volume = {10}, number = {111}, pages = {8101},
    author = {Hadzic, Ibrahim and Pai, Suraj and Bressem, Keno and Foldyna, Borek and Aerts, Hugo JWL},
    title = {Lighter: Configuration-Driven Deep Learning},
    journal = {Journal of Open Source Software}
}
```
