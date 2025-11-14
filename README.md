<div align="center">
  <img alt="Lighter logo" src="assets/images/lighter.png" width="80%">
</div>
<br/><br/>
<p align="center">
  <a href="https://github.com/project-lighter/lighter/actions"><img alt="Tests" src="https://github.com/project-lighter/lighter/workflows/Tests/badge.svg"></a>
  <a href="https://codecov.io/gh/project-lighter/lighter"><img alt="Coverage" src="https://codecov.io/gh/project-lighter/lighter/branch/main/graph/badge.svg"></a>
  <a href="https://pypi.org/project/lighter/"><img alt="PyPI" src="https://img.shields.io/pypi/v/lighter"></a>
  <a href="https://github.com/project-lighter/lighter/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <a href="https://project-lighter.github.io/lighter"><img alt="Documentation" src="https://img.shields.io/badge/docs-latest-olive"></a>
  <a href="https://discord.gg/zJcnp6KrUp"><img alt="Discord" src="https://dcbadge.limes.pink/api/server/https://discord.gg/zJcnp6KrUp?style=flat"></a>
</p>
<br/>

<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="assets/images/features_dark.png" width="85%">
        <source media="(prefers-color-scheme: light)" srcset="assets/images/features_light.png" width="85%">
        <img alt="Features" src="assets/images/features_light.png" width="85%">
    </picture>
</div>
<br/>

**Lighter** is a YAML-driven deep learning framework built on PyTorch Lightning. Define your model, data, and training in config files instead of writing boilerplate code.

## Quick Start

```bash
pip install lighter
```

Create `config.yaml`:
```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10

system:
  _target_: lighter.System
  model:
    _target_: torchvision.models.resnet18
    num_classes: 10
  criterion:
    _target_: torch.nn.CrossEntropyLoss
  optimizer:
    _target_: torch.optim.Adam
    params: "$@system::model.parameters()"
    lr: 0.001
  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      batch_size: 32
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: ./data
        train: true
        download: true
        transform:
          _target_: torchvision.transforms.ToTensor
```

Run:
```bash
lighter fit config.yaml
```

Override from CLI:
```bash
lighter fit config.yaml system::optimizer::lr=0.01
```

**[→ Full tutorial](https://project-lighter.github.io/lighter/tutorials/get-started/)**

## Documentation

- 📚 [Get Started](https://project-lighter.github.io/lighter/tutorials/get-started/)
- ⚙️ [Configuration Guide](https://project-lighter.github.io/lighter/how-to/configuration/)
- 🔌 [Adapters](https://project-lighter.github.io/lighter/how-to/adapters/)
- 🏗️ [Architecture](https://project-lighter.github.io/lighter/design/overview/)

## Projects Using Lighter

- 🏥 [Foundation Models for Cancer Imaging](https://aim.hms.harvard.edu/foundation-cancer-image-biomarker)
- 🧠 [Vision Foundation Models for CT](https://arxiv.org/abs/2501.09001)

## Community

- 💬 [Discord](https://discord.gg/zJcnp6KrUp)
- 🐛 [GitHub Issues](https://github.com/project-lighter/lighter/issues)
- 📺 [YouTube](https://www.youtube.com/channel/UCef1oTpv2QEBrD2pZtrdk1Q)
- 🤝 [Contributing](CONTRIBUTING.md)

## Citation

```bibtex
@article{lighter,
    doi = {10.21105/joss.08101},
    year = {2025}, publisher = {The Open Journal}, volume = {10}, number = {111}, pages = {8101},
    author = {Hadzic, Ibrahim and Pai, Suraj and Bressem, Keno and Foldyna, Borek and Aerts, Hugo JWL},
    title = {Lighter: Configuration-Driven Deep Learning},
    journal = {Journal of Open Source Software}
}
```
