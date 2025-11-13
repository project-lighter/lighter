---
title: Lighter
toc_depth: 1
---

<!-- Fake title -->
#

<style>
    /* Remove content from the left bar (otherwise there's "Home" just sitting there) */
    .md-nav--primary {
    display: none;
    }
</style>


<!-- Logo -->
<div style="display: flex; justify-content: center;"><img src="assets/images/lighter_banner.png" style="width:65%;"/></div>

<!-- pip install -->
<div style="width:65%; margin:auto; text-align:center">
</br>

```bash
pip install lighter
```

<!-- [![PyPI](https://img.shields.io/pypi/v/lighter)](https://pypi.org/project/lighter/)
[![Python](https://img.shields.io/pypi/pyversions/lighter)](https://pypi.org/project/lighter/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://img.shields.io/discord/1234567890?label=Discord&logo=discord)](https://discord.gg/zJcnp6KrUp) -->

</div>
</br>

<!-- Body -->

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle }  __From Idea to Experiment in Seconds__

    ---

    No boilerplate. No training loops. Just define your model, data, and optimizer in YAML and run `lighter fit config.yaml`.

-   :material-refresh:{ .lg .middle }  __100% Reproducible__

    ---

    Every experiment is a YAML file. Version control configs like code. Share experiments with collaborators. No hidden state.

-   :material-tune:{ .lg .middle }  __Hyperparameter Sweeps Made Easy__

    ---

    Override any parameter from CLI: `lighter fit config.yaml system::optimizer::lr=0.01`. Run 100 experiments without editing files.

-   :material-puzzle-outline:{ .lg .middle }  __Task-Agnostic Adapters__

    ---

    Classification, segmentation, or self-supervised learning? Adapters handle any data format. One system, unlimited tasks.

-   :material-feather:{ .lg .middle }  __~1,000 Lines of Code__

    ---

    Read the entire framework in an afternoon. Debug easily. Understand exactly what's happening. No magic.

-   :material-lightning-bolt:{ .lg .middle }  __Built on PyTorch Lightning__

    ---

    Multi-GPU, mixed precision, gradient accumulation, profiling—all Lightning features work out of the box.

</div>

## Quick Start: 60 Seconds

<div class="annotate" markdown>

1. **Install Lighter**

    ```bash
    pip install lighter
    ```

2. **Create a config** (`config.yaml`)

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
        train: # (1)!
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

    1. Define your data like any PyTorch component

3. **Run training**

    ```bash
    lighter fit config.yaml
    ```

That's it. Automatic training loops, validation, checkpointing, and logging.

</div>

!!! tip "Experiment with different hyperparameters"
    ```bash
    # Change learning rate without editing files
    lighter fit config.yaml system::optimizer::lr=0.01

    # Train longer
    lighter fit config.yaml trainer::max_epochs=100

    # Use multiple GPUs
    lighter fit config.yaml trainer::devices=4
    ```


## Lighter vs. PyTorch Lightning

!!! abstract "Same Power, Different Interface"
    Lighter uses PyTorch Lightning under the hood. You get all Lightning features (multi-GPU, callbacks, profilers) but define experiments in YAML instead of Python classes.

See how training a model on CIFAR-10 differs:

=== "Lighter"
    ```bash title="Terminal"
    lighter fit config.yaml
    ```

    ```yaml title="config.yaml"
    trainer:
      _target_: pytorch_lightning.Trainer
      max_epochs: 2

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
          shuffle: true
          dataset:
            _target_: torchvision.datasets.CIFAR10
            download: true
            root: .datasets
            train: true
            transform:
              _target_: torchvision.transforms.Compose
              transforms:
                - _target_: torchvision.transforms.ToTensor
                - _target_: torchvision.transforms.Normalize
                  mean: [0.5, 0.5, 0.5]
                  std: [0.5, 0.5, 0.5]
    ```

    **Benefits:**

    - :material-check: Experiment is self-documenting
    - :material-check: Change hyperparameters from CLI without editing files
    - :material-check: Version control and compare configs with git diff
    - :material-check: Share experiments as single files

=== "PyTorch Lightning"
    ```bash title="Terminal"
    python cifar10.py
    ```

    ```py title="cifar10.py"
    from pytorch_lightning import Trainer, LightningModule
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from torchvision.models import resnet18
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor, Normalize, Compose


    class Model(LightningModule):
        def __init__(self):
            super().__init__()
            self.model = resnet18(num_classes=10)
            self.criterion = CrossEntropyLoss()

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            return loss

        def configure_optimizers(self):
            return Adam(self.model.parameters(), lr=0.001)


    transform = Compose([
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = CIFAR10(
        root=".datasets",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = Model()
    trainer = Trainer(max_epochs=2)
    trainer.fit(model, train_loader)
    ```

    **Challenges:**

    - :material-close: Need to edit Python code for hyperparameter changes
    - :material-close: Harder to compare experiments (code vs config)
    - :material-close: More boilerplate for each experiment

---

## Who Should Use Lighter?

<div class="grid" markdown>

<div markdown>

### :material-check-circle:{ .green } **Perfect For**

- **Researchers** running many experiments with hyperparameter variations
- **Teams** sharing reproducible experiments and baselines
- **Engineers** who value configuration over code for ML pipelines
- **Anyone** tired of writing boilerplate training loops

[Get Started →](tutorials/get-started.md){ .md-button .md-button--primary }

</div>

<div markdown>

### :material-information:{ .blue } **Consider Alternatives If**

- You need highly custom training loops with exotic logic
- You prefer pure Python workflows without YAML
- You're doing rapid prototyping where code is faster than config
- Your project has few experimental variations

[Compare Frameworks →](design/overview.md#framework-comparison){ .md-button }

</div>

</div>

---

## Key Features in Depth

### Configuration-Driven Everything

Every component is defined in YAML. Model, optimizer, scheduler, metrics, data—all configurable.

```yaml
# Differential learning rates? Easy.
optimizer:
  _target_: torch.optim.SGD
  params:
    - params: "$@system::model.backbone.parameters()"
      lr: 0.0001  # Low LR for pretrained backbone
    - params: "$@system::model.head.parameters()"
      lr: 0.01    # High LR for new head
```

[Learn Config Syntax →](how-to/configuration.md)

### Task-Agnostic Adapters

Adapters transform data between pipeline stages. This makes Lighter work for **any** task.

```yaml
# Dict-based dataset? No problem.
system:
  adapters:
    train:
      batch:
        _target_: lighter.adapters.BatchAdapter
        input_accessor: "image"   # Extract from dict
        target_accessor: "label"
```

Classification, segmentation, detection, self-supervised learning—adapters handle it all.

[Learn About Adapters →](how-to/adapters.md)

### Built on Solid Foundations

- **PyTorch Lightning** - Battle-tested training engine with multi-GPU, profiling, callbacks
- **[Sparkwheel](https://project-lighter.github.io/sparkwheel/)** - Powerful config system with references, expressions, and validation
- **~1,000 lines** - Read the entire framework, understand exactly what's happening

[Architecture Deep Dive →](design/overview.md)

---

## Choose Your Path

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle }  __New to Lighter?__

    ---

    **Start here:** Follow our comprehensive tutorial from installation to running your first experiments.

    Time: 15 minutes

    [:octicons-arrow-right-24: Get Started Tutorial](tutorials/get-started.md)

-   :material-lightning-bolt:{ .lg .middle }  __PyTorch Lightning User?__

    ---

    **Migration guide:** Translate your existing Lightning code to Lighter configs in minutes.

    Time: 10 minutes

    [:octicons-arrow-right-24: Migration Guide](migration/from-pytorch-lightning.md)

-   :material-book-open-variant:{ .lg .middle }  __Learn the Syntax__

    ---

    **Configuration reference:** Master Sparkwheel syntax: `_target_`, references (`@` and `%`), expressions (`$`), and path notation (`::`).

    Time: 20 minutes

    [:octicons-arrow-right-24: Configuration Guide](how-to/configuration.md)

-   :material-code-braces:{ .lg .middle }  __Ready-to-Use Examples__

    ---

    **Recipes & patterns:** Copy-paste configs for common scenarios and best practices.

    Time: 5 minutes per recipe

    [:octicons-arrow-right-24: View Recipes](how-to/recipes.md)

-   :material-puzzle-outline:{ .lg .middle }  __Understand Adapters__

    ---

    **Core concept:** Learn how adapters make Lighter task-agnostic and infinitely flexible.

    Time: 15 minutes

    [:octicons-arrow-right-24: Adapter Pattern](how-to/adapters.md)

-   :material-lightbulb:{ .lg .middle }  __Architecture & Philosophy__

    ---

    **Deep dive:** Understand the design decisions and how Lighter works internally.

    Time: 30 minutes

    [:octicons-arrow-right-24: Design Overview](design/overview.md)

</div>

---

## Community & Support

<div class="grid" markdown>

<div markdown>

### :material-account-group: Get Help

[:fontawesome-brands-discord: Discord](https://discord.gg/zJcnp6KrUp) - Chat with the community

[:material-frequently-asked-questions: FAQ](faq.md) - Common questions answered

[:material-bug: GitHub Issues](https://github.com/project-lighter/lighter/issues) - For bugs and features

[:material-book-open-page-variant: Troubleshooting](how-to/troubleshooting.md) - Common problems

</div>

<div markdown>

### :material-star: Contribute

[:fontawesome-brands-github: GitHub](https://github.com/project-lighter/lighter) - Star the repo

[:material-file-document-edit: Documentation](https://github.com/project-lighter/lighter/tree/main/docs) - Improve the docs

[:material-code-tags: Examples](https://github.com/project-lighter/lighter/tree/main/projects) - Share your configs

</div>

</div>


## Cite

If you find it useful, please cite our [*Journal of Open Source Software* paper](https://joss.theoj.org/papers/10.21105/joss.08101):

```bibtex
@article{lighter,
    doi = {10.21105/joss.08101},
    url = {https://doi.org/10.21105/joss.08101},
    year = {2025},
    publisher = {The Open Journal},
    volume = {10},
    number = {111},
    pages = {8101},
    author = {Hadzic, Ibrahim and Pai, Suraj and Bressem, Keno and Foldyna, Borek and Aerts, Hugo JWL},
    title = {Lighter: Configuration-Driven Deep Learning},
    journal = {Journal of Open Source Software}
}
```
