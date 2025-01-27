---
title: Lighter
---

<!-- Fake title -->
#

<!-- Logo -->
<div style="display: flex; justify-content: center;"><img src="assets/images/lighter_banner.png" style="width:50%;"/></div>

<!-- pip install -->
<div style="width:50%; margin:auto; text-align:center">
</br>

```bash
pip install lighter-framework
```
</div>
</br>

<!-- Body -->

**Lighter** is designed to streamline your deep learning experiments through a **configuration file**, eliminating boilerplate code and allowing you to focus on what truly matters: your research.  In the complex world of deep learning, managing experiments can become a significant overhead. Lighter offers a solution by automating the repetitive aspects, letting you concentrate on model architecture, datasets, and hyperparameters.

**Key Features**

*   **Configuration-Driven Workflow:** Define your entire experiment in a single YAML configuration file.
*   **Seamless Integration:** Built to work harmoniously with PyTorch Lightning and MONAI.
*   **Simplified Workflow:** Streamlines training, validation, testing, and prediction stages.
*   **Extensible Architecture:** Easily customize and extend functionality with adapters and writers.
*   **Dynamic Module Loading:**  Integrate your custom modules effortlessly.

**Quick Start**

Let's get started with a minimal example. First, ensure you have Lighter installed:

```bash
pip install lighter-framework
```

Now, create a `config.yaml` file for a simple MNIST classification task:

```yaml title="config.yaml"
trainer:
    _target_: pytorch_lightning.Trainer
    max_epochs: 1

system:
    _target_: lighter.System

    model:
        _target_: torch.nn.Linear
        in_features: 784
        out_features: 10

    criterion:
        _target_: torch.nn.CrossEntropyLoss

    optimizer:
        _target_: torch.optim.Adam
        params: "$@system#model.parameters()"
        lr: 0.001

    dataloaders:
        train:
            _target_: torch.utils.data.DataLoader
            dataset:
                _target_: torchvision.datasets.MNIST
                root: .datasets/
                download: true
                transform:
                    _target_: torchvision.transforms.ToTensor
            batch_size: 64
```

To run this experiment, execute the following command in your terminal:

```bash
lighter fit --config config.yaml
```

You should see output indicating the training progress. This minimal example demonstrates how Lighter allows you to define and run deep learning experiments with just a configuration file and a single command.

**Next Steps**

To delve deeper into Lighter's capabilities, explore the [Tutorials](tutorials/01_configuration_basics.md) section for step-by-step guides on various tasks. For a comprehensive understanding of Lighter's design and underlying principles, refer to the [Explanation](explanation/overview.md) section.

## Cite

If you find Lighter useful in your research or project, please consider citing it:

```bibtex
@software{lighter,
  author       = {Ibrahim Hadzic and
                  Suraj Pai and
                  Keno Bressem and
                  Hugo Aerts},
  title        = {Lighter},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.8007711},
  url          = {https://doi.org/10.5281/zenodo.8007711}
}
