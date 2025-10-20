---
title: Lighter
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
</div>
</br>

<!-- Body -->


<div class="grid cards" markdown>

-   :material-cog-outline:{ .lg .middle }  __Configuration-based__

    ---

    Define, reproduce, and share experiments through YAML configs. No more scattered Python scripts!

-   :material-cube-outline:{ .lg .middle }  __Task-agnostic__

    ---

    Classification, segmentation, or self-supervised learning? Lighter can handle it.

-   :material-file-code-outline:{ .lg .middle }  __Minimal__

    ---

    Lighter handles the boilerplate so that your projects stay lightweight and maintainable.

-   :material-puzzle-outline:{ .lg .middle }  __Customizable__

    ---

    Add custom models, datasets, losses, or any other component. If it's Python, Lighter can use it.

</div>


## Lighter vs. PyTorch Lightning

See how training a model on CIFAR-10 differs between Lighter and PyTorch Lightning.

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
            params: "$@system#model.parameters()"
            lr: 0.001

        dataloaders:
            train:
                _target_: torch.utils.data.DataLoader
                batch_size: 32
                shuffle: True
                dataset:
                    _target_: torchvision.datasets.CIFAR10
                    download: True
                    root: .datasets
                    train: True
                    transform:
                        _target_: torchvision.transforms.Compose
                        transforms:
                            - _target_: torchvision.transforms.ToTensor
                            - _target_: torchvision.transforms.Normalize
                              mean: [0.5, 0.5, 0.5]
                              std: [0.5, 0.5, 0.5]
    ```

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

## Next Steps

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle }  __Start Here: Tutorials__

    ---

    New to Lighter? Start with our Zero to Hero guide
    [:octicons-arrow-right-24: Get Started](tutorials/zero_to_hero.md)

-   :material-hammer-wrench:{ .lg .middle }  __How-To Guides__

    ---

    Learn Lighter's features with concept and practice guides
    [:octicons-arrow-right-24: Explore](how-to/run.md)

-   :material-lightbulb:{ .lg .middle }  __Design & Architecture__

    ---

    Understand Lighter's design principles
    [:octicons-arrow-right-24: Learn More](design/overview.md)

-   :material-api:{ .lg .middle }  __API Reference__

    ---

    Explore Lighter's classes and functions
    [:octicons-arrow-right-24: View API](reference/)

</div>


## Cite

If you find it useful, please cite our [*Journal of Open Source Sofware* paper](https://joss.theoj.org/papers/10.21105/joss.08101):

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
