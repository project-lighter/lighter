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

<!-- --8<-- [start:intro] -->
**Lighter** runs PyTorch Lightning from YAML: you write a LightningModule, a config builds it together with the
trainer and the data, and any value can be changed from the command line.
<!-- --8<-- [end:intro] -->

<!-- examples/quickstart/model.py -->
```python
import pytorch_lightning as pl
import torch


class Regressor(pl.LightningModule):
    def __init__(self, hidden: int = 32, lr: float = 0.01):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(8, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.net(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
```

<!-- examples/quickstart/config.yaml -->
```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 10

model:
  _target_: project.model.Regressor
  hidden: 32
  lr: 0.01

data:
  _target_: lighter.LighterDataModule
  train_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 16
    shuffle: true
    dataset:
      _target_: project.data.RandomData
```

<!-- --8<-- [start:run] -->
```console
$ cd lighter/examples/quickstart
$ lighter fit config.yaml model::lr=0.001
Seed set to 0
...
Lighter 0.1.0 | project: /home/you/lighter/examples/quickstart | config.yaml + 1 overrides | seed: 0
Logs: /home/you/lighter/examples/quickstart/lightning_logs/version_0
...
Epoch 9: 100%|██████████| 64/64 [00:00<00:00, 422.91it/s, v_num=0, train_loss=0.0319]
```
<!-- --8<-- [end:run] -->

## Quick start

Train it yourself; the run takes seconds on a CPU and downloads no data:

<!-- --8<-- [start:try] -->
```bash
pip install lighter
git clone --depth 1 https://github.com/project-lighter/lighter
lighter fit lighter/examples/quickstart/config.yaml
```
<!-- --8<-- [end:try] -->

[`examples/quickstart`](examples/quickstart) also holds `data.py`, a dataset of random numbers, and an empty
`__lighter__.py`, which lets the config name the folder's code `project.model` and `project.data`. Lighter builds
each `_target_` with the keys below it as arguments, then calls `trainer.fit(model, datamodule=data)`. Each run
writes to a new `lightning_logs/version_N/` folder and saves the config it ran there as `config.yaml`.

## What Lighter is

- **A thin layer over Lightning.** The config builds Lightning objects and Lightning runs them, so any
  LightningModule, LightningDataModule, callback or logger works as it is.
- **A record of every run.** The composed config, with its seed, is saved next to the logs and inside every
  checkpoint.
- **Errors at your line.** Config errors name the file, line and key where they can, with a suggestion when
  there is one.
- **Small.** ~1,200 lines of code.

Lighter does not run sweeps, compose config groups or track runs: a shell loop or a sweep tool, stacked config
files, and Lightning's loggers do those jobs.

`LighterModule` is an optional base class that takes the network, loss, optimizer, scheduler and metrics from
the config and logs the loss and metrics; you still write the step methods. See
[LighterModule](https://project-lighter.github.io/lighter/lighter-module/).

## Documentation

[project-lighter.github.io/lighter](https://project-lighter.github.io/lighter): a
[quickstart](https://project-lighter.github.io/lighter/quickstart/),
[how a run works](https://project-lighter.github.io/lighter/run/),
a [config cheat sheet](https://project-lighter.github.io/lighter/config/),
[recipes](https://project-lighter.github.io/lighter/recipes/),
[troubleshooting](https://project-lighter.github.io/lighter/troubleshooting/), and a page for those
[coming from LightningCLI or Hydra](https://project-lighter.github.io/lighter/coming-from/).

## Used in

- [Foundation Models for Cancer Imaging](https://aim.hms.harvard.edu/foundation-cancer-image-biomarker)
- [Vision Foundation Models for CT](https://arxiv.org/abs/2501.09001)

## Community

- [Discord](https://discord.gg/zJcnp6KrUp) for questions
- [GitHub Issues](https://github.com/project-lighter/lighter/issues) for bugs
- [YouTube](https://www.youtube.com/channel/UCef1oTpv2QEBrD2pZtrdk1Q) for video tutorials
- [Contributing](CONTRIBUTING.md)

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
