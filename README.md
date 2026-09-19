# Lighter

**Readable recipes for complete PyTorch Lightning experiments.** Keep the scientific code in Python; use YAML to compose the model, data, Trainer and callbacks, inspect a proposed run, and change settings from the command line.

Lighter supports ordinary LightningModules with their own hooks and optimization. Its optional LighterModule supplies managed optimizer construction and automatic measurements. A recipe records intent; reproducibility also needs code, data, environment and checkpoint identities.

## Start here

This checkout is the development pair **Lighter 0.2.0.dev0 / Sparkwheel 0.1.0.dev0**. Older published Sparkwheel packages do not provide the APIs this Lighter version requires. Use the source documentation linked below for this pair; the hosted site may describe an earlier release.

1. [Install the public source pair](docs/guides/compatibility.md#install-the-current-source-pair) in a fresh Python 3.11 environment.
2. [Run your first experiment](docs/quickstart.md): inspect, change, fit, evaluate, export and continue a download-free CPU example.
3. [Compare and Continue](projects/experiment_comparison/README.md): make a research decision on fixed CIFAR-10 populations, then follow the selected checkpoint and restored optimizer state.

The first experiment uses the same ordinary commands you use in your own project:

```bash
# From projects/tabular_regression, after installation:
python -m lighter inspect config.yaml model::optimizer::lr=0.05 --json
```

Inspection composes source without executing configured expressions or constructing the model. The [quick start](docs/quickstart.md) supplies the complete execution sequence and expected files.

## Find the next task

| I want to… | Read |
|---|---|
| Define a model or dataset | [Custom code](docs/guides/custom-code.md) |
| Reuse settings and compose recipes | [Configuration](docs/guides/configuration.md) |
| Keep existing Lightning code | [Native ownership](docs/guides/lightning-module.md) |
| Use Lighter's optimizer and logging conveniences | [LighterModule](docs/guides/lighter-module.md) |
| Select, evaluate or continue a checkpoint | [Training workflows](docs/guides/training.md) |
| Inspect requested and observed settings | [Experiment records](docs/guides/experiment-records.md) |
| Save identifiable predictions | [Prediction output](docs/guides/predictions.md) |
| Diagnose an error | [Troubleshooting](docs/faq.md) |

See the [project index](projects/README.md) for the diagnostic, recommended research walkthrough and status of specialist references. Support and execution limits are in [compatibility](docs/guides/compatibility.md#exact-validation-profiles). Native APIs remain available; individual strategies and scientific workflows still need their own qualification.

## Community and citation

[Documentation](https://project-lighter.github.io/lighter/) · [Issues](https://github.com/project-lighter/lighter/issues) · [Discord](https://discord.gg/zJcnp6KrUp) · [Contributing](CONTRIBUTING.md) · [MIT license](LICENSE)

If Lighter helps your research, cite the [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.08101):

```bibtex
@article{lighter,
    doi = {10.21105/joss.08101},
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
