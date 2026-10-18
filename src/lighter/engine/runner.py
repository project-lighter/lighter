"""Build a Lightning run from YAML configs and overrides, then run one stage (`lighter fit config.yaml`)."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from sparkwheel import Config

from lighter.utils.dynamic_imports import import_module_from_path

logger = logging.getLogger(__name__)

STAGES = {
    "fit": "Train a model",
    "validate": "Validate a model",
    "test": "Test a model",
    "predict": "Run predictions with a model",
}


def _import_project() -> bool:
    """Import the folder holding `__lighter__.py` (the current directory) as the `project` package."""
    cwd = Path.cwd()
    if not (cwd / "__lighter__.py").exists():
        return False
    import_module_from_path("project", cwd)
    logger.info(f"Imported 'project' module from '{cwd}'")
    return True


def _lightning_package(obj: Any, kind: str, key: str) -> str:
    """Return which Lightning package (`pytorch_lightning` or `lightning.pytorch`) defines the `kind` that `obj` is."""
    for package in ("pytorch_lightning", "lightning.pytorch"):
        if package in sys.modules and isinstance(obj, getattr(sys.modules[package], kind)):
            return package
    hint = f" Add a `_target_` to `{key}:`." if isinstance(obj, dict) else ""
    raise TypeError(f"`{key}` must build a {kind}, got {type(obj).__name__}.{hint}")


def load_config(*inputs: Any) -> Config:
    """Compose config files, dicts and overrides in order. Strings with `=` are overrides, others YAML files."""
    config = Config()
    for item in inputs:
        config.update(item)
    return config


def build(*inputs: Any) -> tuple[Any, Any, Any]:
    """Build `(model, trainer, datamodule)` from config files, dicts and overrides without running a stage.

    Inputs are composed by `load_config`. The seed comes from the top-level `seed:` key (default:
    `PL_GLOBAL_SEED`, else 0) and is set before anything is built.
    `datamodule` is None when the config has no `data:` key (the model's own dataloader hooks are then used).
    """
    from pytorch_lightning import seed_everything

    config = load_config(*inputs)
    seed = config.get("seed")
    if seed is None:
        seed = int(os.environ.get("PL_GLOBAL_SEED", 0))
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"`seed` must be an integer, got {seed!r}.")
    config.set("seed", seed)
    seed_everything(seed, workers=True)
    _import_project()

    model = config.resolve("model")
    trainer = config.resolve("trainer")
    datamodule = None if config.get("data") is None else config.resolve("data")
    packages = {_lightning_package(model, "LightningModule", "model"), _lightning_package(trainer, "Trainer", "trainer")}
    if datamodule is not None:
        packages.add(_lightning_package(datamodule, "LightningDataModule", "data"))
    if len(packages) > 1:
        raise TypeError(
            "The run mixes `pytorch_lightning` and `lightning.pytorch` objects, which Lightning rejects. Use one "
            "package throughout, e.g. `trainer::_target_: lightning.pytorch.Trainer` if your model imports `lightning`."
        )
    _save_config(config, trainer, model)
    return model, trainer, datamodule


def _save_config(config: Config, trainer: Any, model: Any) -> None:
    """Save the config to the model's hparams, the logger and `config.yaml` in the log directory."""
    model.save_hyperparameters({"config": config.get()})
    if not trainer.logger:
        return
    trainer.logger.log_hyperparams(config.get())
    if trainer.log_dir:
        config_file = Path(trainer.log_dir) / "config.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            yaml.dump(config.get(), f, default_flow_style=False, sort_keys=False, indent=4)
        logger.info(f"Saved config to: {config_file}")


class Runner:
    """Runs a Lightning stage from configuration inputs."""

    def run(self, stage: str, inputs: list, **stage_kwargs: Any) -> Any:
        """Build the run from `inputs` (see `build`) and return the result of `trainer.<stage>(...)`.

        Args:
            stage: "fit", "validate", "test" or "predict".
            inputs: Config file paths, dicts and overrides, applied in order.
            **stage_kwargs: Keyword arguments for the trainer stage method (e.g., ckpt_path, verbose).
        """
        if stage not in STAGES:
            raise ValueError(f"Unknown stage '{stage}', expected one of {list(STAGES)}.")
        model, trainer, datamodule = build(*inputs)
        return getattr(trainer, stage)(model, datamodule=datamodule, **stage_kwargs)


def cli() -> None:
    """Entry point for the lighter CLI."""
    parser = argparse.ArgumentParser(prog="lighter", description="Lighter: YAML-based deep learning framework")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    for stage, help_text in STAGES.items():
        stage_parser = subparsers.add_parser(
            stage,
            help=help_text,
            description=f"{help_text} using the specified configuration files and overrides.",
            epilog=f"Examples:\n  lighter {stage} config.yaml\n  lighter {stage} base.yaml experiment.yaml trainer::max_epochs=10",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        stage_parser.add_argument("inputs", nargs="+", help="Config files and overrides. Example: config.yaml model::lr=0.001")
        stage_parser.add_argument("--ckpt_path", help='Path to checkpoint. Can be "last", "best", or a file path.')
        stage_parser.add_argument(
            "--weights_only",
            action=argparse.BooleanOptionalAction,
            help="Restrict checkpoint loading to state_dicts of torch.Tensor (safer for untrusted sources).",
        )
        if stage in ("validate", "test"):
            stage_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, help="Print results.")
        if stage == "predict":
            stage_parser.add_argument(
                "--return_predictions",
                action=argparse.BooleanOptionalAction,
                help="Keep predictions in memory and return them (Lightning's default unless spawning processes).",
            )

    args = parser.parse_args()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("lighter").addHandler(handler)
    logging.getLogger("lighter").setLevel(logging.INFO)
    stage_kwargs = {k: v for k, v in vars(args).items() if k not in ["command", "inputs"] and v is not None}
    try:
        Runner().run(args.command, args.inputs, **stage_kwargs)
    except Exception as e:
        # Suppress exception chain to avoid duplicate tracebacks
        raise e from None
