"""
Runner module for executing training stages with configuration management.
Contains the Runner class and CLI entry point.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from sparkwheel import Config

from lighter.utils.dynamic_imports import import_module_from_path

logger = logging.getLogger(__name__)


def _import_project() -> bool:
    """Import the folder holding `__lighter__.py` (the current directory) as the `project` package."""
    cwd = Path.cwd()
    if not (cwd / "__lighter__.py").exists():
        return False
    import_module_from_path("project", cwd)
    logger.info(f"Imported 'project' module from '{cwd}'")
    return True


class Runner:
    """Loads the configuration, builds the model, trainer and datamodule, and runs a Lightning stage."""

    def run(self, stage: str, inputs: list, **stage_kwargs: Any) -> None:
        """
        Run a training stage with configuration inputs.

        Args:
            stage: Stage to run (fit, validate, test, predict)
            inputs: Config file paths, dicts and overrides, applied in order. Sparkwheel treats strings
                with '=' as overrides and other strings as file paths.
            **stage_kwargs: Keyword arguments for the trainer stage method (e.g., ckpt_path, verbose)
        """
        seed_everything()

        config = Config()
        for item in inputs:
            config.update(item)
        _import_project()

        model = self._resolve_model(config)
        trainer = self._resolve_trainer(config)
        datamodule = self._resolve_datamodule(config, model)
        self._save_config(config, trainer, model)
        self._execute(stage, model, trainer, datamodule, **stage_kwargs)

    def _resolve_model(self, config: Config) -> LightningModule:
        """Resolve and validate model from config."""
        model = config.resolve("model")
        if not isinstance(model, LightningModule):
            raise TypeError(f"model must be LightningModule or LighterModule, got {type(model)}")
        return model

    def _resolve_trainer(self, config: Config) -> Trainer:
        """Resolve and validate trainer from config."""
        trainer = config.resolve("trainer")
        if not isinstance(trainer, Trainer):
            raise TypeError(f"trainer must be Trainer, got {type(trainer)}")
        return trainer

    def _resolve_datamodule(self, config: Config, model: LightningModule) -> LightningDataModule | None:
        """Resolve the optional `data:` key. Without it, the model's own dataloader hooks are used."""
        if config.get("data") is None:
            return None

        datamodule = config.resolve("data")
        if not isinstance(datamodule, LightningDataModule):
            raise TypeError(
                f"data must be LightningDataModule (or lighter.LighterDataModule), got {type(datamodule)}. "
                "Example:\n"
                "data:\n"
                "  _target_: lighter.LighterDataModule\n"
                "  train_dataloader:\n"
                "    _target_: torch.utils.data.DataLoader\n"
                "    # ... config ..."
            )
        return datamodule

    def _save_config(self, config: Config, trainer: Trainer, model: LightningModule) -> None:
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

    def _execute(
        self, stage: str, model: LightningModule, trainer: Trainer, datamodule: LightningDataModule | None, **stage_kwargs: Any
    ) -> None:
        """Call `trainer.<stage>()`, passing the datamodule when there is one."""
        stage_method = getattr(trainer, str(stage))
        if datamodule is not None:
            stage_method(model, datamodule=datamodule, **stage_kwargs)
        else:
            stage_method(model, **stage_kwargs)


STAGES = {
    "fit": "Train a model",
    "validate": "Validate a model",
    "test": "Test a model",
    "predict": "Run predictions with a model",
}


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
