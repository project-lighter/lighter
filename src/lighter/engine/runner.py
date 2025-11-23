"""
Runner module for executing training stages with configuration management.
Contains the Runner class and CLI entry point.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from sparkwheel import Config, ValidationError

from lighter.utils.dynamic_imports import import_module_from_path
from lighter.utils.types.enums import Stage

# ============================================================================
# Helper Classes - Each Does One Thing
# ============================================================================


@dataclass
class OutputDir:
    """Creates and manages output directory."""

    path: Path

    @classmethod
    def create_timestamped(cls, base: Path = Path("outputs")) -> "OutputDir":
        """Create timestamped output directory (current default behavior)."""
        timestamp = datetime.now()
        path = base / timestamp.strftime("%Y-%m-%d") / timestamp.strftime("%H-%M-%S")
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {path}")
        return cls(path)

    def save_config(self, config: Config) -> None:
        """Save config YAML to output directory."""
        config_file = self.path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config.get(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved config to: {config_file}")

    def as_trainer_default_root_dir(self) -> str:
        """Return path formatted for trainer.default_root_dir."""
        return str(self.path)


class ProjectImporter:
    """Discovers and imports user project modules."""

    @staticmethod
    def auto_discover_and_import() -> bool:
        """
        Auto-discover project from __lighter__.py marker file.
        Returns True if project was imported, False otherwise.
        """
        cwd = Path.cwd()
        marker = cwd / "__lighter__.py"

        if not marker.exists():
            return False

        import_module_from_path("project", cwd)
        logger.info(f"Auto-discovered project at {cwd} (imported as 'project')")
        return True


class ConfigLoader:
    """Loads and validates configuration using Sparkwheel."""

    @staticmethod
    def load(inputs: list) -> Config:
        """
        Load config from inputs (files, dicts, overrides).

        Sparkwheel auto-detects:
        - Strings without '=' → file paths
        - Strings with '=' → overrides
        - Dicts → merged into config
        """
        try:
            config = Config()  # No schema validation for now
            for item in inputs:
                config.update(item)
            return config
        except ValidationError as e:
            raise ValueError(f"Configuration loading failed:\n{e}") from e

    @staticmethod
    def set_default_root_dir(config: Config, output_dir: OutputDir) -> None:
        """Set trainer.default_root_dir if not already specified by user."""
        if config.get("trainer::default_root_dir") is None:
            config.update({"trainer::default_root_dir": output_dir.as_trainer_default_root_dir()})


class Runner:
    """
    Executes training stages using validated and resolved configurations.

    Simplified in v3.1: delegates to helper classes for specific tasks.
    """

    def run(
        self,
        stage: Stage,
        inputs: list,
        **stage_kwargs: Any,
    ) -> None:
        """
        Run a training stage with configuration inputs.

        Args:
            stage: Stage to run (fit, validate, test, predict)
            inputs: List of config file paths, dicts, and/or overrides.
                   Sparkwheel auto-detects based on content:
                   - Strings without '=' → file paths
                   - Strings with '=' → overrides
                   - Dicts → merged into config
            **stage_kwargs: Additional keyword arguments from CLI (e.g., ckpt_path, verbose)
                           passed directly to the trainer stage method

        Raises:
            ValueError: If config validation fails or required components are missing
            TypeError: If model or trainer are not the correct type
        """
        seed_everything()

        # 1. Load configuration
        config = ConfigLoader.load(inputs)

        # 2. Setup output directory
        output_dir = OutputDir.create_timestamped()
        ConfigLoader.set_default_root_dir(config, output_dir)
        output_dir.save_config(config)

        # 3. Auto-discover and import project
        ProjectImporter.auto_discover_and_import()

        # 4. Resolve components
        model = self._resolve_model(config)
        trainer = self._resolve_trainer(config)
        datamodule = self._resolve_datamodule(config, model)

        # 5. Save hyperparameters
        self._save_hyperparameters(model, trainer, config)

        # 6. Execute stage
        self._execute(stage, config, model, trainer, datamodule, **stage_kwargs)

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
        """
        Resolve and validate datamodule from config.

        Args:
            config: Configuration object
            model: Resolved model (checked for built-in dataloaders)

        Returns:
            LightningDataModule instance or None if model defines its own dataloaders

        Raises:
            TypeError: If data key exists but is not a LightningDataModule
        """
        # Data key is optional - plain Lightning modules can define their own dataloaders
        if config.get("data") is None:
            # Check if model has dataloader methods (plain Lightning module)
            has_dataloaders = any(
                hasattr(model, method)
                for method in ["train_dataloader", "val_dataloader", "test_dataloader", "predict_dataloader"]
            )
            if not has_dataloaders:
                raise ValueError(
                    "Missing required 'data:' config key and model does not define dataloader methods. "
                    "Either:\n"
                    "1. Add 'data:' config key:\n"
                    "   data:\n"
                    "     _target_: lighter.LighterDataModule\n"
                    "     train_dataloader: ...\n"
                    "2. Or define dataloader methods in your LightningModule (train_dataloader, val_dataloader, etc.)"
                )
            return None

        # Resolve and validate data key
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

    def _save_hyperparameters(self, model: LightningModule, trainer: Trainer, config: Config) -> None:
        """Save config to model checkpoint and trainer logger."""
        model.save_hyperparameters(config.get())
        if trainer.logger:
            trainer.logger.log_hyperparams(config.get())

    def _execute(
        self,
        stage: Stage,
        config: Config,
        model: LightningModule,
        trainer: Trainer,
        datamodule: LightningDataModule | None,
        **stage_kwargs: Any,
    ) -> None:
        """
        Execute the training stage.

        Args:
            stage: Stage to execute (fit, validate, test, predict)
            config: Configuration object
            model: Resolved model
            trainer: Resolved trainer
            datamodule: Resolved datamodule (None if model defines its own dataloaders)
            **stage_kwargs: Additional keyword arguments from CLI (e.g., ckpt_path, verbose)
        """
        # Execute the stage method with CLI kwargs
        stage_method = getattr(trainer, str(stage))
        if datamodule is not None:
            stage_method(model, datamodule=datamodule, **stage_kwargs)
        else:
            # Plain Lightning module with built-in dataloaders
            stage_method(model, **stage_kwargs)


def cli() -> None:
    """Entry point for the lighter CLI."""
    parser = argparse.ArgumentParser(
        prog="lighter",
        description="Lighter: YAML-based deep learning framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available commands",
    )

    # Fit subcommand
    fit_parser = subparsers.add_parser(
        "fit",
        help="Train a model",
        description="Train a model using the specified configuration file.",
        epilog="Examples:\n"
        "  lighter fit config.yaml\n"
        "  lighter fit config.yaml --ckpt_path checkpoint.ckpt\n"
        "  lighter fit config.yaml model::optimizer::lr=0.001\n"
        "  lighter fit base.yaml experiment.yaml --ckpt_path last trainer::max_epochs=100",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    fit_parser.add_argument(
        "inputs",
        nargs="+",
        help="Config files and overrides. Example: config.yaml model::optimizer::lr=0.001",
    )
    fit_parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help='Path to checkpoint to resume training from. Can be "last", "best", or a file path.',
    )
    fit_parser.add_argument(
        "--weights_only",
        type=bool,
        default=None,
        help="Load only weights from checkpoint (security option, restricts to tensors/primitives).",
    )

    # Validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a model",
        description="Validate a model using the specified configuration file.",
        epilog="Examples:\n"
        "  lighter validate config.yaml\n"
        "  lighter validate config.yaml --ckpt_path best\n"
        "  lighter validate config.yaml --ckpt_path checkpoint.ckpt --verbose",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    validate_parser.add_argument(
        "inputs",
        nargs="+",
        help="Config files and overrides. Example: config.yaml model::network::weights=checkpoint.ckpt",
    )
    validate_parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help='Path to checkpoint for validation. Can be "last", "best", or a file path.',
    )
    validate_parser.add_argument(
        "--verbose",
        type=bool,
        default=None,
        help="Print validation results (default: True).",
    )
    validate_parser.add_argument(
        "--weights_only",
        type=bool,
        default=None,
        help="Load only weights from checkpoint (security option).",
    )

    # Test subcommand
    test_parser = subparsers.add_parser(
        "test",
        help="Test a model",
        description="Test a model using the specified configuration file.",
        epilog="Examples:\n"
        "  lighter test config.yaml\n"
        "  lighter test config.yaml --ckpt_path best\n"
        "  lighter test config.yaml --ckpt_path checkpoint.ckpt --verbose",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    test_parser.add_argument(
        "inputs",
        nargs="+",
        help="Config files and overrides. Example: config.yaml model::network::weights=checkpoint.ckpt",
    )
    test_parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help='Path to checkpoint for testing. Can be "last", "best", or a file path.',
    )
    test_parser.add_argument(
        "--verbose",
        type=bool,
        default=None,
        help="Print test results (default: True).",
    )
    test_parser.add_argument(
        "--weights_only",
        type=bool,
        default=None,
        help="Load only weights from checkpoint (security option).",
    )

    # Predict subcommand
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run predictions with a model",
        description="Run predictions using the specified configuration file.",
        epilog="Examples:\n"
        "  lighter predict config.yaml\n"
        "  lighter predict config.yaml --ckpt_path best\n"
        "  lighter predict config.yaml --ckpt_path checkpoint.ckpt --return_predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    predict_parser.add_argument(
        "inputs",
        nargs="+",
        help="Config files and overrides. Example: config.yaml model::network::weights=checkpoint.ckpt",
    )
    predict_parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help='Path to checkpoint for predictions. Can be "last", "best", or a file path.',
    )
    predict_parser.add_argument(
        "--return_predictions",
        type=bool,
        default=None,
        help="Whether to return predictions (default: True except with process-spawning accelerators).",
    )
    predict_parser.add_argument(
        "--weights_only",
        type=bool,
        default=None,
        help="Load only weights from checkpoint (security option).",
    )

    # Parse arguments
    args = parser.parse_args()

    # Extract stage kwargs (exclude command and inputs)
    stage_kwargs = {k: v for k, v in vars(args).items() if k not in ["command", "inputs"] and v is not None}

    # Execute command
    try:
        Runner().run(args.command, args.inputs, **stage_kwargs)
    except Exception as e:
        # Suppress exception chain to avoid duplicate tracebacks
        raise e from None
