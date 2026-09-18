"""
Runner module for executing training stages with configuration management.
Contains the Runner class and CLI entry point.
"""

import argparse
import inspect
import sys
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
        logger.info(f"Imported 'project' module from '{cwd}'")
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


class _ResolutionView:
    """Private source/runtime view for one Runner invocation."""

    def __init__(self, source: Config, stage: Stage, overrides: dict[str, Any]):
        self.source = source
        self.recipe = source.retain()
        arguments = source.get("args", {})
        selected = arguments.get(str(stage), {})
        bindings = {f"args::{stage}::{key}": value for key, value in overrides.items() if key in selected}
        blocked = {f"args::{name}" for name in arguments if name != str(stage)}
        self.scope = self.recipe.scope(bindings=bindings, blocked_paths=blocked)

    def get(self, path: str = "", default: Any = None) -> Any:
        return self.source.get(path, default)

    def resolve(self, path: str) -> Any:
        return self.scope.resolve(path)


class Runner:
    """
    Orchestrates training stage execution by coordinating helper classes.

    Runner delegates responsibilities to specialized helper classes:
    - ProjectImporter: Auto-discovers and imports user project modules via __lighter__.py marker
    - ConfigLoader: Loads and validates configurations using Sparkwheel

    Runner focuses on resolving and validating components (model, trainer, datamodule)
    and executing the requested training stage.
    """

    def run(
        self,
        stage: Stage | str,
        inputs: list,
        **stage_kwargs: Any,
    ) -> Any:
        """
        Run a training stage with configuration inputs.

        Orchestrates the complete training workflow:
        1. Loads configuration via ConfigLoader (delegates to Sparkwheel for auto-detection)
        2. Seeds Python, NumPy and Torch, then imports project modules via ProjectImporter
        3. Resolves and validates model, trainer, and datamodule components
        4. Saves configuration (to log directory, logger, and model hyperparameters)
        5. Executes the requested training stage

        Args:
            stage: Stage to run (fit, validate, test, predict)
            inputs: List of config file paths, dicts, and/or overrides.
                   Passed to ConfigLoader.load() which delegates to Sparkwheel for auto-detection:
                   - Strings without '=' → file paths
                   - Strings with '=' → overrides
                   - Dicts → merged into config
            **stage_kwargs: Native Trainer arguments overriding args::<stage> recipe defaults.
                           Only selected, non-overridden argument recipes are constructed.

        Returns:
            The native Trainer stage result. Fit normally returns None; evaluation
            and prediction return their native values, subject to Trainer options.

        Raises:
            ValueError: If config validation fails or required components are missing
            TypeError: If model or trainer are not the correct type
        """
        stage = Stage(stage)

        # Load static intent, then seed before project imports or object construction.
        config = ConfigLoader.load(inputs)
        seed = config.get("seed", 0)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer between 0 and 4294967295 (default: 0)")
        seed_everything(seed, workers=True)
        self._validate_stage_args(config, stage, stage_kwargs)

        # 2. Auto-discover and import project
        ProjectImporter.auto_discover_and_import()

        # Authoritative overrides also apply to references inside the runtime graph.
        resolution = _ResolutionView(config, stage, stage_kwargs)

        # 3. Resolve components
        model = self._resolve_model(resolution)
        trainer = self._resolve_trainer(resolution)
        datamodule = self._resolve_datamodule(resolution, model)

        # Resolve only selected, non-overridden native stage arguments.
        configured_args = config.get(f"args::{stage}", {})
        stage_kwargs = {
            **{key: resolution.resolve(f"args::{stage}::{key}") for key in configured_args if key not in stage_kwargs},
            **stage_kwargs,
        }

        # 4. Save configuration to trainer's log directory, logger, and model hparams for checkpoint access
        self._save_config(config, trainer, model)

        # 5. Execute stage
        return self._execute(stage, model, trainer, datamodule, **stage_kwargs)

    @staticmethod
    def _validate_stage_args(config: Config, stage: Stage, overrides: dict[str, Any]) -> None:
        """Validate the envelope without constructing inactive or overridden values."""
        arguments = config.get("args", {})
        if not isinstance(arguments, dict):
            raise TypeError("args must be a mapping of stage names to native Trainer arguments")
        selected = arguments.get(str(stage), {})
        if not isinstance(selected, dict) or any(not isinstance(key, str) for key in selected):
            raise TypeError(f"args::{stage} must be a mapping with string argument names")
        if "model" in selected or "model" in overrides:
            raise ValueError(f"args::{stage} cannot replace model; use the top-level model recipe")

    def _resolve_model(self, config: Config | _ResolutionView) -> LightningModule:
        """Resolve and validate model from config."""
        model = config.resolve("model")
        if not isinstance(model, LightningModule):
            raise TypeError(f"model must be LightningModule or LighterModule, got {type(model)}")
        return model

    def _resolve_trainer(self, config: Config | _ResolutionView) -> Trainer:
        """Resolve and validate trainer from config."""
        trainer = config.resolve("trainer")
        if not isinstance(trainer, Trainer):
            raise TypeError(f"trainer must be Trainer, got {type(trainer)}")
        return trainer

    def _resolve_datamodule(self, config: Config | _ResolutionView, model: LightningModule) -> LightningDataModule | None:
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
            # Lightning validates stage-specific data ownership, including direct loaders.
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

    def _save_config(self, config: Config, trainer: Trainer, model: LightningModule) -> None:
        """
        Save configuration to multiple destinations.

        Saves the configuration to:
        - Model (for checkpoint access via model.hparams)
        - Logger (for experiment tracking via log_hyperparams)
        - Log directory (as config.yaml file)

        Args:
            config: Configuration object to save
            trainer: Trainer (uses trainer.logger and trainer.log_dir)
            model: Model to save hyperparameters to
        """

        # Save to model checkpoint (for model.hparams access)
        model.save_hyperparameters({"config": config.get()})

        # If no logger, skip other saves
        if not trainer.logger:
            return

        # Save to logger (for experiment tracking)
        trainer.logger.log_hyperparams(config.get())

        # Save as config.yaml to log directory if it exists
        if trainer.log_dir:
            config_file = Path(trainer.log_dir) / "config.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, "w") as f:
                yaml.dump(config.get(), f, default_flow_style=False, sort_keys=False, indent=4)
            logger.info(f"Saved config to: {config_file}")

    def _execute(
        self,
        stage: Stage | str,
        model: LightningModule,
        trainer: Trainer,
        datamodule: LightningDataModule | None,
        **stage_kwargs: Any,
    ) -> Any:
        """
        Execute the training stage and return its native result.

        Args:
            stage: Stage to execute (fit, validate, test, predict)
            model: Resolved model
            trainer: Resolved trainer
            datamodule: Resolved datamodule (None if model defines its own dataloaders)
            **stage_kwargs: Additional keyword arguments from CLI (e.g., ckpt_path, verbose)
        """
        stage_method = getattr(trainer, str(stage))
        if datamodule is not None:
            if "datamodule" in stage_kwargs:
                raise ValueError("Specify a datamodule through data or stage arguments, not both")
            stage_kwargs = {"datamodule": datamodule, **stage_kwargs}
        try:
            inspect.signature(stage_method).bind(model, **stage_kwargs)
        except TypeError as error:
            raise TypeError(f"Arguments for Trainer.{stage} are unsupported by the installed Lightning: {error}") from error
        return stage_method(model, **stage_kwargs)


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

    # Common arguments shared by all stages
    def add_common_args(stage_parser):
        """Add common arguments to a stage subparser."""
        stage_parser.add_argument(
            "inputs",
            nargs="+",
            help="Config files and overrides. Example: config.yaml model::optimizer::lr=0.001",
        )
        stage_parser.add_argument(
            "--ckpt_path",
            "--ckpt-path",
            type=str,
            default=None,
            help='Path to checkpoint. Can be "last", "best", or a file path.',
        )
        stage_parser.add_argument(
            "--weights_only",
            "--weights-only",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Pass weights_only to Trainer checkpoint loading (requires a supporting Lightning version).",
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
    add_common_args(fit_parser)

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
    add_common_args(validate_parser)
    validate_parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print validation results (default: True).",
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
    add_common_args(test_parser)
    test_parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print test results (default: True).",
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
    add_common_args(predict_parser)
    predict_parser.add_argument(
        "--return_predictions",
        "--return-predictions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to return predictions (default: True except with process-spawning accelerators).",
    )

    # Parse arguments
    # argparse cannot intermix a variable positional and options through subparsers.
    # Select the known stage first, then use its normal intermixing parser.
    if len(sys.argv) > 1 and sys.argv[1] in subparsers.choices:
        args = subparsers.choices[sys.argv[1]].parse_intermixed_args(sys.argv[2:])
        args.command = sys.argv[1]
    else:
        args = parser.parse_args()

    # Extract stage kwargs (exclude command and inputs)
    stage_kwargs = {k: v for k, v in vars(args).items() if k not in ["command", "inputs"] and v is not None}

    # Execute command
    try:
        Runner().run(args.command, args.inputs, **stage_kwargs)
    except Exception as e:
        # Suppress exception chain to avoid duplicate tracebacks
        raise e from None
