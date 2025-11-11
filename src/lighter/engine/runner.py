"""
Runner module for executing training stages with configuration management.
Contains the Runner class and CLI entry point.
"""

import argparse

from pytorch_lightning import Trainer, seed_everything
from sparkwheel import Config, ValidationError

from lighter.engine.schema import LighterConfig
from lighter.system import System
from lighter.utils.dynamic_imports import import_module_from_path
from lighter.utils.types.enums import Mode, Stage


class Runner:
    """
    Executes training stages using validated and resolved configurations.

    The Runner loads configurations using Sparkwheel, applies CLI overrides,
    validates against the schema, prunes unused components for the stage,
    and executes the appropriate PyTorch Lightning trainer method.
    """

    STAGE_MODES = {
        Stage.FIT: [Mode.TRAIN, Mode.VAL],
        Stage.VALIDATE: [Mode.VAL],
        Stage.TEST: [Mode.TEST],
        Stage.PREDICT: [Mode.PREDICT],
    }

    def __init__(self) -> None:
        """Initialize the runner with empty state."""
        self.config: Config | None = None
        self.system: System | None = None
        self.trainer: Trainer | None = None

    def run(
        self,
        stage: Stage,
        config: str | list[str] | dict,
        overrides: list[str] | None = None,
    ) -> None:
        """
        Run a training stage with configuration and overrides.

        Args:
            stage: Stage to run (fit, validate, test, predict)
            config: Config file path(s) or dict. If string, supports comma-separated paths.
            overrides: List of CLI override strings in format "key::path=value"

        Raises:
            ValueError: If config validation fails or required components are missing
            TypeError: If system or trainer are not the correct type
        """
        seed_everything()

        # Handle comma-separated config files
        if isinstance(config, str) and "," in config:
            config = config.split(",")

        # Load config with CLI overrides and validation (all in one step!)
        try:
            self.config = Config.from_cli(
                config,
                overrides or [],
                schema=LighterConfig,
            )
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed:\n{e}") from e

        # Prune unused components for this stage
        self._prune_for_stage(stage)

        # Setup and run
        self._setup(stage)
        self._execute(stage)

    def _prune_for_stage(self, stage: Stage) -> None:
        """
        Remove unused components using Sparkwheel's delete directive (~).

        Args:
            stage: Current stage being executed
        """
        if self.config is None:
            raise ValueError("Config must be loaded before pruning")

        required = set(self.STAGE_MODES[stage])
        all_modes = {Mode.TRAIN, Mode.VAL, Mode.TEST, Mode.PREDICT}

        # Build delete directives for unused modes
        deletes = {}
        for mode in all_modes - required:
            deletes[f"~system::dataloaders::{mode}"] = None
            deletes[f"~system::metrics::{mode}"] = None

        # Remove optimizer/scheduler/criterion for non-training stages
        if stage != Stage.FIT:
            deletes["~system::optimizer"] = None
            deletes["~system::scheduler"] = None
            if stage != Stage.VALIDATE:
                deletes["~system::criterion"] = None

        # Keep only args for this stage
        for s in [Stage.FIT, Stage.VALIDATE, Stage.TEST, Stage.PREDICT]:
            if s != stage:
                deletes[f"~args::{s}"] = None

        # Apply deletions
        self.config.update(deletes)

    def _setup(self, stage: Stage) -> None:
        """
        Setup system and trainer from configuration.

        Args:
            stage: Current stage being executed

        Raises:
            TypeError: If system or trainer are not the correct type
        """
        if self.config is None:
            raise ValueError("Config must be loaded before setup")

        # Import project module if specified
        project = self.config.get("project")
        if project:
            import_module_from_path("project", project)

        # Resolve system
        self.system = self.config.resolve("system")
        if not isinstance(self.system, System):
            raise TypeError(f"system must be System, got {type(self.system)}")

        # Resolve trainer
        self.trainer = self.config.resolve("trainer")
        if not isinstance(self.trainer, Trainer):
            raise TypeError(f"trainer must be Trainer, got {type(self.trainer)}")

        # Save config to system checkpoint and trainer logger
        if self.system:
            self.system.save_hyperparameters(self.config.get())
        if self.trainer and self.trainer.logger:
            self.trainer.logger.log_hyperparams(self.config.get())

    def _execute(self, stage: Stage) -> None:
        """
        Execute the training stage.

        Args:
            stage: Stage to execute

        Raises:
            AttributeError: If trainer doesn't have the stage method
        """
        if self.config is None or self.trainer is None or self.system is None:
            raise ValueError("Config, trainer, and system must be set up before execution")

        # Get stage-specific arguments
        args = self.config.resolve(f"args::{stage}", default={})

        # Execute the stage method
        stage_method = getattr(self.trainer, str(stage))
        stage_method(self.system, **args)


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
        "  lighter fit config.yaml system::optimizer::lr=0.001\n"
        "  lighter fit base.yaml,experiment.yaml trainer::max_epochs=100",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    fit_parser.add_argument(
        "config",
        help="Path to config file(s), comma-separated for multiple files",
    )
    fit_parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help='Configuration overrides in format "key::path=value"',
    )

    # Validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a model",
        description="Validate a model using the specified configuration file.",
        epilog="Examples:\n"
        "  lighter validate config.yaml\n"
        "  lighter validate config.yaml system::model::weights=checkpoint.ckpt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    validate_parser.add_argument(
        "config",
        help="Path to config file(s), comma-separated for multiple files",
    )
    validate_parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help='Configuration overrides in format "key::path=value"',
    )

    # Test subcommand
    test_parser = subparsers.add_parser(
        "test",
        help="Test a model",
        description="Test a model using the specified configuration file.",
        epilog="Examples:\n  lighter test config.yaml\n  lighter test config.yaml system::model::weights=checkpoint.ckpt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    test_parser.add_argument(
        "config",
        help="Path to config file(s), comma-separated for multiple files",
    )
    test_parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help='Configuration overrides in format "key::path=value"',
    )

    # Predict subcommand
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run predictions with a model",
        description="Run predictions using the specified configuration file.",
        epilog="Examples:\n"
        "  lighter predict config.yaml\n"
        "  lighter predict config.yaml system::model::weights=checkpoint.ckpt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    predict_parser.add_argument(
        "config",
        help="Path to config file(s), comma-separated for multiple files",
    )
    predict_parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help='Configuration overrides in format "key::path=value"',
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    try:
        Runner().run(args.command, args.config, args.overrides)
    except Exception as e:
        # Suppress exception chain to avoid duplicate tracebacks
        raise e from None
