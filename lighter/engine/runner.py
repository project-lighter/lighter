"""
This module provides the `Config` and `LighterRunner` classes for managing and running 
machine learning experiments with PyTorch Lightning and MONAI.

It leverages a YAML configuration file to define the experiment parameters, system 
components, and training process. The `LighterRunner` class orchestrates the 
experiment execution based on the provided configuration and stage.
"""

from typing import Any, Dict, Optional

import copy
from functools import partial
from pathlib import Path

import fire
from monai.bundle.config_parser import ConfigParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.tuner import Tuner

from lighter.system import LighterSystem
from lighter.utils.dynamic_imports import import_module_from_path
from lighter.utils.enums import Mode, Stage


class LighterConfig:

    REQUIRED_SECTIONS = {"system", "trainer"}
    OPTIONAL_SECTIONS = {"_requires_", "args", "vars", "project"}
    ALLOWED_SECTIONS = REQUIRED_SECTIONS | OPTIONAL_SECTIONS

    # Stage-specific config rules
    PROHIBITED_STAGE_ARGS = {"model", "train_loaders", "validation_loaders", "dataloaders", "datamodule"}

    # Dataloader-specific config rules
    DATALOADER_SECTIONS = {Mode.TRAIN, Mode.VAL, Mode.TEST, Mode.PREDICT}

    # Metrics-specific config rules
    METRICS_SECTIONS = {Mode.TRAIN, Mode.VAL, Mode.TEST}

    def __init__(self, config_path: Optional[str] = None, **override_kwargs: Any):
        """
        Initialize the Config object.

        Args:
            config_path: Path to the YAML configuration file.
            override_kwargs: Keyword arguments to override values in the configuration file.
        """
        self.config_parser = ConfigParser()
        if config_path is not None:
            self.load_config(config_path, **override_kwargs)

    def load_config(self, config_path: str, **override_kwargs: Any) -> None:
        """
        Load and parse the configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.
            override_kwargs: Keyword arguments to override values in the configuration file.

        Raises:
            ValueError: If the configuration is invalid.
        """
        config = self.config_parser.load_config_files(config_path)
        config.update(override_kwargs)
        self.validate_config(config)
        self.config_parser.update(config)
        self.config_parser.parse()

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration sections and arguments.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if not self.REQUIRED_SECTIONS.issubset(config.keys()):
            raise ValueError(f"Config must have these required sections: {self.REQUIRED_SECTIONS}")

        if not set(config.keys()).issubset(self.ALLOWED_SECTIONS):
            invalid_sections = set(config.keys()) - self.ALLOWED_SECTIONS
            raise ValueError(f"Config contains invalid sections: {invalid_sections}")

        for section in self.REQUIRED_SECTIONS:
            if not isinstance(config.get(section), dict):
                raise ValueError(f"Config must have a '{section}' section.")

        # Validate stage-specific config rules
        args_config = config.get("args", {})
        for stage, value in args_config.items():
            if stage not in Stage:  # Use module-level constant
                raise ValueError(f"Invalid stage found in 'args': {stage}")

            if isinstance(value, dict):
                if self.PROHIBITED_STAGE_ARGS.intersection(value.keys()):
                    raise ValueError(
                        f"Found prohibited argument(s) in 'args#{stage}': "
                        f"{self.PROHIBITED_STAGE_ARGS.intersection(value.keys())}. "
                        f"Model and datasets should be defined within the 'system'."
                    )
            elif isinstance(value, str) and not (value.startswith("%") or value.startswith("@")):
                raise ValueError(f"Only dict or interpolators starting with '%' or '@' are allowed for 'args#{stage}'.")
            elif not isinstance(value, (dict, str)):
                raise ValueError(f"Invalid value type for 'args#{stage}'. Expected dict or str, got {type(value)}.")

        # Validate dataloader and metrics sections
        system_config = config.get("system", {})
        for key, valid_sections in [("dataloaders", self.DATALOADER_SECTIONS), ("metrics", self.METRICS_SECTIONS)]:
            if key in system_config:
                invalid_sections = set(system_config[key].keys()) - valid_sections
                if invalid_sections:
                    raise ValueError(
                        f"Invalid section(s) found in 'system.{key}': {invalid_sections}. "
                        f"Allowed sections are: {valid_sections}"
                    )

    def get(self, key: str = None, default: Any = None) -> Any:
        """Get raw content for the given key. If key is None, get the entire config."""
        return self.config_parser.config if key is None else self.config_parser.config.get(key, default)

    def get_parsed_content(self, key: str = None, default: Any = None) -> Any:
        """
        Get the parsed content for the given key. If key is None, get the entire parsed config.
        """
        return self.config_parser.get_parsed_content(key, default=default)


class LighterRunner:

    STAGE_MODES = {
        Stage.FIT: [Mode.TRAIN, Mode.VAL],
        Stage.VALIDATE: [Mode.VAL],
        Stage.TEST: [Mode.TEST],
        Stage.PREDICT: [Mode.PREDICT],
        Stage.LR_FIND: [Mode.TRAIN, Mode.VAL],
        Stage.SCALE_BATCH_SIZE: [Mode.TRAIN, Mode.VAL],
    }

    def __init__(self):
        self.config = None
        self.system = None
        self.trainer = None

    def _disable_unused_modes(self, config, enabled_modes):
        """Disables dataloaders and metrics for modes not used in the current stage."""
        system_config = config.get("system", {})
        for key in ["dataloaders", "metrics"]:
            if key in system_config:
                system_config[key] = {
                    mode: value
                    for mode, value in system_config[key].items()
                    if mode in enabled_modes and not (key == "metrics" and mode == "predict")
                }
        return config

    def _setup_stage(self, stage: str) -> None:
        """
        Set up the experiment for the given stage.

        Args:
            stage: The stage to set up.

        Raises:
            ValueError: If the stage is invalid or the configuration is missing required elements.
        """
        if stage not in self.STAGE_MODES:  # Use module-level constant
            raise ValueError(f"Invalid stage: {stage}. Must be one of {list(self.STAGE_MODES.keys())}.")

        if self.config is None:
            raise ValueError("Config must be loaded before setting up the stage.")

        # Create stage-specific configuration
        stage_config = copy.deepcopy(self.config.get())
        enabled_modes = self.STAGE_MODES[stage]  # Use module-level constant
        stage_config = self._disable_unused_modes(stage_config, enabled_modes)

        # Import project module if specified
        if project_path := stage_config.get("project"):
            import_module_from_path("project", Path(project_path))

        # Create new parser for stage configuration and parse
        stage_parser = ConfigParser(stage_config)
        stage_parser.parse()

        # Initialize system and trainer
        self.system = stage_parser.get_parsed_content("system")
        self.trainer = stage_parser.get_parsed_content("trainer")

        if not isinstance(self.system, LighterSystem):
            raise ValueError("'system' must be an instance of LighterSystem.")
        if not isinstance(self.trainer, Trainer):
            raise ValueError("'trainer' must be an instance of Trainer.")

        # Save hyperparameters
        self.system.save_hyperparameters(self.config.get())
        if self.trainer.logger:
            self.trainer.logger.log_hyperparams(self.config.get())

    def _execute_stage(self, stage: str) -> None:
        """
        Execute the given stage.

        Args:
            stage: The stage to execute.
        """
        # Get the stage-specific method
        if stage in [Stage.FIT, Stage.VALIDATE, Stage.TEST, Stage.PREDICT]:
            stage_method = getattr(self.trainer, stage)
        else:
            stage_method = getattr(Tuner(self.trainer), stage)

        # Get the stage-specific arguments
        stage_arguments = self.config.get_parsed_content(f"args#{stage}", default={})

        # Run the stage
        stage_method(self.system, **stage_arguments)

    def run(self, stage: str, config: str = None, **kwargs: Any) -> None:
        """
        Run the experiment for the specified stage.

        Args:
            stage: The stage to run (e.g., 'fit', 'validate', 'test', 'predict', 'lr_find').
            config: Path to the YAML configuration file.
            **kwargs: Additional keyword arguments to override values in the configuration file.
        """
        seed_everything()
        self.config = LighterConfig(config, **kwargs)
        self._setup_stage(stage)
        self._execute_stage(stage)


def cli():
    """Command-line interface entry point for LighterRunner."""
    runner = LighterRunner()
    commands = {stage: partial(runner.run, stage=stage) for stage in Stage}
    fire.Fire(commands)
