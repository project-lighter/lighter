from typing import Any

from pathlib import Path

import fire
from monai.bundle.config_parser import ConfigParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.tuner import Tuner

from lighter.system import System
from lighter.utils.dynamic_imports import import_module_from_path
from lighter.utils.types.enums import Mode, Stage


class Config:
    """
    Handles loading and overriding configurations using ConfigParser.
    """

    def __init__(self, config: str | dict | None = None, **config_overrides: Any):
        """
        Initialize the Config object.

        Args:
            config: Path to a YAML configuration file or a dictionary containing the configuration.
            config_overrides: Keyword arguments to override values in the configuration file
        """
        self._config_parser = ConfigParser(globals=False)
        if isinstance(config, dict):
            self._config_parser.update(config)
        elif isinstance(config, str):
            # load_config_files() reads and merges multiple config paths separated by comma
            config = self._config_parser.load_config_files(config)
            self._config_parser.read_config(config)
        else:
            raise ValueError("Invalid type for 'config'. Must be a dictionary or a path to a YAML file.")
        self._config_parser.parse()
        self._config_parser.update(config_overrides)

    def get(self, key: str | None = None, default: Any = None) -> Any:
        """Get raw content for the given key. If key is None, get the entire config."""
        return self._config_parser.config if key is None else self._config_parser.config.get(key, default)

    def get_parsed_content(self, key: str | None = None, default: Any = None) -> Any:
        """
        Get the parsed content for the given key. If key is None, get the entire parsed config.
        """
        return self._config_parser.get_parsed_content(key, default=default)


class Validator:
    """
    Validates the configuration to ensure correctness and completeness.
    """

    # Root sections
    REQUIRED_SECTIONS = {"system", "trainer"}
    OPTIONAL_SECTIONS = {"_meta_", "_requires_", "args", "vars", "project"}
    ALLOWED_SECTIONS = REQUIRED_SECTIONS | OPTIONAL_SECTIONS

    # Subsections
    DATALOADER_SECTIONS = {Mode.TRAIN, Mode.VAL, Mode.TEST, Mode.PREDICT}
    METRICS_SECTIONS = {Mode.TRAIN, Mode.VAL, Mode.TEST}
    ADAPTERS_SECTIONS = {
        Mode.TRAIN: {"batch", "model", "criterion", "metrics", "logging"},
        Mode.VAL: {"batch", "model", "criterion", "metrics", "logging"},
        Mode.TEST: {"batch", "model", "metrics", "logging"},
        Mode.PREDICT: {"batch", "model", "logging"},
    }

    # Prohibited arguemnts for stage args (e.g., args#fit#model)
    PROHIBITED_ARGS = {"model", "train_loaders", "validation_loaders", "dataloaders", "datamodule"}

    def __init__(self, config: Config):
        self.config = config
        self.validate()

    def validate(self) -> None:
        """Perform all validation steps."""
        self._validate_sections()
        self._validate_args_config()
        self._validate_system_config()

    def _validate_sections(self) -> None:
        """Validate the presence and validity of configuration sections."""
        config_keys = self.config.get().keys()

        missing_sections = self.REQUIRED_SECTIONS - set(config_keys)
        invalid_sections = set(config_keys) - self.ALLOWED_SECTIONS

        errors = []
        if missing_sections:
            errors.append(f"Missing required sections: {missing_sections}")
        if invalid_sections:
            errors.append(f"Invalid sections found: {invalid_sections}")
        if errors:
            raise ValueError("\n".join(errors))

    def _validate_args_config(self) -> None:
        """Validate stage-specific arguments in the configuration."""
        args_config = self.config.get("args", {})

        for stage, value in args_config.items():
            if stage not in Stage:
                raise ValueError(f"Invalid stage in 'args': {stage}")

            if isinstance(value, dict):
                prohibited = self.PROHIBITED_ARGS.intersection(value.keys())
                if prohibited:
                    raise ValueError(
                        f"Prohibited argument(s) in 'args#{stage}': {prohibited}. "
                        "Model and datasets should be defined within the 'system'."
                    )
            elif isinstance(value, str) and not value.startswith(("%", "@")):
                raise ValueError(f"Stage value must be dict or interpolator starting with '%' or '@', got: {value}")
            else:
                raise ValueError(f"Invalid type for 'args#{stage}'. Expected dict or str, got {type(value)}")

    def _validate_system_config(self) -> None:
        """Validate system-specific sections in the configuration."""
        system_config = self.config.get("system", {})

        # Validate dataloaders
        if "dataloaders" in system_config:
            invalid = set(system_config["dataloaders"].keys()) - self.DATALOADER_SECTIONS
            if invalid:
                raise ValueError(
                    f"Invalid section(s) in 'system.dataloaders': {invalid}. Allowed sections are: {self.DATALOADER_SECTIONS}"
                )

        # Validate metrics
        if "metrics" in system_config:
            invalid = set(system_config["metrics"].keys()) - self.METRICS_SECTIONS
            if invalid:
                raise ValueError(
                    f"Invalid section(s) in 'system.metrics': {invalid}. Allowed sections are: {self.METRICS_SECTIONS}"
                )

        # Validate adapters
        if "adapters" in system_config:
            adapters_config = system_config["adapters"]
            invalid = set(adapters_config.keys()) - set(self.ADAPTERS_SECTIONS.keys())
            if invalid:
                raise ValueError(
                    f"Invalid section(s) in 'system.adapters': {invalid}. "
                    f"Allowed modes are: {set(self.ADAPTERS_SECTIONS.keys())}"
                )
            self._validate_adapters_structure(adapters_config)

    def _validate_adapters_structure(self, adapters_config: dict[str, Any]) -> None:
        """Validate the structure of adapters configuration."""
        for mode, adapter in adapters_config.items():
            if not isinstance(adapter, dict):
                raise ValueError(f"Adapter configuration for mode '{mode}' must be a dictionary")

            allowed_keys = self.ADAPTERS_SECTIONS[mode]

            for key, value in adapter.items():
                # Skip validation for interpolated values (starting with '%' or '@')
                if isinstance(value, str) and value.startswith(("%", "@")):
                    continue

                if key not in allowed_keys:
                    raise ValueError(f"Invalid key '{key}' in adapter '{mode}'. Allowed keys for '{mode}' are: {allowed_keys}")


class Resolver:
    """
    Resolves stage-specific configurations from the main configuration.
    """

    TRAINER_STAGE_MODES = {
        Stage.FIT: [Mode.TRAIN, Mode.VAL],
        Stage.VALIDATE: [Mode.VAL],
        Stage.TEST: [Mode.TEST],
        Stage.PREDICT: [Mode.PREDICT],
    }

    TUNER_STAGE_MODES = {
        Stage.LR_FIND: [Mode.TRAIN, Mode.VAL],
        Stage.SCALE_BATCH_SIZE: [Mode.TRAIN, Mode.VAL],
    }

    STAGE_MODES = TRAINER_STAGE_MODES | TUNER_STAGE_MODES

    def __init__(self, config: Config):
        self.config = config

    def get_stage_config(self, stage: str) -> Config:
        """Get stage-specific configuration by filtering unused components."""
        if stage not in self.STAGE_MODES:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {list(self.STAGE_MODES.keys())}")

        stage_config = self.config.get().copy()
        system_config = stage_config.get("system", {})
        dataloader_config = system_config.get("dataloaders", {})
        metrics_config = system_config.get("metrics", {})

        # Remove dataloaders not relevant to the current stage
        for mode in set(dataloader_config) - set(self.STAGE_MODES[stage]):
            dataloader_config.pop(mode, None)

        # Remove metrics not relevant to the current stage
        for mode in set(metrics_config) - set(self.STAGE_MODES[stage]):
            metrics_config.pop(mode, None)

        # Remove optimizer, scheduler, and criterion if not relevant to the current stage
        if stage in [Stage.VALIDATE, Stage.TEST, Stage.PREDICT]:
            if stage != Stage.VALIDATE:
                system_config.pop("criterion", None)
            system_config.pop("optimizer", None)
            system_config.pop("scheduler", None)

        # Retain only relevant args for the current stage
        if "args" in stage_config:
            stage_config["args"] = {stage: stage_config["args"].get(stage, {})}

        return Config(stage_config)


class Runner:
    """
    Executes the specified stage using the validated and resolved configurations.
    """

    def __init__(self):
        self.config: Config | None = None
        self.system: System | None = None
        self.trainer: Trainer | None = None
        self.args: dict[str, Any] | None = None

    def run(self, stage: str, config: str | dict | None = None, **config_overrides: Any) -> None:
        """Run the specified stage with the given configuration."""
        seed_everything()
        self.config = Config(config, **config_overrides)

        # Validate configuration
        Validator(self.config)

        # Resolve stage-specific configuration
        resolver = Resolver(self.config)

        # Setup stage
        self._setup_stage(stage, resolver)

        # Run stage
        self._run_stage(stage)

    def _run_stage(self, stage: str) -> None:
        """Execute the specified stage using the trainer or tuner."""
        if stage in Resolver.TRAINER_STAGE_MODES:
            stage_method = getattr(self.trainer, stage)
        elif stage in Resolver.TUNER_STAGE_MODES:
            stage_method = getattr(Tuner(self.trainer), stage)
        else:
            raise ValueError(f"Unsupported stage: {stage}")
        stage_method(self.system, **self.args)

    def _setup_stage(self, stage: str, resolver: Resolver) -> None:
        """Set up the system and trainer based on the stage-specific configuration."""
        # Get and parse stage-specific configuration
        stage_config = resolver.get_stage_config(stage)

        # Import project module if specified
        project_path = stage_config.get("project")
        if project_path:
            import_module_from_path("project", Path(project_path))

        # Initialize system
        self.system = stage_config.get_parsed_content("system")
        if not isinstance(self.system, System):
            raise ValueError("'system' must be an instance of System")

        # Initialize trainer
        self.trainer = stage_config.get_parsed_content("trainer")
        if not isinstance(self.trainer, Trainer):
            raise ValueError("'trainer' must be an instance of Trainer")

        # Set up arguments for the stage
        self.args = stage_config.get_parsed_content(f"args#{stage}", default={})

        # Save config to system checkpoint and trainer logger
        self._save_config()

    def _save_config(self) -> None:
        """Save config to system checkpoint and trainer logger."""
        if self.system:
            self.system.save_hyperparameters(self.config.get())
        if self.trainer and self.trainer.logger:
            self.trainer.logger.log_hyperparams(self.config.get())


def cli():
    runner = Runner()

    class Commands:
        def __init__(self):
            for stage in Stage:
                setattr(self, stage.value, self._make_command(stage))

        def _make_command(self, stage: Stage):
            def command(config: str, **config_overrides: Any):
                return runner.run(stage=stage.value, config=config, **config_overrides)

            command.__name__ = stage.value
            command.__doc__ = f"Run the '{stage.value}' stage."
            return command

    fire.Fire(Commands())
