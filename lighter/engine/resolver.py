from lighter.engine.config import Config
from lighter.utils.types.enums import Mode, Stage


class Resolver:
    """
    Resolves stage-specific configurations from the main configuration.
    """

    STAGE_MODES = {
        Stage.FIT: [Mode.TRAIN, Mode.VAL],
        Stage.VALIDATE: [Mode.VAL],
        Stage.TEST: [Mode.TEST],
        Stage.PREDICT: [Mode.PREDICT],
        Stage.LR_FIND: [Mode.TRAIN, Mode.VAL],
        Stage.SCALE_BATCH_SIZE: [Mode.TRAIN, Mode.VAL],
    }

    def __init__(self, config: Config):
        self.config = config

    def get_stage_config(self, stage: str) -> Config:
        """Get stage-specific configuration by filtering unused components."""
        if stage not in self.STAGE_MODES:
            raise ValueError(f"Invalid stage: {stage}. Allowed stages are {list(self.STAGE_MODES)}")

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

        return Config(stage_config, validate=False)
