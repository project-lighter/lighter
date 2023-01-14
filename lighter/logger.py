from pathlib import Path
from typing import Optional

from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger, WandbLogger
from pytorch_lightning.loggers.logger import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


class LighterLogger(LightningLoggerBase):
    def __init__(
        self, save_dir: str, tensorboard: bool = False, wandb: bool = False, wandb_project: Optional[str] = None
    ):
        """Logger that unifies tensorboard and wandb loggers.

        Args:
            save_dir (str): path to the directory where the logging data is stored.
            tensorboard (bool, optional): whether to use tensorboard. Defaults to False.
            wandb (bool, optional): whether to use wandb. Defaults to False.
            wandb_project (str, optional): wandb project name. Defaults to None.
        """
        super().__init__()
        assert True in [tensorboard, wandb], "You need to use at least one logger!"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self._save_dir = save_dir

        self.tensorboard_logger = None
        if tensorboard:
            self.tensorboard_logger = TensorBoardLogger(save_dir=self._save_dir, name="", version="tensorboard")

        self.wandb_logger = None
        if wandb:
            self.wandb_logger = WandbLogger(save_dir=self._save_dir, project=wandb_project)

    @property
    def name(self):
        return ""

    @property
    @rank_zero_experiment
    def experiment(self):
        experiments = {}
        if self.wandb_logger is not None:
            experiments["wandb"] = self.wandb_logger.experiment
        if self.tensorboard_logger is not None:
            experiments["tensorboard"] = self.wandb_logger.experiment
        return experiments

    @property
    def version(self):
        return self._save_dir.split("/")[-1]

    @rank_zero_only
    def log_hyperparams(self, params, metrics=None):
        if self.wandb_logger is not None:
            self.wandb_logger.log_hyperparams(params)

        if self.tensorboard_logger is not None:
            self.tensorboard_logger.log_hyperparams(params, metrics)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if self.wandb_logger is not None:
            self.wandb_logger.log_metrics(metrics, step)

        if self.tensorboard_logger is not None:
            self.tensorboard_logger.log_metrics(metrics, step)

    @rank_zero_only
    def save(self):
        # Note: Wandb does not do save()
        super().save()
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.save()

    @rank_zero_only
    def finalize(self, status):
        if self.wandb_logger is not None:
            self.wandb_logger.finalize(status)

        if self.tensorboard_logger is not None:
            self.tensorboard_logger.finalize(status)
