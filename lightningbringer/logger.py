
from datetime import datetime
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger, WandbLogger
from pytorch_lightning.loggers.base import rank_zero_experiment


class LightningBringerLogger(LightningLoggerBase):
    def __init__(self, save_dir, timestamp=None, project=None, wandb=True, tensorboard=True):
        
        if timestamp == "auto":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if timestamp is not None:
            save_dir += f"/{timestamp}"
        if project is not None:
            save_dir += f"_{project}"

        self._save_dir = save_dir

        self.tensorboard_logger = None
        if tensorboard:
            self.tensorboard_logger = TensorBoardLogger(
                save_dir=self._save_dir,
                name="",
                version="tensorboard"
            )
        
        self.wandb_logger = None
        if wandb:
            self.wandb_logger = WandbLogger(
                save_dir=self._save_dir,
                project=project,
                name=timestamp
            )

    @property
    def name(self):
        return "LightningBringerLogger"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        experiments = []
        if self.wandb_logger is not None:
            experiment.append(self.wandb_logger.experiment)
        if self.tensorboard_logger is not None:
            experiment.append(self.tensorboard_logger.experiment)
        return experiments

    @property
    def version(self):
        # Return the experiment version, int or str.
        return self._save_dir.split("/")[-1]

    @rank_zero_only
    def log_hyperparams(self, params, metrics=None):
        # TODO: metrics here for tensorboard, not wandb
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
