from pathlib import Path
from typing import Dict

import torch
import torchvision
from loguru import logger

from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger, WandbLogger
from pytorch_lightning.loggers.logger import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


class LighterLogger(LightningLoggerBase):

    def __init__(self,
                 save_dir: str,
                 tensorboard: bool = False,
                 wandb: bool = False,
                 wandb_project: str = None):
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
            self.tensorboard_logger = TensorBoardLogger(save_dir=self._save_dir,
                                                        name="",
                                                        version="tensorboard")

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


def preprocess_image(image: torch.Tensor) -> torch.Tensor:
    """Preprocess the image before logging it. If it is a batch of multiple images,
    it will create a grid image of them. In case of 3D, a single image is displayed
    with slices stacked vertically, while a batch as a grid where each column is
    a different 3D image.

    Args:
        image (torch.Tensor): 2D or 3D image tensor.

    Returns:
        torch.Tensor: image ready for logging.
    """
    image = image.detach().cpu()
    # 3D image (NCDHW)
    has_three_dims = image.ndim == 5
    if has_three_dims:
        shape = image.shape
        image = image.view(shape[0], shape[1], shape[2] * shape[3], shape[4])
    # If more than one image, create a grid
    if image.shape[0] > 1:
        nrow = image.shape[0] if has_three_dims else 8
        image = torchvision.utils.make_grid(image, nrow=nrow)
    return image


def debug_message(mode: str, input: torch.Tensor, target: torch.Tensor, pred: torch.Tensor,
                  metrics: Dict, loss: torch.Tensor) -> None:
    """Logs the debug message.

    Args:
        mode (str): the mode of the system.
        input (torch.Tensor): input.
        target (torch.Tensor): target.
        pred (torch.Tensor): prediction.
        metrics (Dict): a dict where keys are the metric names and values the measured values.
        loss (torch.Tensor): calculated loss.
    """
    msg = f"\n----------- Debugging Output -----------\nMode: {mode}"
    for name, data in {"input": input, "target": target, "pred": pred}.items():
        if isinstance(data, list):
            msg += f"\n\n{name.capitalize()} is a {type(data).__name__} of {len(data)} elements."
            for idx, tensor in enumerate(data):
                if is_tensor_debug_loggable(tensor):
                    msg += f"\nTensor {idx} shape and value:\n{tensor.shape}\n{tensor}"
                else:
                    msg += f"\n*Tensor {idx} is too big to log."
        else:
            if is_tensor_debug_loggable(data):
                msg += f"\n\n{name.capitalize()} tensor shape and value:\n{data.shape}\n{data}"
            else:
                msg += f"\n\n*{name.capitalize()} tensor is too big to log."
    msg += f"\n\nLoss:\n{loss}"
    msg += f"\n\nMetrics:\n{metrics}"
    logger.debug(msg)


def is_tensor_debug_loggable(tensor):
    """A tensor is loggable for debugging if its shape is smaller than 16 in each axis."""
    return (torch.tensor(tensor.shape[1:]) < 16).all()
