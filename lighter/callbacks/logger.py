import sys
from pathlib import Path
from datetime import datetime
from yaml import safe_load
from typing import Any

from torch.distributed import broadcast_object_list, gather
from torch.utils import tensorboard
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from monai.utils.module import optional_import
from loguru import logger

from lighter import LighterSystem


class LighterLogger(Callback):

    def __init__(self, project, log_dir, tensorboard=False, wandb=False):
        self.project = project
        self.log_dir = Path(log_dir) / project / datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.log_dir = broadcast_object_list([self.log_dir], 0)[0]
        self.log_dir.mkdir(parents=True)

        self.tensorboard = tensorboard
        self.wandb = wandb

        # Not using Trainer's `global_step` as it counts the number of optimizer steps,
        # which becomes an issue when gradient accumulation is used.
        self.step_counter = {"train": 0, "val": 0, "test": 0}

        # Total loss
        self.loss = {"train": 0, "val": 0, "test": 0}

    @rank_zero_only
    def setup(self, trainer: Trainer, pl_module: LighterSystem, stage: str) -> None:
        if trainer.logger is not None:
            logger.warning("When using this logger it is advised to set Trainer(..., logger=None).")

        # Loguru log file
        logger.add(sink=self.log_dir / f"{stage}.log")
        
        # Load the dumped config file to add it to the loggers
        # config = safe_load(open(self.log_dir / "config.yaml"))

        # Tensorboard
        if self.tensorboard:
            tensorboard_dir = self.log_dir / "tensorboard"
            tensorboard_dir.mkdir()
            self.tensorboard = tensorboard.SummaryWriter(log_dir=tensorboard_dir)
            # self.tensorboard.add_hparams(config)
        
        # W&B
        if self.wandb:
            wandb, wandb_available = optional_import("wandb")
            if not wandb_available:
                logger.error("W&B not installed. To install it, run `pip install wandb`. Exiting.")
                sys.exit()
            wandb_dir = self.log_dir / "wandb"
            wandb_dir.mkdir()
            self.wandb = wandb.init(project=self.project, dir=wandb_dir)
            # self.wandb.config.update(config)

    @rank_zero_only
    def teardown(self, trainer: Trainer, pl_module: LighterSystem, stage: str) -> None:
        self.tensorboard.close()

    # Train step
    def on_train_batch_end(self, trainer: Trainer, pl_module: LighterSystem,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        self._on_batch_end(outputs, trainer)
        
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LighterSystem,
                                outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self._on_batch_end(outputs, trainer)

    def on_test_batch_end(self, trainer: Trainer, pl_module: LighterSystem,
                          outputs: Any, batch: Any, batch_idx: int) -> None:
        self._on_batch_end(outputs, trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        self._on_epoch_end("train", trainer, pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        print("Yooo")
        self._on_epoch_end("val", trainer, pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        self._on_epoch_end("test", trainer, pl_module)

    def _on_batch_end(self, outputs, trainer):
        if trainer.sanity_checking:
            return
        mode = outputs["mode"]
        # Should loss be gathered or not on batch end?
        # if distributed:
        #   gather(outputs["loss"])
        self.loss[mode] += outputs["loss"]
        if self.step_counter[mode] % trainer.log_every_n_steps == 0:
            self._log(outputs, global_step=self.step_counter[mode])
        self.step_counter[mode] += 1

    def _on_epoch_end(self, mode, trainer, pl_module):
        if trainer.sanity_checking:
            return
        loss = self.loss[mode]
        # if distributed:
        #   loss = gather(loss)
        loss = loss / self.step_counter[mode]
        metrics = getattr(pl_module, f"{mode}_metrics")
        outputs = {"mode": mode, "loss": loss, "metrics": metrics.compute()}
        self._log(outputs, is_epoch=True, global_step=self.step_counter[mode])
        metrics.reset()

    @rank_zero_only
    def _log(self, data: dict, global_step, is_epoch=False):
        step_or_epoch = "epoch" if is_epoch else "step"

        # Mode
        mode = data["mode"]

        # Loss
        loss = data["loss"]
        name = f"{mode}/loss/{step_or_epoch}"
        self.tensorboard.add_scalar(name, loss, global_step=global_step)
        self.wandb.log({name: loss}, step=global_step)

        # Metrics
        metrics = data["metrics"]
        for name, metric in metrics.items():
            name = f"{mode}/metric/{name}_{step_or_epoch}"
            self.tensorboard.add_scalar(name, metric, global_step=global_step)
            self.wandb.log({name: metric}, step=global_step)

        # Input

        # Target

        # Pred
