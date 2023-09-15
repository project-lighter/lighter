from typing import Any, Dict, Union

import itertools
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

from lighter import LighterSystem
from lighter.callbacks.utils import get_lighter_mode, preprocess_image
from lighter.utils.dynamic_imports import OPTIONAL_IMPORTS


class LighterLogger(Callback):
    def __init__(
        self,
        project: str,
        log_dir: str,
        tensorboard: bool = False,
        wandb: bool = False,
        input_type: str = None,
        target_type: str = None,
        pred_type: str = None,
        max_samples: int = None,
    ) -> None:
        self.project = project
        # Only used on rank 0, the dir is created in setup().
        self.log_dir = Path(log_dir) / project / datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_types = {"input": input_type, "target": target_type, "pred": pred_type}
        # Max number of samples from the batch to log.
        self.max_samples = max_samples

        self.tensorboard = tensorboard
        self.wandb = wandb
        self.lr_monitor = LighterLearningRateMonitor()

        # Running loss. Resets at each epoch. Loss is not calculated for `test` mode.
        self.loss = {"train": 0, "val": 0}
        # Epoch steps. Resets at each epoch. No `test` mode step counter as loss is not calculated for it.
        self.epoch_step_counter = {"train": 0, "val": 0}
        # Global steps. Replaces `Trainer.global_step` because it counts optimizer steps
        # instead of batch steps, which can be problematic when using gradient accumulation.
        self.global_step_counter = {"train": 0, "val": 0, "test": 0}

        # Initialized from runner module
        self.config = None

    def setup(self, trainer: Trainer, pl_module: LighterSystem, stage: str) -> None:
        """
        Initialize logging for the LighterSystem.
        TODO: improve this docstring.

        Args:
            trainer (Trainer): Trainer, passed automatically by PyTorch Lightning.
            pl_module (LighterSystem): LighterSystem, passed automatically by PyTorch Lightning.
            stage (str): stage of the training process. Passed automatically by PyTorch Lightning.
        """
        if trainer.logger is not None:
            raise ValueError("When using LighterLogger, set Trainer(logger=None).")

        if not trainer.is_global_zero:
            return

        self.log_dir.mkdir(parents=True)
        logger.info(f"Logging to {self.log_dir}")

        # Loguru log file.
        logger.add(sink=self.log_dir / f"{stage}.log")

        # Tensorboard initialization.
        if self.tensorboard:
            tensorboard_dir = self.log_dir / "tensorboard"
            tensorboard_dir.mkdir()
            self.tensorboard = OPTIONAL_IMPORTS["tensorboard"].SummaryWriter(log_dir=tensorboard_dir)
            # self.tensorboard.add_hparams(config, {}) # TODO: Tensorboard asks for a metric dict along with hparam dict
            # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams

        # Wandb initialization.
        if self.wandb:
            wandb_dir = self.log_dir / "wandb"
            wandb_dir.mkdir()
            self.wandb = OPTIONAL_IMPORTS["wandb"].init(project=self.project, dir=wandb_dir, config=self.config)

    def on_train_start(self, trainer: Trainer, pl_module: LighterSystem):
        # Setup the learning rate monitor.
        self.lr_monitor.on_train_start(trainer=trainer)

    def teardown(self, trainer: Trainer, pl_module: LighterSystem, stage: str) -> None:
        if not trainer.is_global_zero:
            return
        if self.tensorboard:
            self.tensorboard.close()

    def _log_by_type(self, name: str, data: Any, log_type: str, global_step: int) -> None:
        """Logs data according to the specified type.

        Args:
            name (str): name of the data being logged.
            data (Any): data to be logged.
            log_type (str): type of the data. Supported types are 'scalar', 'image', and 'histogram'.
            global_step (int): current global step.

        Raises:
            TypeError: if the specified `log_type` is not supported.
        """
        # Log scalar
        if log_type == "scalar":
            self._log_scalar(name, data, global_step)

        # Log image
        elif log_type == "image":
            # Slice to `max_samples` only if it less than the batch size.
            if self.max_samples is not None and self.max_samples < data.shape[0]:
                data = data[: self.max_samples]
            # Preprocess a batch of images into a single, loggable, image.
            data = preprocess_image(data)
            self._log_image(name, data, global_step)

        # Log histogram
        elif log_type == "histogram":
            self._log_histogram(name, data, global_step)

        else:
            raise TypeError(f"`{log_type}` not supported for logging.")

    def _log_scalar(self, name: str, scalar: Union[int, float, torch.Tensor], global_step: int) -> None:
        """Logs the scalar.

        Args:
            name (str): name of the image to be logged.
            scalar (int, float, torch.Tensor): image to be logged.
            global_step (int): current global step.
        """
        if not isinstance(scalar, (int, float, torch.Tensor)):
            raise NotImplementedError("LighterLogger currently supports only single scalars.")
        if isinstance(scalar, torch.Tensor) and scalar.dim() > 0:
            raise NotImplementedError("LighterLogger currently supports only single scalars.")

        if isinstance(scalar, torch.Tensor):
            scalar = scalar.item()

        if self.tensorboard:
            self.tensorboard.add_scalar(name, scalar, global_step=global_step)
        if self.wandb:
            self.wandb.log({name: scalar}, step=global_step)

    def _log_image(self, name: str, image: torch.Tensor, global_step: int) -> None:
        """Logs the image.

        Args:
            name (str): name of the image to be logged.
            image (torch.Tensor): image to be logged.
            global_step (int): current global step.
        """
        image = image.detach().cpu()
        if self.tensorboard:
            self.tensorboard.add_image(name, image, global_step=global_step)
        if self.wandb:
            self.wandb.log({name: OPTIONAL_IMPORTS["wandb"].Image(image)}, step=global_step)

    def _log_histogram(self, name: str, tensor: torch.Tensor, global_step: int) -> None:
        """Logs the histogram.

        Args:
            name (str): name of the image to be logged.
            tensor (torch.Tensor): tensor to be logged.
            global_step (int): current global step.
        """
        tensor = tensor.detach().cpu()
        if self.tensorboard:
            self.tensorboard.add_histogram(name, tensor, global_step=global_step)
        if self.wandb:
            self.wandb.log({name: OPTIONAL_IMPORTS["wandb"].Histogram(tensor)}, step=global_step)

    def _on_batch_end(self, outputs: Dict, trainer: Trainer) -> None:
        """Performs logging at the end of a batch/step. It logs the loss and metrics,
        and, if their logging type is specified, the input, target, and pred data.

        Args:
            outputs (Dict): output dict from the model.
            trainer (Trainer): Trainer, passed automatically by PyTorch Lightning.
        """
        if trainer.sanity_checking:
            return

        mode = get_lighter_mode(trainer.state.stage)

        # Accumulate the loss.
        if mode in ["train", "val"]:
            self.loss[mode] += outputs["loss"].item()

        # Log only on rank 0 and according to the `log_every_n_steps` parameter. Otherwise, only increment the step counters.
        if not trainer.is_global_zero or self.global_step_counter[mode] % trainer.log_every_n_steps != 0:
            self._increment_step_counters(mode)
            return

        global_step = self._get_global_step(trainer)

        # Loss.
        if outputs["loss"] is not None:
            self._log_scalar(f"{mode}/loss/step", outputs["loss"], global_step)

        # Metrics.
        if outputs["metrics"] is not None:
            for name, metric in outputs["metrics"].items():
                self._log_scalar(f"{mode}/metrics/{name}/step", metric, global_step)

        # Input, target, and pred.
        for name in ["input", "target", "pred"]:
            if self.log_types[name] is not None:
                self._log_by_type(f"{mode}/data/{name}", outputs[name], self.log_types[name], global_step)

        # LR info. Logs at step if a scheduler's interval is step-based.
        if mode == "train":
            lr_stats = self.lr_monitor.get_stats(trainer, "step")
            for name, value in lr_stats.items():
                self._log_scalar(f"{mode}/optimizer/{name}/step", value, global_step)

        # Increment the step counters.
        self._increment_step_counters(mode)

    def _on_epoch_end(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        """Performs logging at the end of an epoch. Logs the epoch number, the loss, and the metrics.
        It averages the loss and metrics over the epoch. In distributed mode, it averages over all ranks.

        Args:
            trainer (Trainer): Trainer, passed automatically by PyTorch Lightning.
            pl_module (LighterSystem): LighterSystem, passed automatically by PyTorch Lightning.
        """
        if not trainer.sanity_checking:
            mode = get_lighter_mode(trainer.state.stage)
            loss, metrics = None, None

            # Get the accumulated loss over the epoch and processes.
            if mode in ["train", "val"]:
                loss = self.loss[mode]
                loss = trainer.strategy.reduce(loss, reduce_op="mean")
                loss /= self.epoch_step_counter[mode]

            # Get the torchmetrics.
            # TODO: Remove the "_" prefix when fixed https://github.com/pytorch/pytorch/issues/71203
            metric_collection = pl_module.metrics["_" + mode]
            if metric_collection is not None:
                # Compute the epoch metrics.
                metrics = metric_collection.compute()
                # Reset the metrics for the next epoch.
                metric_collection.reset()

            # Log only on rank 0.
            if not trainer.is_global_zero:
                return

            global_step = self._get_global_step(trainer)

            # Epoch number.
            self._log_scalar("epoch", trainer.current_epoch, global_step)

            # Loss.
            if loss is not None:
                self._log_scalar(f"{mode}/loss/epoch", loss, global_step)

            # Metrics.
            if metrics is not None:
                for name, metric in metrics.items():
                    self._log_scalar(f"{mode}/metrics/{name}/epoch", metric, global_step)

            # LR info. Logged at epoch if the scheduler's interval is epoch-based, or if no scheduler is used.
            if mode == "train":
                lr_stats = self.lr_monitor.get_stats(trainer, "epoch")
                for name, value in lr_stats.items():
                    self._log_scalar(f"{mode}/optimizer/{name}/epoch", value, global_step)

    def _get_global_step(self, trainer: Trainer) -> int:
        """Return the global step for the current mode. Note that when Trainer
        is running Trainer.fit() and is in `val` mode, this method will return
        the global step of the `train` mode in order to correctly log the validation
        steps against training steps.

        Args:
            trainer (Trainer): Trainer, passed automatically by PyTorch Lightning.

        Returns:
            int: global step.
        """
        mode = get_lighter_mode(trainer.state.stage)
        # When validating in Trainer.fit(), return the train steps instead of the
        # val steps to correctly
        if mode == "val" and trainer.state.fn == "fit":
            return self.global_step_counter["train"]
        return self.global_step_counter[mode]

    def _increment_step_counters(self, mode: str) -> None:
        """Increment the global step and epoch step counters for the specified mode.

        Args:
            mode (str): mode to increment the global step counter for.
        """
        self.global_step_counter[mode] += 1
        if mode in ["train", "val"]:
            self.epoch_step_counter[mode] += 1

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        # Reset the loss and the epoch step counter for the next epoch.
        self.loss["train"] = 0
        self.epoch_step_counter["train"] = 0

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        # Reset the loss and the epoch step counter for the next epoch.
        self.loss["val"] = 0
        self.epoch_step_counter["val"] = 0

    def on_train_batch_end(self, trainer: Trainer, pl_module: LighterSystem, outputs: Any, batch: Any, batch_idx: int) -> None:
        self._on_batch_end(outputs, trainer)

    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LighterSystem, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._on_batch_end(outputs, trainer)

    def on_test_batch_end(
        self, trainer: Trainer, pl_module: LighterSystem, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._on_batch_end(outputs, trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        self._on_epoch_end(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        self._on_epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        self._on_epoch_end(trainer, pl_module)


class LighterLearningRateMonitor(LearningRateMonitor):
    def __init__(self) -> None:
        # Instantiate LearningRateMonitor with default values.
        super().__init__()

    def on_train_start(self, trainer: Trainer) -> None:
        # Set the `self.log_momentum` flag based on the optimizers instead of manually through __init__().
        for key in ["momentum", "betas"]:
            if trainer.lr_scheduler_configs:
                self.log_momentum = any(key not in conf.scheduler.optimizer.defaults for conf in trainer.lr_scheduler_configs)
            else:
                self.log_momentum = any(key not in optimizer.defaults for optimizer in trainer.optimizers)
            if self.log_momentum:
                break

        # Find names for schedulers
        names = []
        result = self._find_names_from_schedulers(trainer.lr_scheduler_configs)
        sched_hparam_keys = result[0]
        optimizers_with_scheduler = result[1]
        optimizers_with_scheduler_types = result[2]
        names.extend(sched_hparam_keys)

        # Find names for leftover optimizers
        optimizer_hparam_keys, _ = self._find_names_from_optimizers(
            trainer.optimizers,
            seen_optimizers=optimizers_with_scheduler,
            seen_optimizer_types=optimizers_with_scheduler_types,
        )
        names.extend(optimizer_hparam_keys)

        # Initialize for storing values
        names_flatten = list(itertools.chain.from_iterable(names))
        self.lrs = {name: [] for name in names_flatten}
        self.last_momentum_values = {f"{name}-momentum": None for name in names_flatten}

    def get_stats(self, trainer: Trainer, interval: str) -> Dict[str, float]:
        # If a scheduler is not defined, log the learning rate over epochs.
        if not trainer.lr_scheduler_configs and interval == "step":
            return {}
        lr_stats = self._extract_stats(trainer, interval)
        # Remove 'lr-' prefix from keys and replace '-' with '/'.
        lr_stats = {k[3:].replace("-", "/"): v for k, v in lr_stats.items()}
        # Add '/lr' to the end of the keys that don't end with 'momentum'.
        lr_stats = {k if k.endswith("momentum") else f"{k}/lr": v for k, v in lr_stats.items()}
        return lr_stats
