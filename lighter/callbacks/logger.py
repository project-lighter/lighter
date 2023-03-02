from typing import Any, Dict, Union

import sys
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger
from monai.utils.module import optional_import
from pytorch_lightning import Callback, Trainer

from lighter import LighterSystem
from lighter.callbacks.utils import check_supported_data_type, get_lighter_mode, parse_data, preprocess_image

OPTIONAL_IMPORTS = {}


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

        self.data_types = {"input": input_type, "target": target_type, "pred": pred_type}
        # Max number of samples from the batch to log.
        self.max_samples = max_samples

        self.tensorboard = tensorboard
        self.wandb = wandb

        # Running loss. Resets at each epoch. Loss is not calculated for `test` mode.
        self.loss = {"train": 0, "val": 0}
        # Epoch steps. Resets at each epoch. No `test` mode step counter as loss is not calculated for it.
        self.epoch_step_counter = {"train": 0, "val": 0}
        # Global steps. Replaces `Trainer.global_step` because it counts optimizer steps
        # instead of batch steps, which can be problematic when using gradient accumulation.
        self.global_step_counter = {"train": 0, "val": 0, "test": 0}

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
            logger.error("When using LighterLogger, set Trainer(logger=None).")
            sys.exit()

        if not trainer.is_global_zero:
            return

        self.log_dir.mkdir(parents=True)

        # Load the dumped config file to log it to the loggers.
        # config = yaml.safe_load(open(self.log_dir / "config.yaml"))

        # Loguru log file.
        # logger.add(sink=self.log_dir / f"{stage}.log")

        # Tensorboard initialization.
        if self.tensorboard:
            # Tensorboard is a part of PyTorch, no need to check if it is not available.
            OPTIONAL_IMPORTS["tensorboard"], _ = optional_import("torch.utils.tensorboard")
            tensorboard_dir = self.log_dir / "tensorboard"
            tensorboard_dir.mkdir()
            self.tensorboard = OPTIONAL_IMPORTS["tensorboard"].SummaryWriter(log_dir=tensorboard_dir)
            # self.tensorboard.add_hparams(config)

        # Wandb initialization.
        if self.wandb:
            OPTIONAL_IMPORTS["wandb"], wandb_available = optional_import("wandb")
            if not wandb_available:
                logger.error("Weights & Biases not installed. To install it, run `pip install wandb`. Exiting.")
                sys.exit()
            wandb_dir = self.log_dir / "wandb"
            wandb_dir.mkdir()
            self.wandb = OPTIONAL_IMPORTS["wandb"].init(project=self.project, dir=wandb_dir)
            # self.wandb.config.update(config)

    def teardown(self, trainer: Trainer, pl_module: LighterSystem, stage: str) -> None:
        if not trainer.is_global_zero:
            return
        if self.tensorboard:
            self.tensorboard.close()

    def _log(self, outputs: dict, mode: str, global_step: int, is_epoch=False) -> None:
        """Logs the outputs to TensorBoard and Weights & Biases (if enabled).
        The outputs are logged as scalars and images, depending on the configuration.

        Args:
            outputs (dict): model outputs.
            mode (str): current mode (train/val/test).
            global_step (int): current global step.
            is_epoch (bool): whether the log is being done at the end
                of an epoch or astep. Default is False.
        """
        step_or_epoch = "epoch" if is_epoch else "step"

        # Loss
        if outputs["loss"] is not None:
            name = f"{mode}/loss/{step_or_epoch}"
            self._log_scalar(name, outputs["loss"], global_step)

        # Metrics
        if outputs["metrics"] is not None:
            for name, metric in outputs["metrics"].items():
                name = f"{mode}/metrics/{name}_{step_or_epoch}"
                self._log_scalar(name, metric, global_step)

        # Epoch does not log input, target, and pred.
        if is_epoch:
            self._log_scalar("epoch", outputs["epoch"], global_step)
            return

        # Input, Target, Pred
        for data_name in ["input", "target", "pred"]:
            if self.data_types[data_name] is None:
                continue
            self._log_by_type(data_name, outputs, mode, step_or_epoch, global_step)

    def _log_by_type(self, data_name: str, outputs: dict, mode: str, step_or_epoch: str, global_step: int) -> None:
        """Logs the data to TensorBoard and Weights & Biases (if enabled).
        The data is logged as scalars, images, or histograms, depending on the configuration.
        """
        data_type = self.data_types[data_name]
        data = outputs[data_name]
        tag = f"{mode}/data/{data_name}/{step_or_epoch}"

        # Scalar
        if data_type == "scalar":
            self._log_scalar(tag, data, global_step)

        # Image
        elif data_type == "image":
            # Check if the data type is valid.
            check_supported_data_type(data, data_name)
            for identifier, image in parse_data(data).items():
                item_name = tag if identifier is None else f"{tag}_{identifier}"
                # Slice to `max_samples` only if it less than the batch size.
                if self.max_samples is not None and self.max_samples < image.shape[0]:
                    image = image[: self.max_samples]
                # Preprocess a batch of images into a single, loggable, image.
                image = preprocess_image(image)
                self._log_image(item_name, image, global_step)

        # Histogram
        elif data_type == "histogram":
            check_supported_data_type(data, data_name)
            for identifier, tensor in parse_data(data).items():
                item_name = tag if identifier is None else f"{tag}_{identifier}"
                self._log_histogram(item_name, tensor, global_step)
        else:
            logger.error(f"`{data_name}_type` does not support `{data_type}`.")
            sys.exit()

    def _log_scalar(self, name: str, scalar: Union[int, float, torch.Tensor], global_step: int) -> None:
        """Logs the scalar to TensorBoard and Weights & Biases (if enabled).

        Args:
            name (str): name of the image to be logged.
            scalar (Union[int, float, torch.Tensor]): image to be logged.
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
        """Logs the image to TensorBoard and Weights & Biases (if enabled).

        Args:
            name (str): name of the image to be logged.
            image (torch.Tensor): image to be logged.
            global_step (int): current global step.
        """
        if self.tensorboard:
            self.tensorboard.add_image(name, image, global_step=global_step)
        if self.wandb:
            self.wandb.log({name: OPTIONAL_IMPORTS["wandb"].Image(image)}, step=global_step)

    def _log_histogram(self, name: str, tensor: torch.Tensor, global_step: int) -> None:
        """Logs the histogram to TensorBoard and Weights & Biases (if enabled).

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
        and, if specified how, the input, target, and pred data.

        Args:
            outputs (Dict): output dict from the model.
            trainer (Trainer): Trainer, passed automatically by PyTorch Lightning.
        """
        if not trainer.sanity_checking:
            mode = get_lighter_mode(trainer.state.stage)
            # Accumulate the loss.
            if mode in ["train", "val"]:
                self.loss[mode] += outputs["loss"].item()
            # Logging frequency. Log only on rank 0.
            if trainer.is_global_zero and self.global_step_counter[mode] % trainer.log_every_n_steps == 0:
                self._log(outputs, mode, global_step=self._get_global_step(trainer))
            # Increment the step counters.
            self.global_step_counter[mode] += 1
            if mode in ["train", "val"]:
                self.epoch_step_counter[mode] += 1

    def _on_epoch_end(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        """Performs logging at the end of an epoch. It calculates the average
        loss and metrics for the epoch and logs them. In distributed mode, it averages
        the losses and metrics from all ranks.

        Args:
            trainer (Trainer): Trainer, passed automatically by PyTorch Lightning.
            pl_module (LighterSystem): LighterSystem, passed automatically by PyTorch Lightning.
        """
        if not trainer.sanity_checking:
            mode = get_lighter_mode(trainer.state.stage)
            outputs = {"loss": None, "metrics": None, "epoch": trainer.current_epoch}

            # Loss
            if mode in ["train", "val"]:
                # Get the accumulated loss.
                loss = self.loss[mode]
                # Reduce the loss and average it on each rank.
                loss = trainer.strategy.reduce(loss, reduce_op="mean")
                # Divide the accumulated loss by the number of steps in the epoch.
                loss /= self.epoch_step_counter[mode]
                outputs["loss"] = loss

            # Metrics
            # Get the torchmetrics.
            metrics = getattr(pl_module, f"{mode}_metrics")
            if metrics is not None:
                # Compute the epoch metrics.
                outputs["metrics"] = metrics.compute()
                # Reset the metrics for the next epoch.
                metrics.reset()

            # Log. Only on rank 0.
            if trainer.is_global_zero:
                self._log(outputs, mode, is_epoch=True, global_step=self._get_global_step(trainer))

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
        self, trainer: Trainer, pl_module: LighterSystem, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self._on_batch_end(outputs, trainer)

    def on_test_batch_end(
        self, trainer: Trainer, pl_module: LighterSystem, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self._on_batch_end(outputs, trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        self._on_epoch_end(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        self._on_epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        self._on_epoch_end(trainer, pl_module)
