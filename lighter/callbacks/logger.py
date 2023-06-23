from typing import Any, Dict, Union

import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from loguru import logger
from monai.bundle.config_parser import ConfigParser
from monai.utils.module import optional_import
from pytorch_lightning import Callback, Trainer

from lighter import LighterSystem
from lighter.callbacks.utils import get_lighter_mode, is_data_type_supported, parse_data, preprocess_image
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
            logger.error("When using LighterLogger, set Trainer(logger=None).")
            sys.exit()

        if not trainer.is_global_zero:
            return

        self.log_dir.mkdir(parents=True)
        logger.info(f"Logging to {self.log_dir}")

        # Loguru log file.
        logger.add(sink=self.log_dir / f"{stage}.log")

        # Tensorboard initialization.
        if self.tensorboard:
            # Tensorboard is a part of PyTorch, no need to check if it is not available.
            OPTIONAL_IMPORTS["tensorboard"], _ = optional_import("torch.utils.tensorboard")
            tensorboard_dir = self.log_dir / "tensorboard"
            tensorboard_dir.mkdir()
            self.tensorboard = OPTIONAL_IMPORTS["tensorboard"].SummaryWriter(log_dir=tensorboard_dir)
            # self.tensorboard.add_hparams(config, {}) # TODO: Tensorboard asks for a metric dict along with hparam dict
            # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams

        # Wandb initialization.
        if self.wandb:
            OPTIONAL_IMPORTS["wandb"], wandb_available = optional_import("wandb")
            if not wandb_available:
                logger.error("Weights & Biases not installed. To install it, run `pip install wandb`. Exiting.")
                sys.exit()
            wandb_dir = self.log_dir / "wandb"
            wandb_dir.mkdir()
            self.wandb = OPTIONAL_IMPORTS["wandb"].init(project=self.project, dir=wandb_dir, config=self.config)

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
        if not trainer.sanity_checking:
            mode = get_lighter_mode(trainer.state.stage)
            # Accumulate the loss.
            if mode in ["train", "val"]:
                self.loss[mode] += outputs["loss"].item()
            # Logging frequency. Log only on rank 0.
            if trainer.is_global_zero and self.global_step_counter[mode] % trainer.log_every_n_steps == 0:
                # Get global step.
                global_step = self._get_global_step(trainer)

                # Log loss.
                if outputs["loss"] is not None:
                    self._log_scalar(f"{mode}/loss/step", outputs["loss"], global_step)

                # Log metrics.
                if outputs["metrics"] is not None:
                    for name, metric in outputs["metrics"].items():
                        self._log_scalar(f"{mode}/metrics/{name}/step", metric, global_step)

                # Log input, target, and pred.
                for name in ["input", "target", "pred"]:
                    if self.log_types[name] is None:
                        continue
                    # Ensure data is of a valid type.
                    if not is_data_type_supported(outputs[name]):
                        raise ValueError(
                            f"`{name}` has to be a Tensor, List[Tensor], Tuple[Tensor],  Dict[str, Tensor], "
                            f"Dict[str, List[Tensor]], or Dict[str, Tuple[Tensor]]. `{type(outputs[name])}` is not supported."
                        )
                    for identifier, item in parse_data(outputs[name]).items():
                        item_name = f"{mode}/data/{name}" if identifier is None else f"{mode}/data/{name}_{identifier}"
                        self._log_by_type(item_name, item, self.log_types[name], global_step)

            # Increment the step counters.
            self.global_step_counter[mode] += 1
            if mode in ["train", "val"]:
                self.epoch_step_counter[mode] += 1

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

            # Loss
            if mode in ["train", "val"]:
                # Get the accumulated loss.
                loss = self.loss[mode]
                # Reduce the loss and average it on each rank.
                loss = trainer.strategy.reduce(loss, reduce_op="mean")
                # Divide the accumulated loss by the number of steps in the epoch.
                loss /= self.epoch_step_counter[mode]

            # Metrics
            # Get the torchmetrics.
            metric_collection = getattr(pl_module, f"{mode}_metrics")
            if metric_collection is not None:
                # Compute the epoch metrics.
                metrics = metric_collection.compute()
                # Reset the metrics for the next epoch.
                metric_collection.reset()

            # Log. Only on rank 0.
            if trainer.is_global_zero:
                # Get global step.
                global_step = self._get_global_step(trainer)

                # Log epoch number.
                self._log_scalar("epoch", trainer.current_epoch, global_step)

                # Log loss.
                if loss is not None:
                    self._log_scalar(f"{mode}/loss/epoch", loss, global_step)

                # Log metrics.
                if metrics is not None:
                    for name, metric in metrics.items():
                        self._log_scalar(f"{mode}/metrics/{name}/epoch", metric, global_step)

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
