from typing import Any, Dict, List, Optional, Tuple, Union

import re
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision
from loguru import logger
from monai.utils.module import optional_import
from pytorch_lightning import Callback, Trainer
from torch.utils import tensorboard
from yaml import safe_load

from lighter import LighterSystem

LIGHTNING_TO_LIGHTER_STAGE = {"train": "train", "validate": "val", "test": "test"}
OPTIONAL_IMPORTS = {}


class LighterLogger(Callback):
    def __init__(
        self,
        project,
        log_dir,
        tensorboard=False,
        wandb=False,
        input_type=None,
        target_type=None,
        pred_type=None,
        max_samples=None,
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

        if dist.is_initialized() and dist.get_rank() != 0:
            return

        self.log_dir.mkdir(parents=True)

        # Load the dumped config file to log it to the loggers.
        # config = safe_load(open(self.log_dir / "config.yaml"))

        # Loguru log file.
        # logger.add(sink=self.log_dir / f"{stage}.log")

        # Tensorboard initialization.
        if self.tensorboard:
            tensorboard_dir = self.log_dir / "tensorboard"
            tensorboard_dir.mkdir()
            self.tensorboard = tensorboard.SummaryWriter(log_dir=tensorboard_dir)
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
        if dist.is_initialized() and dist.get_rank() != 0:
            return
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
        if dist.is_initialized() and dist.get_rank() != 0:
            return

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
            return

        # Input, Target, Pred
        for data_name in ["input", "target", "pred"]:
            if self.data_types[data_name] is None:
                continue

            data_type = self.data_types[data_name]
            data = outputs[data_name]
            name = f"{mode}/data/{data_name}_{step_or_epoch}"

            # Scalar
            if data_type == "scalar":
                self._log_scalar(name, data, global_step)

            # Image
            elif data_type == "image":
                # Check if the data type is valid.
                check_image_data_type(data, data_name)
                for identifier, image in parse_image_data(data):
                    name = name if identifier is None else f"{name}_{identifier}"
                    # Slice to `max_samples` only if it less than the batch size.
                    if self.max_samples is not None and self.max_samples < image.shape[0]:
                        image = image[: self.max_samples]
                    # Preprocess a batch of images into a single, loggable, image.
                    image = preprocess_image(image)
                    self._log_image(name, image, global_step)
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

    def _on_batch_end(self, outputs: Dict, trainer: Trainer) -> None:
        """Performs logging at the end of a batch/step. It logs the loss and metrics,
        and, if specified how, the input, target, and pred data.

        Args:
            outputs (Dict): output dict from the model.
            trainer (Trainer): Trainer, passed automatically by PyTorch Lightning.
        """
        if not trainer.sanity_checking:
            mode = LIGHTNING_TO_LIGHTER_STAGE[trainer.state.stage]
            # Accumulate the loss.
            if mode in ["train", "val"]:
                self.loss[mode] += outputs["loss"].item()
            # Logging frequency.
            if self.global_step_counter[mode] % trainer.log_every_n_steps == 0:
                # Log. Done only on rank 0.
                self._log(outputs, mode, global_step=self._get_global_step(trainer))
            # Increment the step counters.
            self.global_step_counter[mode] += 1
            if mode in ["train", "val"]:
                self.epoch_step_counter[mode] += 1

    def _on_epoch_end(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        """Performs logging at the end of an epoch. It calculates the average
        loss and metrics for the epoch and logs them. In distributed mode, it averages
        the losses and metrics from all processes.

        Args:
            trainer (Trainer): Trainer, passed automatically by PyTorch Lightning.
            pl_module (LighterSystem): LighterSystem, passed automatically by PyTorch Lightning.
        """
        if not trainer.sanity_checking:
            mode = LIGHTNING_TO_LIGHTER_STAGE[trainer.state.stage]
            outputs = {"loss": None, "metrics": None}

            # Loss
            if mode in ["train", "val"]:
                # Get the accumulated loss.
                loss = self.loss[mode]
                # Reduce the loss to rank 0 and average it.
                if dist.is_initialized():
                    # Distributed communication works only tensors.
                    loss = torch.tensor(loss).to(pl_module.device)
                    # On rank 0, sum the losses from all ranks. Other ranks remain with the same loss as before.
                    dist.reduce(loss, dst=0)
                    # On rank 0, average the loss sum by dividing it with the number of processes.
                    loss = loss.item() / dist.get_world_size() if dist.get_rank() == 0 else loss.item()
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

            # Log. Done only on rank 0.
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
        mode = LIGHTNING_TO_LIGHTER_STAGE[trainer.state.stage]
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
        # Reshape 3D image from NCDHW to NC(D*H)W format
        shape = image.shape
        image = image.view(shape[0], shape[1], shape[2] * shape[3], shape[4])
    if image.shape[0] == 1:
        image = image[0]
    else:
        # If more than one image, create a grid
        nrow = image.shape[0] if has_three_dims else 8
        image = torchvision.utils.make_grid(image, nrow=nrow)
    return image


def check_image_data_type(data: Any, name: str) -> None:
    """Check the input image data for its type. Valid image data types are:
        - torch.Tensor
        - List[torch.Tensor]
        - Dict[str, torch.Tensor]
        - Dict[str, List[torch.Tensor]]

    Args:
        data (Any): image data to check
        name (str): name of the image data, for logging purposes.
    """
    if isinstance(data, dict):
        is_valid = all(check_image_data_type(elem) for elem in data.values())
    elif isinstance(data, list):
        is_valid = all(check_image_data_type(elem) for elem in data)
    elif isinstance(data, torch.Tensor):
        is_valid = True
    else:
        is_valid = False

    if not is_valid:
        logger.error(
            f"`{name}` has to be a Tensor, List[Tensors], Dict[str, Tensor]"
            f", or Dict[str, List[Tensor]]. `{type(data)}` is not supported."
        )
        sys.exit()


def parse_image_data(
    data: Union[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], List[torch.Tensor], torch.Tensor]
) -> List[Tuple[Optional[str], torch.Tensor]]:
    """Given input data, this function will parse it and return a list of tuples where
    each tuple contains an identifier and a tensor.

    Args:
        data (Union[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], List[torch.Tensor], torch.Tensor]): image tensor(s).

    Returns:
        List[Tuple[Optional[str], torch.Tensor]]: a list of tuples where the first element is
            a string identifier (or `None` if there is only one image) and the second an image tensor.
    """
    result = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                for i, tensor in enumerate(value):
                    result.append((f"{key}_{i}", tensor) if len(value > 1) else (key, tensor))
            else:
                result.append((key, value))
    elif isinstance(data, list):
        for i, tensor in enumerate(data):
            result.append((str(i), tensor))
    else:
        result.append((None, data))
    return result
