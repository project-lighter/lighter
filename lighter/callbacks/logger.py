import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

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

    def __init__(self, project, log_dir, log_input_as=None, log_target_as=None, log_pred_as=None, tensorboard=False, wandb=False):
        self.project = project
        # Only used on rank 0, the dir is created in setup().
        self.log_dir = Path(log_dir) / project / datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_as = {"input": log_input_as, "target": log_target_as, "pred": log_pred_as}

        self.tensorboard = tensorboard
        self.wandb = wandb

        # Global steps. Replaces `Trainer.global_step` because it counts optimizer steps
        # instead of batch steps, which can be problematic when using gradient accumulation.
        self.global_step_counter = {"train": 0, "val": 0, "test": 0}
        # Epoch steps. Resets at each epoch.
        self.epoch_step_counter = {"train": 0, "val": 0, "test": 0}
        # Running loss. Resets at each epoch.
        self.loss = {"train": 0, "val": 0, "test": 0}

    def setup(self, trainer: Trainer, pl_module: LighterSystem, stage: str) -> None:
        if trainer.logger is not None:
            logger.error("When using LighterLogger, set Trainer(logger=None).")
            sys.exit()

        if dist.is_initialized() and dist.get_rank() != 0:
            return

        self.log_dir.mkdir(parents=True)

        # Load the dumped config file to log it to the loggers.
        # config = safe_load(open(self.log_dir / "config.yaml"))

        # Loguru log file.
        logger.add(sink=self.log_dir / f"{stage}.log")

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
                logger.error("W&B not installed. To install it, run `pip install wandb`. Exiting.")
                sys.exit()
            wandb_dir = self.log_dir / "wandb"
            wandb_dir.mkdir()
            self.wandb = OPTIONAL_IMPORTS["wandb"].init(project=self.project, dir=wandb_dir)
            # self.wandb.config.update(config)

    def teardown(self, trainer: Trainer, pl_module: LighterSystem, stage: str) -> None:
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        self.tensorboard.close()

    def _log(self, outputs: dict, mode: str, global_step, is_epoch=False):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        step_or_epoch = "epoch" if is_epoch else "step"

        # Loss
        loss = outputs["loss"]
        name = f"{mode}/loss/{step_or_epoch}"
        self._log_scalar(name, loss, global_step)

        # Metrics
        metrics = outputs["metrics"]
        for name, metric in metrics.items():
            name = f"{mode}/metrics/{name}_{step_or_epoch}"
            self._log_scalar(name, metric, global_step)

        # Epoch does not log input, target, and pred.
        if is_epoch:
            return

        # Input, Target, Pred
        for data_name in ["input", "target", "pred"]:
            if self.log_as[data_name] is None:
                continue

            log_as = self.log_as[data_name]
            data = outputs[data_name]
            name = f"{mode}/data/{data_name}_{step_or_epoch}"

            # Scalar
            if log_as == "scalar":
                self._log_scalar(name, data, global_step)

            # Image
            elif log_as.startswith("image_"):
                # Check if the data type is valid.
                check_image_data_type(data, data_name)
                # Check if the `log_as` format is correct.
                if not re.match("image_\d+", log_as):
                    logger.error(f"`log_{data_name}_as` needs to be in `image_N` format where `N` is an integer")
                    sys.exit()
                # Number of images to extract from the batch.
                n_images_to_log = int(log_as.split("_")[1])
                for identifier, image in parse_image_data(data):
                    name = name if identifier is None else f"{name}_{identifier}"
                    # Slice to `n_images_to_log` only if it less than the batch size.
                    if n_images_to_log < image.shape[0]:
                        image = image[:min(len(image), n_images_to_log)]
                    # Preprocess a batch of images into a single, loggable, image.
                    image = preprocess_image(image)
                    self._log_image(name, image, global_step)
            else:
                logger.error(f"`log_{data_name}_as` does not support `{log_as}`.")
                sys.exit()

    def _log_scalar(self, name, scalar, global_step):
        if self.tensorboard:
            self.tensorboard.add_scalar(name, scalar, global_step=global_step)
        if self.wandb:
            self.wandb.log({name: scalar}, step=global_step)

    def _log_image(self, name, image, global_step):
        if self.tensorboard:
            self.tensorboard.add_image(name, image, global_step=global_step)
        if self.wandb:
            self.wandb.log({name: OPTIONAL_IMPORTS["wandb"].Image(image)}, step=global_step)

    def _on_batch_end(self, outputs: Any, trainer: Trainer):
        if not trainer.sanity_checking:
            mode = LIGHTNING_TO_LIGHTER_STAGE[trainer.state.stage]
            # Accumulate the loss.
            self.loss[mode] += outputs["loss"].item()
            # Logging frequency.
            if self.global_step_counter[mode] % trainer.log_every_n_steps == 0:
                # Log. Done only on rank 0.
                self._log(outputs, mode, global_step=self._get_global_step(trainer))
            # Increment the step counters.
            self.global_step_counter[mode] += 1
            self.epoch_step_counter[mode] += 1

    def _on_epoch_end(self, trainer: Trainer, pl_module: LighterSystem):
        if not trainer.sanity_checking:
            mode = LIGHTNING_TO_LIGHTER_STAGE[trainer.state.stage]
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
            # Get the torchmetrics.
            metrics = getattr(pl_module, f"{mode}_metrics")
            # Outputs to log. Compute the metrics over the epoch.
            outputs = {"loss": loss, "metrics": metrics.compute()}
            # Log. Done only on rank 0.
            self._log(outputs, mode, is_epoch=True, global_step=self._get_global_step(trainer))
            # Reset the metrics for the next epoch.
            metrics.reset()

    def _get_global_step(self, trainer: Trainer) -> int:
        mode = LIGHTNING_TO_LIGHTER_STAGE[trainer.state.stage]
        # When validating in Trainer.fit(), return the train steps instead of the
        # val steps to correctly log the validation steps against training steps.
        if mode == "val" and trainer.state.fn == "fit":
            return self.global_step_counter["train"]
        return self.global_step_counter[mode]

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        # Reset the epoch step counter and the loss for the next epoch.
        self.epoch_step_counter["train"] = 0
        self.loss["train"] = 0

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        # Reset the epoch step counter and the loss for the next epoch.
        self.epoch_step_counter["val"] = 0
        self.loss["val"] = 0

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LighterSystem) -> None:
        # Reset the epoch step counter and the loss for the next epoch.
        self.epoch_step_counter["test"] = 0
        self.loss["test"] = 0

    def on_train_batch_end(self, trainer: Trainer, pl_module: LighterSystem, outputs: Any,
                           batch: Any, batch_idx: int) -> None:
        self._on_batch_end(outputs, trainer)

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LighterSystem, outputs: Any,
                                batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self._on_batch_end(outputs, trainer)

    def on_test_batch_end(self, trainer: Trainer, pl_module: LighterSystem, outputs: Any,
                          batch: Any, batch_idx: int) -> None:
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
        logger.error(f"`{data_name}` has to be a Tensor, List[Tensors], Dict[str, Tensor]"
                        f", or Dict[str, List[Tensor]]. `{type(data)}` is not supported.")
        sys.exit()

def parse_image_data(data: Union[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], List[torch.Tensor], torch.Tensor]) -> List[Tuple[str, torch.Tensor]]:
    """Given input data, this function will parse it and return a list of tuples where 
    each tuple contains an identifier and a tensor.

    Args:
        data (Union[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], List[torch.Tensor], torch.Tensor]): image tensor(s).

    Returns:
        List[Tuple[str, torch.Tensor]]: a list of tuples where the first element is a string identifier and the
            second an image tensor.
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