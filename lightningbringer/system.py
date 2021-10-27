import functools
import sys
from typing import Callable, List, Optional, Union

import pytorch_lightning as pl
import wandb
from loguru import logger
from torch.nn import Module, ModuleList
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from lightningbringer.utils import (collate_fn_replace_corrupted, get_name, preprocess_image,
                                    wrap_into_list)


class System(pl.LightningModule):

    def __init__(self,
                 model: Module,
                 batch_size: int,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 criterion: Optional[Callable] = None,
                 optimizers: Optional[Union[Optimizer, List[Optimizer]]] = None,
                 schedulers: Optional[Union[Callable, List[Callable]]] = None,
                 metrics: Optional[Union[Callable, List[Callable]]] = None,
                 train_dataset: Optional[Union[Dataset, List[Dataset]]] = None,
                 val_dataset: Optional[Union[Dataset, List[Dataset]]] = None,
                 test_dataset: Optional[Union[Dataset, List[Dataset]]] = None,
                 log_input_as: Optional[str] = None,
                 log_target_as: Optional[str] = None,
                 log_pred_as: Optional[str] = None):

        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.criterion = criterion
        self.optimizers = wrap_into_list(optimizers)
        self.schedulers = wrap_into_list(schedulers)
        self.metrics = ModuleList(wrap_into_list(metrics))

        self.log_input_as = log_input_as
        self.log_target_as = log_target_as
        self.log_pred_as = log_pred_as

        # `train_dataloader()`and `training_step()` are defined in `self.setup()`.
        #  LightningModule checks for them at init, these prevent it from complaining.
        self.train_dataloader = lambda: None
        self.training_step = lambda: None

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx, mode):
        input, target = batch
        pred = self(input)

        loss = self.criterion(pred, target) if mode != "test" else None
        metrics = [metric(pred, target) for metric in self.metrics]

        self._log(mode, input, target, pred, metrics, loss)

        return loss

    def _dataloader(self, mode):
        """Instantiate the dataloader for a mode (train/val/test).
        Includes a collate function that enables the DataLoader to replace
        None's (alias for corrupted examples) in the batch with valid examples.
        To make use of it, write a try-except in your Dataset that handles
        corrupted data and returns None instead.

        Args:
            mode (str): mode for which to create the dataloader. ['train', 'val', 'test']

        Returns:
            torch.utils.data.DataLoader: instantiated DataLoader
        """
        dataset = getattr(self, f"{mode}_dataset")
        # A dataset can return None when a corrupted example occurs. This collate
        # function replaces them with valid examples from the dataset.
        collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=dataset)
        return DataLoader(dataset,
                          shuffle=(mode == "train"),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          collate_fn=collate_fn)

    def configure_optimizers(self):
        if self.optimizers is None:
            logger.error("Please specify 'optimizers' in the config. Exiting.")
            sys.exit()
        if self.schedulers is None:
            return self.optimizers
        return self.optimizers, self.schedulers

    def setup(self, stage):
        dataset_required_by_stage = {
            "fit": "train_dataset",
            "validate": "val_dataset",
            "test": "test_dataset"
        }
        dataset_name = dataset_required_by_stage[stage]
        if getattr(self, dataset_name) is None:
            logger.error(f"Please specify '{dataset_name}' in the config. Exiting.")
            sys.exit()

        # Definition of stage-specific PyTorch Lightning methods.
        # Dynamically defined to enable flexible configuration system.

        # Training methods.
        if stage == "fit":
            self.train_dataloader = functools.partial(self._dataloader, mode="train")
            self.training_step = functools.partial(self._step, mode="train")

        # Validation methods. Required in 'validate' stage and optionally in 'fit' stage.
        if stage == "validate" or (stage == "fit" and self.val_dataset is not None):
            self.val_dataloader = functools.partial(self._dataloader, mode="val")
            self.validation_step = functools.partial(self._step, mode="val")

        # Test methods.
        if stage == "test":
            self.test_dataloader = functools.partial(self._dataloader, mode="test")
            self.test_step = functools.partial(self._step, mode="test")

    def _log(self, mode, input, target, pred, metrics=None, loss=None):

        def log_by_type(data, name, data_type, on_step=True, on_epoch=True):
            # Scalars
            if data_type == "scalar":
                self.log(name, data, on_step=on_step, on_epoch=on_epoch)

            # Temporary, https://github.com/PyTorchLightning/pytorch-lightning/issues/6720
            # Images
            elif data_type in ["image_single", "image_batch"]:
                for lgr in self.logger:
                    image = data[0:1] if data_type == "image_single" else data
                    image = preprocess_image(image)
                    if isinstance(lgr, pl.loggers.WandbLogger):
                        # Temporary, log every 50 steps
                        if self.global_step % 50:
                            lgr.experiment.log({name: wandb.Image(image)})
            else:
                raise NotImplementedError(f"'data_type' '{data_type}' not supported.")

        # Loss
        if loss is not None:
            log_by_type(loss, name=f"{mode}/loss", data_type="scalar")

        # Metrics
        if metrics:
            for metric, metric_fn in zip(metrics, self.metrics):
                name = get_name(metric_fn)
                log_by_type(metric, name=f"{mode}/metric_{name}", data_type="scalar")

        # Input, target, pred
        for key, value in {"input": input, "target": target, "pred": pred}.items():
            log_as = getattr(self, f"log_{key}_as")
            if log_as is not None:
                log_by_type(value, name=f"{mode}/{key}", data_type=log_as)
