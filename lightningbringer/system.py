import functools
import sys
from typing import Callable, List, Optional, Union

import pytorch_lightning as pl
import torch
from torch.nn import Module, ModuleList
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
import torchvision
from loguru import logger
import wandb

from lightningbringer.utils import (collate_fn_replace_corrupted, get_name,
                                    wrap_into_list, preprocess_image)


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

        # Placeholders for these methods. They are actually defined in `self.setup()`.
        # We do this to prevent PyTorch Lightning from complaining about not having
        # these methods defined since it checks for them before calling self.setup().
        self.train_dataloader = lambda: None
        self.training_step = lambda: None
        # Temporary, see self._log() for more comments
        self.next_epoch = 1

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx, mode):
        input, target = batch
        pred = self(input)

        loss = self.criterion(pred, target) if mode != "test" else None
        metrics = [metric(pred, target) for metric in self.metrics]

        self._log(mode, input, target, pred, metrics, loss)
        if mode != "test":
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

        # Training methods. They always need to be defined (PyTorch Lightning requirement).
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
            
            # Temporary, support for Wandb only, until PL 1.5:
            # https://github.com/PyTorchLightning/pytorch-lightning/issues/6720
            # Images
            elif self.logger is not None and str(data_type).startswith("image"):
                for lgr in self.logger:
                    image = data[0:1] if data_type == "image_single" else data
                    if isinstance(lgr, pl.loggers.WandbLogger):
                        # Once per epoch
                        if self.current_epoch == self.next_epoch:
                            lgr.experiment.log({name: wandb.Image(image)})
                            self.next_epoch += 1
            else:
                raise NotImplementedError(f"'data_type' {data_type} not supported.")

        # Loss
        if loss is not None:
            log_by_type(loss, name=f"{mode}/loss", data_type="scalar")

        # Metrics
        if metrics:
            for i in range(len(metrics)):
                metric = metrics[i]
                name = f"{mode}/metric_{get_name(self.metrics[i])}"
                log_by_type(metric, name=name, data_type="scalar")

        # # Input, target, pred
        if self.log_input_as is not None:
            log_by_type(input, name=f"{mode}/input", data_type=self.log_input_as)
        if self.log_target_as is not None:
            log_by_type(target, name=f"{mode}/target", data_type=self.log_target_as)
        if self.log_pred_as is not None:
            log_by_type(pred, name=f"{mode}/pred", data_type=self.log_pred_as)
