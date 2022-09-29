import sys
from functools import partial
from typing import Callable, List, Optional, Union

import pytorch_lightning as pl
import torch
import wandb
from loguru import logger
from torch.nn import Module, ModuleList
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Sampler
from torchmetrics import Metric

from lightningbringer.utils import (collate_fn_replace_corrupted, get_name, import_attr,
                                    preprocess_image, wrap_into_list, debug_message,
                                    reshape_pred_if_single_value_prediction)


class System(pl.LightningModule):

    def __init__(self,
                 model: Module,
                 batch_size: int,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 optimizers: Optional[Union[Optimizer, List[Optimizer]]] = None,
                 schedulers: Optional[Union[Callable, List[Callable]]] = None,
                 criterion: Optional[Callable] = None,
                 cast_label_dtype_to: Optional[str] = None,
                 post_criterion_activation: Optional[str] = None,
                 patch_based_inferer: Optional[Callable] = None,
                 train_metrics: Optional[Union[Metric, List[Metric]]] = None,
                 val_metrics: Optional[Union[Metric, List[Metric]]] = None,
                 test_metrics: Optional[Union[Metric, List[Metric]]] = None,
                 train_dataset: Optional[Union[Dataset, List[Dataset]]] = None,
                 val_dataset: Optional[Union[Dataset, List[Dataset]]] = None,
                 test_dataset: Optional[Union[Dataset, List[Dataset]]] = None,
                 train_sampler: Optional[Sampler] = None,
                 val_sampler: Optional[Sampler] = None,
                 test_sampler: Optional[Sampler] = None,
                 log_input_as: Optional[str] = None,
                 log_label_as: Optional[str] = None,
                 log_pred_as: Optional[str] = None,
                 debug: bool = False):

        super().__init__()
        self._init_placeholders_for_dataloader_and_step_methods()

        self.debug = debug

        # Model setup
        self.model = model
        self.batch_size = batch_size

        # Criterion, optimizer, and scheduler
        self.criterion = criterion
        self.optimizers = wrap_into_list(optimizers)
        self.schedulers = wrap_into_list(schedulers)

        # Datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Samplers
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler

        # Metrics
        self.train_metrics = ModuleList(wrap_into_list(train_metrics))
        self.val_metrics = ModuleList(wrap_into_list(val_metrics))
        self.test_metrics = ModuleList(wrap_into_list(test_metrics))

        # Criterion-specific activation function and data type casting
        self._post_criterion_activation = post_criterion_activation
        self._cast_label_dtype_to = cast_label_dtype_to

        self._patch_based_inferer = patch_based_inferer

        self._log_input_as = log_input_as
        self._log_label_as = log_label_as
        self._log_pred_as = log_pred_as

    def forward(self, x):
        """Forward pass. Allows calling self(x) to do it."""
        return self.model(x)

    def _step(self, batch, batch_idx, mode):
        """Step for all modes ('train', 'val', 'test')

        Args:
            batch (Tuple(torch.Tensor, Any)): output of the DataLoader.
            batch_idx (int): index of the batch. Not using it, but Pytorch Lightning requires it.
            mode (str): mode in which the system is. ['train', 'val', 'test']

        Returns:
            Union[torch.Tensor, None]: returns the calculated loss in training and validation step,
                and None in test step.
        """
        input, label = batch
        # Predict
        if self._patch_based_inferer and mode in ["val", "test"]:
            pred = self._patch_based_inferer(input, self)
        else:
            pred = self(input)

        pred = reshape_pred_if_single_value_prediction(pred, label)

        # Calculate the loss
        loss = None
        if mode != "test":
            loss = self.criterion(pred,
                                  label if label is None else label.to(self._cast_label_dtype_to))

        # Apply the post-criterion activation. Necessary for measuring the metrics
        # correctly in cases when using a criterion such as `BCELossWithLogits`` which
        # requires the model to output logits, i.e. non-activated outputs.
        if self._post_criterion_activation is not None:
            pred = self._post_criterion_activation(pred)

        # Calculate the metrics
        metrics = getattr(self, f"{mode}_metrics")
        step_metrics = {get_name(m): m(pred, label) for m in metrics}

        ## Logging part ##

        on_step = (mode != "val")

        # Metrics. Note that torchmetrics objects are passed.
        for metric in metrics:
            name = f"{mode}/metric_{get_name(metric)}"
            self._log_by_type(name, metric, "scalar", on_step=on_step, on_epoch=True)

        # Loss
        if loss is not None:
            name = f"{mode}/loss"
            self._log_by_type(name, loss, "scalar", on_step=on_step, on_epoch=True)

        # Input, label, pred
        for key in ["input", "label", "pred"]:
            data_type = getattr(self, f"_log_{key}_as")
            if data_type is not None:
                name = f"{mode}/{key}"
                self._log_by_type(name, eval(key), data_type, on_step=on_step, on_epoch=True)

        # Debug message
        if self.debug:
            debug_message(mode, input, label, pred, step_metrics, loss)

        return loss

    def _dataloader(self, mode):
        """Instantiate the dataloader for a mode (train/val/test).
        Includes a collate function that enables the DataLoader to replace
        None's (alias for corrupted examples) in the batch with valid examples.
        To make use of it, write a try-except in your Dataset that handles
        corrupted data by returning None instead.

        Args:
            mode (str): mode for which to create the dataloader. ['train', 'val', 'test']

        Returns:
            torch.utils.data.DataLoader: instantiated DataLoader.
        """
        dataset = getattr(self, f"{mode}_dataset")
        sampler = getattr(self, f"{mode}_sampler")

        # Batch size is 1 when using patch based inference for two reasons:
        # 1) Patch based inference splits an input into a batch of patches,
        # so the batch size will actually be defined for it;
        # 2) In settings where patch based inference is needed, the input data often
        # varies in shape, preventing the data loader to stack them into a batch.
        batch_size = self.batch_size
        if self._patch_based_inferer is not None and mode in ["val", "test"]:
            logger.info(f"Setting the general batch size to 1 for {mode} "
                        "mode because a patch-based inferer is used.")
            batch_size = 1

        # A dataset can return None when a corrupted example occurs. This collate
        # function replaces them with valid examples from the dataset.
        collate_fn = partial(collate_fn_replace_corrupted, dataset=dataset)
        return DataLoader(dataset,
                          sampler=sampler,
                          shuffle=(mode == "train" and sampler is None),
                          batch_size=batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          collate_fn=collate_fn)

    def configure_optimizers(self):
        """LightningModule method. Returns optimizers and, if defined, schedulers."""
        if not self.optimizers:
            logger.error("Please specify 'optimizers' in the config. Exiting.")
            sys.exit()
        if not self.schedulers:
            return self.optimizers
        return {"optimizer": self.optimizers, "lr_scheduler": self.schedulers}

    def setup(self, stage):
        """LightningModule method. Called after the initialization but before running the system.
        Here, it checks if the required dataset is provided in the config and sets up
        LightningModule methods for the stage (mode) in which the system is.

        Args:
            stage (str): passed by PyTorch Lightning. ['fit', 'validate', 'test']
                # TODO: update when all stages included
        """
        dataset_required_by_stage = {
            "fit": "train_dataset",
            "validate": "val_dataset",
            "test": "test_dataset",
            # "predict": TODO
        }
        dataset_name = dataset_required_by_stage[stage]
        if getattr(self, dataset_name) is None:
            logger.error(f"Please specify '{dataset_name}' in the config. Exiting.")
            sys.exit()

        # Stage-specific PyTorch Lightning methods. Defined dynamically so that the system
        # only has methods used in the stage and for which the configuration was provided.

        if not self._lightning_module_methods_defined:
            del self.train_dataloader, self.val_dataloader, self.test_dataloader
            del self.training_step, self.validation_step, self.test_step
            # `Trainer.tune()` calls the `self.setup()` method whenever it runs for a new
            #  parameter, and deleting the above methods again breaks it. This flag prevents it.
            self._lightning_module_methods_defined = True

        # Training methods.
        if stage in ["fit", "tune"]:
            self.train_dataloader = partial(self._dataloader, mode="train")
            self.training_step = partial(self._step, mode="train")

        # Validation methods. Required in 'validate' stage and optionally in 'fit' or 'tune' stage.
        if stage == "validate" or (stage in ["fit", "tune"] and self.val_dataset is not None):
            self.val_dataloader = partial(self._dataloader, mode="val")
            self.validation_step = partial(self._step, mode="val")

        # Test methods.
        if stage == "test":
            self.test_dataloader = partial(self._dataloader, mode="test")
            self.test_step = partial(self._step, mode="test")

    def _init_placeholders_for_dataloader_and_step_methods(self):
        """LightningModule checks for `..._dataloader()`and `..._step()` methods at init
        before calling `self.setup()`. However, in `System`, these methods are dynamically
        defined in `self.setup()`. To circumvent this, we set `..._dataloader()`
        and `..._step()` placeholders during the object's initialization.
        """
        self.train_dataloader = lambda: None
        self.val_dataloader = lambda: None
        self.test_dataloader = lambda: None
        self.training_step = lambda: None
        self.validation_step = lambda: None
        self.test_step = lambda: None
        self._lightning_module_methods_defined = False

    def _log_by_type(self, name, data, data_type, on_step=True, on_epoch=True):
        """Log data according to its type at each epoch and, during training,
        at each logging step.

        Args:
            name (str): the name under which the data will be logged.
            data (Any): data to log.
            data_type (str): type of the data to be logged.
                ['scalar', 'image_batch', 'image_single']  # TODO update when there's more
        """
        # Scalars
        if data_type == "scalar":
            # TODO: handle a batch of scalars
            self.log(name, data, on_step=on_step, on_epoch=on_epoch, sync_dist=True)

        # Temporary, https://github.com/PyTorchLightning/pytorch-lightning/issues/6720
        # Images
        elif data_type in ["image_single", "image_batch"]:
            for lgr in wrap_into_list(self.logger):
                image = data[0:1] if data_type == "image_single" else data
                image = preprocess_image(image)
                # TODO: handle logging frequency better, add tensorboard support
                if isinstance(lgr, pl.loggers.WandbLogger) and self.global_step % 50:
                    lgr.experiment.log({name: wandb.Image(image)})
        else:
            logger.error(f"type '{data_type}' not supported. Exiting.")
            sys.exit()
