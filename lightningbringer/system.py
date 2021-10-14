import functools
from typing import Callable, List, Optional, Union

import pytorch_lightning as pl
from torch.nn import Module, ModuleList
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from lightningbringer.utils import (collate_fn_replace_corrupted, get_name, wrap_into_list)


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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="test")

    def _step(self, batch, batch_idx, mode="train"):
        assert mode in ["train", "val", "test"]

        output = {"batch_idx": batch_idx}
        logs = {}

        input, target = batch
        pred = self(input)
        output.update({"input": input, "target": target, "pred": pred.detach()})

        # Loss
        if mode != "test":
            loss = self.criterion(pred, target)
            output["loss"] = logs[f"{mode}/loss"] = loss

        # Metrics
        output["metrics"] = {get_name(metric): metric(pred, target) for metric in self.metrics}
        logs.update({f"{mode}/metric_{k}": v for k, v in output["metrics"].items()})

        # Other (text, images, ...)

        # ...

        on_step = on_epoch = (None if mode == "test" else True)
        self.log_dict(logs, on_step=on_step, on_epoch=on_epoch)
        #self.logger.experiment.log_image("")
        return output

    def configure_optimizers(self):
        if self.optimizers is None:
            raise ValueError("Please specify 'optimizers'")
        if self.schedulers is None:
            return self.optimizers
        return self.optimizers, self.schedulers

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")

    def _get_dataloader(self, name):
        dataset = getattr(self, f"{name}_dataset")
        if dataset is None:
            raise ValueError(f"Please specify '{name}_dataset'")

        # A dataset can return None when a corrupted example occurs. This collate
        # function replace them with valid examples from the dataset.
        collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=dataset)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          collate_fn=collate_fn)
