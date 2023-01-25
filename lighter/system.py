import sys
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from loguru import logger
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Sampler
from torchmetrics import Metric, MetricCollection

from lighter.utils.collate import collate_fn_replace_corrupted
from lighter.utils.misc import ensure_list, get_name, hasarg
from lighter.utils.model import reshape_pred_if_single_value_prediction


class LighterSystem(pl.LightningModule):

    def __init__(self,
                 model: Module,
                 batch_size: int,
                 drop_last_batch: bool = False,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 optimizers: Optional[Union[Optimizer, List[Optimizer]]] = None,
                 schedulers: Optional[Union[Callable, List[Callable]]] = None,
                 criterion: Optional[Callable] = None,
                 cast_target_dtype_to: Optional[str] = None,
                 post_criterion_activation: Optional[str] = None,
                 patch_based_inferer: Optional[Callable] = None,
                 train_metrics: Optional[Union[Metric, List[Metric]]] = None,
                 val_metrics: Optional[Union[Metric, List[Metric]]] = None,
                 test_metrics: Optional[Union[Metric, List[Metric]]] = None,
                 train_dataset: Optional[Union[Dataset, List[Dataset]]] = None,
                 val_dataset: Optional[Union[Dataset, List[Dataset]]] = None,
                 test_dataset: Optional[Union[Dataset, List[Dataset]]] = None,
                 predict_dataset: Optional[Union[Dataset, List[Dataset]]] = None,
                 train_sampler: Optional[Sampler] = None,
                 val_sampler: Optional[Sampler] = None,
                 test_sampler: Optional[Sampler] = None,
                 predict_sampler: Optional[Sampler] = None) -> None:
        """_summary_

        Args:
            model (Module): the model.
            batch_size (int): batch size.
            drop_last_batch (bool, optional): whether the last batch in the dataloader
                should be dropped. Defaults to False.
            num_workers (int, optional): number of dataloader workers. Defaults to 0.
            pin_memory (bool, optional): whether to pin the dataloaders memory. Defaults to True.
            optimizers (Optional[Union[Optimizer, List[Optimizer]]], optional):
                a single or a list of optimizers. Defaults to None.
            schedulers (Optional[Union[Callable, List[Callable]]], optional):
                a single or a list of schedulers. Defaults to None.
            criterion (Optional[Callable], optional):
                criterion/loss function. Defaults to None.
            cast_target_dtype_to (Optional[str], optional): whether to cast the target to the
                specified type before calculating the loss. May be necessary for some criterions.
                Defaults to None.
            post_criterion_activation (Optional[str], optional): some criterions
                (e.g. BCEWithLogitsLoss) require non-activated prediction for their calculaiton.
                However, to calculate the metrics and log the data, it may be necessary to activate
                the predictions. Defaults to None.
            patch_based_inferer (Optional[Callable], optional): the patch based inferer needs to be
                either a class with a `__call__` method or function that accepts two arguments -
                first one is the input tensor, and the other one the model itself. It should
                perform the inference over the patches and return the aggregated/averaged output.
                Defaults to None.
            train_metrics (Optional[Union[Metric, List[Metric]]], optional): training metric(s).
                They have to be implemented using `torchmetrics`. Defaults to None.
            val_metrics (Optional[Union[Metric, List[Metric]]], optional): validation metric(s).
                They have to be implemented using `torchmetrics`. Defaults to None.
            test_metrics (Optional[Union[Metric, List[Metric]]], optional): test metric(s).
                They have to be implemented using `torchmetrics`. Defaults to None.
            train_dataset (Optional[Union[Dataset, List[Dataset]]], optional): training dataset(s).
                Defaults to None.
            val_dataset (Optional[Union[Dataset, List[Dataset]]], optional): validation dataset(s).
                Defaults to None.
            test_dataset (Optional[Union[Dataset, List[Dataset]]], optional): test dataset(s).
                Defaults to None.
            predict_dataset (Optional[Union[Dataset, List[Dataset]]], optional): predict dataset(s).
                Defaults to None.
            train_sampler (Optional[Sampler], optional): training sampler(s). Defaults to None.
            val_sampler (Optional[Sampler], optional): validation sampler(s). Defaults to None.
            test_sampler (Optional[Sampler], optional):  test sampler(s). Defaults to None.
            predict_sampler (Optional[Sampler], optional):  predict sampler(s). Defaults to None.
        """
        super().__init__()
        # Bypass LightningModule's check for default methods. We define them in self.setup().
        self._init_placeholders_for_dataloader_and_step_methods()

        # Model setup
        self.model = model
        self.batch_size = batch_size
        self.drop_last_batch = drop_last_batch

        # Criterion, optimizer, and scheduler
        self.criterion = criterion
        self.optimizers = ensure_list(optimizers)
        self.schedulers = ensure_list(schedulers)

        # Datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Samplers
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler
        self.predict_sampler = predict_sampler

        # Metrics
        self.train_metrics = MetricCollection(ensure_list(train_metrics))
        self.val_metrics = MetricCollection(ensure_list(val_metrics))
        self.test_metrics = MetricCollection(ensure_list(test_metrics))

        # Criterion-specific activation function and data type casting
        self._post_criterion_activation = post_criterion_activation
        self._cast_target_dtype_to = cast_target_dtype_to

        # Patch-based inference
        self._patch_based_inferer = patch_based_inferer

        # Checks
        self._lightning_module_methods_defined = False
        self._target_not_used_reported = False

    def forward(self, input: Union[torch.Tensor, List, Tuple]) -> Union[torch.Tensor, List, Tuple]:
        """Forward pass. Multi-input models are supported.

        Args:
            input (Union[torch.Tensor, List, Tuple]): input to the model.

        Returns:
            Union[torch.Tensor, List, Tuple]: output of the model.
        """
        kwargs = {}
        if hasarg(self.model.forward, "epoch"):
            kwargs["epoch"] = self.current_epoch
        if hasarg(self.model.forward, "step"):
            kwargs["step"] = self.global_step

        if isinstance(input, list):
            return self.model(*input, **kwargs)
        return self.model(input, **kwargs)

    def _base_step(self, batch: Tuple, batch_idx: int, mode: str) -> Union[Dict[str, Any], Any]:
        """Base step for all modes ('train', 'val', 'test', 'predict')

        Args:
            batch (Tuple):
                output of the DataLoader and input to the model.
            batch_idx (int): index of the batch. Not used, but PyTorch Lightning requires it.
            mode (str): mode in which the system is.

        Returns:
            Union[Dict[str, Any], Any]: For the training, validation and test step, it returns
                a dict containing loss, metrics, input, target, and pred. Loss will be `None`
                for the test step. Metrics will be `None` if no metrics are specified.
                
                For predict step, it returns pred only.
        """
        input, target = batch if len(batch) == 2 else (batch[:-1], batch[-1])

        # Predict
        if self._patch_based_inferer and mode in ["val", "test", "predict"]:
            # TODO: Patch-based inference doesn't support multiple inputs yet
            pred = self._patch_based_inferer(input, self)
        else:
            pred = self(input)

        pred = reshape_pred_if_single_value_prediction(pred, target)

        # Calculate the loss
        loss = None
        if mode in ["train", "val"]:
            loss = self._calculate_loss(pred, target)

        # Apply the post-criterion activation. Necessary for measuring the metrics
        # correctly in cases when using a criterion such as `BCELossWithLogits`` which
        # requires the model to output logits, i.e. non-activated outputs.
        if self._post_criterion_activation is not None:
            pred = self._post_criterion_activation(pred)

        # In predict mode, skip metrics and logging parts and return the predicted value
        if mode == "predict":
            return pred

        # Calculate the metrics for the step
        step_metrics = getattr(self, f"{mode}_metrics")(pred, target)

        return {
            "loss": loss,
            "metrics": step_metrics,
            "input": input,
            "target": target,
            "pred": pred
        }

    def _calculate_loss(self, pred: Union[torch.Tensor, List, Tuple],
                        target: Union[torch.Tensor, None]) -> torch.Tensor:
        """_summary_

        Args:
            pred (Union[torch.Tensor, List, Tuple]): the predicted values from the model.
            target (Union[torch.Tensor, None]): the target/label.

        Returns:
            torch.Tensor: the calculated loss.
        """
        if hasarg(self.criterion.forward, "target"):
            loss = self.criterion(pred, target.to(self._cast_target_dtype_to))
        else:
            loss = self.criterion(*pred if isinstance(pred, (list, tuple)) else pred)

            if not self._target_not_used_reported and not self.trainer.sanity_checking:
                self._target_not_used_reported = True
                logger.info(f"The criterion `{get_name(self.criterion, True)}` "
                            "has no `target` argument. In such cases, the LighterSystem "
                            "passes only the predicted values to the criterion. "
                            "This is intended as a support for self-supervised "
                            "losses where target is not used. If this is not the "
                            "behavior you expected, redefine your criterion "
                            "so that it has a `target` argument.")
        return loss

    def _base_dataloader(self, mode: str) -> DataLoader:
        """Instantiate the dataloader for a mode (train/val/test).
        Includes a collate function that enables the DataLoader to replace
        None's (alias for corrupted examples) in the batch with valid examples.
        To make use of it, write a try-except in your Dataset that handles
        corrupted data by returning None instead.

        Args:
            mode (str): mode for which to create the dataloader ['train', 'val', 'test'].

        Returns:
            DataLoader: instantiated DataLoader.
        """
        dataset = getattr(self, f"{mode}_dataset")
        sampler = getattr(self, f"{mode}_sampler")

        if dataset is None:
            logger.error(f"Please specify '{mode}_dataset' in the config. Exiting")
            sys.exit()

        # Batch size is 1 when using patch based inference for two reasons:
        # 1) Patch based inference splits an input into a batch of patches,
        # so the batch size will actually be defined for it;
        # 2) In settings where patch based inference is needed, the input data often
        # varies in shape, preventing the data loader to stack them into a batch.
        batch_size = self.batch_size
        if self._patch_based_inferer is not None and mode in ["val", "test", "predict"]:
            logger.info(f"Setting the general batch size to 1 for {mode} "
                        "mode because a patch-based inferer is used.")
            batch_size = 1

        # A dataset can return None when a corrupted example occurs. This collate
        # function replaces None's with valid examples from the dataset.
        collate_fn = partial(collate_fn_replace_corrupted, dataset=dataset)
        return DataLoader(dataset,
                          sampler=sampler,
                          shuffle=(mode == "train" and sampler is None),
                          batch_size=batch_size,
                          drop_last=self.drop_last_batch,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          collate_fn=collate_fn)

    def configure_optimizers(self) -> Dict:
        """LightningModule method. Returns optimizers and, if defined, schedulers.

        Returns:
            Dict: a dict of optimizers and schedulers.
        """
        if not self.optimizers:
            logger.error("Please specify 'optimizers' in the config. Exiting.")
            sys.exit()
        if not self.schedulers:
            return self.optimizers
        return {"optimizer": self.optimizers, "lr_scheduler": self.schedulers}

    def setup(self, stage: str) -> None:
        """Automatically called by the LightningModule after the initialization.
        `LighterSystem`'s setup checks if the required dataset is provided in the config and
        sets up LightningModule methods for the stage in which the system is.

        Args:
            stage (str): passed by PyTorch Lightning. ['fit', 'validate', 'test'].
        """
        # Stage-specific PyTorch Lightning methods. Defined dynamically so that the system
        # only has methods used in the stage and for which the configuration was provided.
        if not self._lightning_module_methods_defined:
            del (self.train_dataloader, self.training_step, self.val_dataloader,
                 self.validation_step, self.test_dataloader, self.test_step,
                 self.predict_dataloader, self.predict_step)
            # `Trainer.tune()` calls the `self.setup()` method whenever it runs for a new
            #  parameter, and deleting the above methods again breaks it. This flag prevents it.
            self._lightning_module_methods_defined = True

        # Training methods.
        if stage in ["fit", "tune"]:
            self.train_dataloader = partial(self._base_dataloader, mode="train")
            self.training_step = partial(self._base_step, mode="train")

        # Validation methods. Required in 'validate' stage and optionally in 'fit' or 'tune' stage.
        if stage == "validate" or (stage in ["fit", "tune"] and self.val_dataset is not None):
            self.val_dataloader = partial(self._base_dataloader, mode="val")
            self.validation_step = partial(self._base_step, mode="val")

        # Test methods.
        if stage == "test":
            self.test_dataloader = partial(self._base_dataloader, mode="test")
            self.test_step = partial(self._base_step, mode="test")

        # Predict methods.
        if stage == "predict":
            self.predict_dataloader = partial(self._base_dataloader, mode="predict")
            self.predict_step = partial(self._base_step, mode="predict")

    def _init_placeholders_for_dataloader_and_step_methods(self) -> None:
        """`LighterSystem` dynamically defines the `..._dataloader()`and `..._step()` methods
        in the `self.setup()` method. However, `LightningModule` excepts them to be defined at
        the initialization. To prevent it from throwing an error, the `..._dataloader()` and
        `..._step()` are initially defined as `lambda: None`, before `self.setup()` is called.
        """
        self.train_dataloader = self.training_step = lambda: None
        self.val_dataloader = self.validation_step = lambda: None
        self.test_dataloader = self.test_step = lambda: None
        self.predict_dataloader = self.predict_step = lambda: None
