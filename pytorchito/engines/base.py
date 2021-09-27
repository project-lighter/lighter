import copy
from abc import ABC, abstractmethod
from pathlib import Path
from loguru import logger

import torch
import torchvision

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pytorchito.utils import sliding_window_inferer, communication
from pytorchito.utils.io import instantiate, instantiate_dict_list_union
from catalyst.data.sampler import DistributedSamplerWrapper


class BaseEngine(ABC):

    def __init__(self, conf):
        # deep copy to isolate the conf.mode of an engine from other engines (e.g train from val)
        self.conf = copy.deepcopy(conf)
        self._get_mode()

        # self.output_dir = Path(conf[conf.mode].output_dir) / self.conf.mode

        self.device = self._get_device()
        self.model = self._get_model()

        self.logger = logger

    @abstractmethod
    def _get_mode(self):
        """Sets the mode for the particular engine.
        E.g., 'train' for Trainer, 'val' for 'Validator' etc."""
        self.conf._mode = ...

    def _get_device(self):
        if torch.distributed.is_initialized():
            local_rank = communication.get_local_rank()
            return torch.device(f"cuda:{local_rank}")  # distributed GPU training
        elif self.conf[self.conf._mode].cuda:
            return torch.device('cuda:0')  # single GPU training
        else:
            return torch.device('cpu')

    def _get_model(self):
        return instantiate(self.conf[self.conf._mode].model).to(self.device)

    def _get_dataloader(self):
        """
        """
        shuffle = self.conf._mode == "train"

        # Dataset
        dataset = instantiate(self.conf[self.conf._mode].dataset)

        # Transforms
        for name in ["transform", "target_transform"]:
            if transform := getattr(self.conf[self.conf._mode], name, False):
                transform = instantiate_dict_list_union(transform)
                transform = torchvision.transforms.Compose(transform)
                if hasattr(dataset, name):
                    setattr(dataset, name, transform)
                else:
                    raise ValueError("asdafda")
                    logger.exception(f"`{name}` specified in the config but the dataset "
                                     f"`{type(dataset).__name__}`` does not have that option")

        # Sampler
        sampler = None
        if sampler := getattr(self.conf[self.conf._mode], "sampler", False):
            sampler = instantiate(sampler)
        if hasattr(sampler, "shuffle"):
            sampler.shuffle = shuffle

        # Distributed sampling
        if torch.distributed.is_initialized():
            # Prevent DDP from running on a dataset smaller than the total batch size over processes
            ddp_batch_size = self.conf[self.conf._mode].batch_size
            ddp_batch_size *= communication.get_world_size()
            if ddp_batch_size > len(dataset):
                logger.exception(f"Dataset has {len(dataset)} examples, while the effective "
                                f"batch size equals to {ddp_batch_size}. Distributed mode does "
                                f"not work as expected in this situation.")

            if sampler is not None:
                sampler = DistributedSamplerWrapper(sampler,
                                                    shuffle=shuffle,
                                                    num_replicas=communication.get_world_size(),
                                                    rank=communication.get_rank())
            else:
                sampler = DistributedSampler(dataset,
                                            shuffle=shuffle,
                                            num_replicas=communication.get_world_size(),
                                            rank=communication.get_rank())

        # When using sampler, shuffle must not be specified in DataLoader
        shuffle = shuffle if sampler is None else None
        return DataLoader(dataset,
                          sampler=sampler,
                          shuffle=shuffle,
                          batch_size=self.conf[self.conf._mode].batch_size,
                          num_workers=self.conf[self.conf._mode].num_workers,
                          pin_memory=self.conf[self.conf._mode].pin_memory)
