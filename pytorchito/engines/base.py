import copy
from abc import ABC, abstractmethod
from loguru import logger

import torch
import torchvision

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from catalyst.data.sampler import DistributedSamplerWrapper

from pytorchito.utils import communication
from pytorchito.utils.importing import instantiate, instantiate_dict_list_union


class BaseEngine(ABC):

    def __init__(self, conf):
        # deep copy to isolate the conf.mode of an engine from other engines (e.g train from val)
        self.conf = copy.deepcopy(conf)

        self._set_mode()
        self.device = self._get_device()

        self.model = self._init_model()
        self.dataloader = self._init_dataloader()
        self.metrics = self._init_metrics()

        self.logger = logger

    @abstractmethod
    def _set_mode(self):
        """Sets the mode for the particular engine.
        E.g., 'train' for Trainer, 'val' for 'Validator' etc."""
        self.conf["_mode"] = ...

    def _get_device(self):
        if torch.distributed.is_initialized():
            local_rank = communication.get_local_rank()
            return torch.device(f"cuda:{local_rank}")  # distributed GPU mode
        elif self.conf[self.conf["_mode"]].cuda:
            return torch.device('cuda:0')  # single GPU mode
        return torch.device('cpu')

    def _init_model(self):
        return instantiate(self.conf[self.conf["_mode"]].model).to(self.device)

    def _init_metrics(self):
        if self.conf[self.conf["_mode"]].metrics:
            return instantiate_dict_list_union(self.conf[self.conf["_mode"]].metrics, to_dict=True)
        return {}

    def _init_dataloader(self):
        shuffle = self.conf["_mode"] == "train"

        # Dataset
        dataset = instantiate(self.conf[self.conf["_mode"]].dataset)

        # Transforms
        for name in ["transform", "target_transform"]:
            if transform := getattr(self.conf[self.conf["_mode"]], name, False):
                transform = instantiate_dict_list_union(transform)
                transform = torchvision.transforms.Compose(transform)
                if hasattr(dataset, name):
                    setattr(dataset, name, transform)
                else:
                    raise ValueError(f"`{name}` specified in the config but the dataset "
                                     f"`{type(dataset).__name__}`` does not have that option")

        # Sampler
        sampler = None
        if sampler := getattr(self.conf[self.conf["_mode"]], "sampler", False):
            sampler = instantiate(sampler)
        if hasattr(sampler, "shuffle"):
            sampler.shuffle = shuffle

        # Distributed sampling
        if torch.distributed.is_initialized():
            # Prevent DDP from running on a dataset smaller than the total batch size over processes
            ddp_batch_size = self.conf[self.conf["_mode"]].batch_size
            ddp_batch_size *= communication.get_world_size()
            if ddp_batch_size > len(dataset):
                raise ValueError(f"Dataset has {len(dataset)} examples, while the effective "
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
                          batch_size=self.conf[self.conf["_mode"]].batch_size,
                          num_workers=self.conf[self.conf["_mode"]].num_workers,
                          pin_memory=self.conf[self.conf["_mode"]].pin_memory)
