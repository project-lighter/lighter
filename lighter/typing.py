from dataclasses import dataclass
from typing import Callable, List, Optional, Union

from torch.utils.data import Dataset, Sampler
from torchmetrics import Metric, MetricCollection

from lighter.utils.misc import ensure_list

# Define a dataclass Datasets that will hold the datasets for train, val, test, and predict. Defaults to None.
@dataclass
class Datasets:
    train: Optional[Union[Dataset, List[Dataset]]] = None
    val: Optional[Union[Dataset, List[Dataset]]] = None
    test: Optional[Union[Dataset, List[Dataset]]] = None
    predict: Optional[Union[Dataset, List[Dataset]]] = None 

# Define a dataclass Samplers that will hold the samplers for train, val, test, and predict. Defaults to None.
@dataclass
class Samplers:
    train: Optional[Sampler] = None
    val: Optional[Sampler] = None
    test: Optional[Sampler] = None
    predict: Optional[Sampler] = None

# Define a dataclass Collate that will hold the collate functions for train, val, test, and predict. Defaults to None.
@dataclass
class CollateFunctions:
    train: Optional[Callable] = None
    val: Optional[Callable] = None
    test: Optional[Callable] = None
    predict: Optional[Callable] = None

# Define a dataclass Metrics that will hold the metrics for train, val, test,. Defaults to None.
@dataclass
class Metrics:
    train: Optional[Union[Metric, List[Metric]]] = None
    val: Optional[Union[Metric, List[Metric]]] = None
    test: Optional[Union[Metric, List[Metric]]] = None

    def _post_init(self):
        self.train = MetricCollection(ensure_list(self.train))
        self.val = MetricCollection(ensure_list(self.val))
        self.test = MetricCollection(ensure_list(self.test))
