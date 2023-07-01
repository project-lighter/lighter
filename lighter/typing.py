from dataclasses import dataclass
from typing import Callable, List, Optional, Union

from torch.utils.data import Dataset, Sampler
from torchmetrics import Metric, MetricCollection

from lighter.utils.misc import ensure_list

@dataclass
class Datasets:
    train: Optional[Dataset] = None
    val: Optional[Dataset] = None
    test: Optional[Dataset] = None
    predict: Optional[Dataset] = None 

@dataclass
class Samplers:
    train: Optional[Sampler] = None
    val: Optional[Sampler] = None
    test: Optional[Sampler] = None
    predict: Optional[Sampler] = None

@dataclass
class CollateFunctions:
    train: Optional[Callable] = None
    val: Optional[Callable] = None
    test: Optional[Callable] = None
    predict: Optional[Callable] = None

@dataclass
class Metrics:
    train: Optional[Union[Metric, List[Metric]]] = None
    val: Optional[Union[Metric, List[Metric]]] = None
    test: Optional[Union[Metric, List[Metric]]] = None

    def __post_init__(self):
        """Converts a list of metrics to MetricCollection after it has been assigned.""" 
        self.train = MetricCollection(ensure_list(self.train))
        self.val = MetricCollection(ensure_list(self.val))
        self.test = MetricCollection(ensure_list(self.test))
