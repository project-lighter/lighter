import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import ListConfig
# Get the name of an object, class or function
get_name = lambda x: type(x).__name__ if isinstance(x, object) else x.__name__
# Wrap into a list if it is not a list or None
wrap_into_list = lambda x: x if isinstance(x, (list, ListConfig)) or x is None else [x]


class System(pl.LightningModule):

    def __init__(self,
                 model,
                 batch_size,
                 num_workers,
                 pin_memory,
                 criterion=None,
                 optimizers=None,
                 schedulers=None,
                 metrics=None,
                 train_dataset=None,
                 val_dataset=None,
                 test_dataset=None):

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
        self.metrics = wrap_into_list(metrics)

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

        x, y = batch
        y_hat = self(x)

        # Loss
        if mode != "test":
            loss = self.criterion(y_hat, y)
            output["loss"] = loss
            logs[f"{mode}/loss"] = loss

        # Metrics
        output["metrics"] = {get_name(metric): metric(y_hat, y) for metric in self.metrics}
        logs.update({f"{mode}/metric/{k}": v for k, v in output["metrics"].items()})
        
        # Other (text, images, ...
        on_step = on_epoch = None if mode == "test" else True
        self.log_dict(logs, on_step=on_step, on_epoch=on_epoch)
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
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
