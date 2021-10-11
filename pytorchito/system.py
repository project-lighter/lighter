import pytorch_lightning as pl
from torch.utils.data import DataLoader

class System(pl.LightningModule):
    def __init__(self,
                 model,
                 batch_size,
                 num_workers,
                 pin_memory,
                 criteria=None,
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
        self.criteria = criteria
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.metrics = metrics
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Losses
        #losses = {name: criterion(y_hat, y) for name, criterion in self.criteria.items()}
        losses = [criterion(y_hat, y) for criterion in self.criteria]
        losses = sum(losses) # temporary
        #losses["loss"] = sum(losses.values())

        # Metrics
        #metrics = {}

        #self.log_dict(losses | metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss", losses)
        return losses# | metrics

    # def training_step_end(self):

    # def training_epoch_end(self):

    def validation_step(self, batch, batch_idx):
        pass

    # def validation_step_end(self):

    # def validation_epoch_end(self):

    def test_step(self, batch, batch_idx):
        pass

    # def test_step_end(self):

    # def test_epoch_end(self):

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