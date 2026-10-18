import pytorch_lightning as pl
import torch


class Regressor(pl.LightningModule):
    def __init__(self, hidden: int = 32, lr: float = 0.01):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(8, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.net(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
