"""Regression on synthetic data whose answer is known: y = 2 * x0 - 3 * x1 + 0.5.

Train, validation and test samples are disjoint, and every sample has an ID that predictions keep.
"""

import math

import torch
from torch.utils.data import Dataset

from lighter import LighterModule

SPLITS = {"train": range(0, 64), "val": range(100, 116), "test": range(200, 216)}


class Samples(Dataset):
    def __init__(self, split: str):
        self.indices = SPLITS[split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        x = torch.tensor([math.sin(i), math.cos(1.3 * i)])
        return {"id": f"{i:05d}", "x": x, "y": 2 * x[0] - 3 * x[1] + 0.5}


class Regression(LighterModule):
    def training_step(self, batch, batch_idx):
        return self.criterion(self(batch["x"]).squeeze(-1), batch["y"])

    def validation_step(self, batch, batch_idx):
        prediction = self(batch["x"]).squeeze(-1)
        self.val_metrics(prediction, batch["y"])
        return self.criterion(prediction, batch["y"])

    def test_step(self, batch, batch_idx):
        prediction = self(batch["x"]).squeeze(-1)
        self.test_metrics(prediction, batch["y"])
        return self.criterion(prediction, batch["y"])

    def predict_step(self, batch, batch_idx):
        return {"id": batch["id"], "prediction": self(batch["x"]).squeeze(-1), "target": batch["y"]}
