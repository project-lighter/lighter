"""Small disjoint synthetic populations with stable sample identities."""

import math

import torch
from torch.utils.data import Dataset

from lighter import LighterModule

PREDICTION_IDS = ("00001", "NA", "NULL", "case,4", 'case"5', "λ-6", "line\n7")
SPLITS = {"train": (0, 24), "val": (40, 9), "test": (80, len(PREDICTION_IDS))}


class RegressionSamples(Dataset):
    def __init__(self, split: str):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; choose one of {tuple(SPLITS)}")
        self.split = split
        self.start, self.count = SPLITS[split]

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        sample = self.start + index
        x = torch.tensor([math.sin(sample), math.cos(sample * 1.3)], dtype=torch.float32)
        y = 2 * x[0] - 3 * x[1] + 0.5
        identifier = PREDICTION_IDS[index] if self.split == "test" else f"{self.split}-{index:04d}"
        return {"x": x, "target": y, "id": identifier}


class RegressionTask(LighterModule):
    def training_step(self, batch, batch_idx):
        prediction = self(batch["x"]).squeeze(-1)
        return self.criterion(prediction, batch["target"])

    def validation_step(self, batch, batch_idx):
        prediction = self(batch["x"]).squeeze(-1)
        return self.criterion(prediction, batch["target"])

    def test_step(self, batch, batch_idx):
        prediction = self(batch["x"]).squeeze(-1)
        return self.criterion(prediction, batch["target"])

    def predict_step(self, batch, batch_idx):
        return {"id": batch["id"], "prediction": self(batch["x"]).squeeze(-1), "target": batch["target"]}
