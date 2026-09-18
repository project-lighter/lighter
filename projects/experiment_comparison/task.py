"""The scientific model, objective and complete-population measurements."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from lighter import LighterModule

PREDICTION_COLUMNS = ["id", "label", *[f"logit_{index}" for index in range(10)]]


class Classifier(nn.Sequential):
    """An ordinary eager network; optional tensors align qualification runs."""

    def __init__(self, initial_state=None):
        super().__init__(
            nn.Conv2d(3, 16, 3, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10, bias=True),
        )
        # Construction precedes native checkpoint restoration. No later hook
        # reloads these tensors, so checkpoint model state always takes priority.
        if initial_state is not None:
            self.load_state_dict(torch.load(Path(initial_state), map_location="cpu", weights_only=True), strict=True)


class EvaluationTotals:
    """Sum per-example observations, including a partial final batch."""

    def __init__(self):
        self.loss_sum = 0.0
        self.count = 0
        self.correct = 0
        self.confusion = torch.zeros((10, 10), dtype=torch.int64)

    def update(self, logits, labels):
        if logits.ndim != 2 or logits.shape != (len(labels), 10) or labels.dtype != torch.int64:
            raise ValueError("Expected ten logits and one int64 label per example")
        if not torch.isfinite(logits).all() or not ((labels >= 0) & (labels < 10)).all():
            raise ValueError("Logits must be finite and labels must be in [0, 9]")
        losses = F.cross_entropy(logits, labels, reduction="none")
        predicted = logits.argmax(dim=1)
        self.loss_sum += losses.detach().double().sum().item()
        self.count += len(labels)
        self.correct += (predicted == labels).sum().item()
        self.confusion += torch.bincount((labels * 10 + predicted).cpu(), minlength=100).reshape(10, 10)

    def result(self):
        if self.count == 0:
            raise ValueError("Cannot summarize an empty population")
        return {
            "count": self.count,
            "loss_sum": self.loss_sum,
            "loss": self.loss_sum / self.count,
            "correct": self.correct,
            "accuracy": self.correct / self.count,
            "confusion": self.confusion.tolist(),
            "predicted_classes": (self.confusion.sum(dim=0) > 0).nonzero().flatten().tolist(),
        }


class ScientificSteps:
    """Shared science for the two entrypoints; independent checks live elsewhere."""

    def training_step(self, batch, batch_idx):
        return F.cross_entropy(self.network(batch["image"]), batch["label"], reduction="mean")

    def on_validation_epoch_start(self):
        self.validation_totals = EvaluationTotals()

    def validation_step(self, batch, batch_idx):
        self.validation_totals.update(self.network(batch["image"]), batch["label"])

    def on_validation_epoch_end(self):
        self.validation_result = self.validation_totals.result()
        self.log(
            "val/ce",
            torch.tensor(self.validation_result["loss"], dtype=torch.float64, device=self.device),
            on_step=False,
            on_epoch=True,
        )
        self.log("val/accuracy", self.validation_result["accuracy"], on_step=False, on_epoch=True)

    def on_test_epoch_start(self):
        self.test_totals = EvaluationTotals()

    def test_step(self, batch, batch_idx):
        self.test_totals.update(self.network(batch["image"]), batch["label"])

    def on_test_epoch_end(self):
        self.test_result = self.test_totals.result()
        self.log("test/ce", self.test_result["loss"], on_step=False, on_epoch=True)
        self.log("test/accuracy", self.test_result["accuracy"], on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        logits = self.network(batch["image"])
        return {"id": batch["id"], "label": batch["label"], **{f"logit_{i}": logits[:, i] for i in range(10)}}


class ClassificationTask(ScientificSteps, LighterModule):
    """Lighter inherits optimizer setup; the step remains ordinary Python."""
