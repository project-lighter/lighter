"""Efficient fine-tuning model using LighterModule."""

import torch

from lighter import LighterModule


class LoRAClassificationModel(LighterModule):
    """Classification model with LoRA fine-tuning support.

    Showcases:
    - Parameter-efficient fine-tuning
    - Tracking trainable parameter percentage
    - Standard classification workflow
    """

    def _shared_step(self, batch: tuple, metrics) -> dict:
        """Shared logic for train/val/test steps."""
        image, label = batch

        # Forward pass
        logits = self(image)

        # Compute loss
        loss = self.criterion(logits, label)

        # Get predictions
        pred = logits.argmax(dim=1)

        # Update metrics
        if metrics is not None:
            metrics(pred, label)

        return {
            "loss": loss,
            "pred": pred,
            "label": label,
            "logits": logits,
        }

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, self.train_metrics)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, self.val_metrics)

    def test_step(self, batch, batch_idx):
        result = self._shared_step(batch, self.test_metrics)
        del result["loss"]
        return result

    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        image, label = batch if isinstance(batch, (tuple, list)) else (batch, None)

        logits = self(image)
        pred = logits.argmax(dim=1)
        confidence = torch.softmax(logits, dim=1).max(dim=1).values

        result = {
            "prediction": pred.tolist(),
            "confidence": confidence.tolist(),
        }

        if label is not None:
            result["ground_truth"] = label.tolist()

        return result

    def on_train_start(self) -> None:
        """Log trainable parameters at training start."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        pct = 100 * trainable / total if total > 0 else 0

        self.log("trainable_params", float(trainable))
        self.log("total_params", float(total))
        self.log("trainable_pct", pct)

        print(f"\nTrainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")
