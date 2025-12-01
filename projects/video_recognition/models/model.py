"""Video action recognition model using LighterModule."""

import torch

from lighter import LighterModule


class VideoClassificationModel(LighterModule):
    """Video action recognition model.

    Showcases:
    - Handling video inputs (5D tensors: B, C, T, H, W)
    - Top-k accuracy metrics
    - Predictions for CsvWriter callback
    """

    def _shared_step(self, batch: tuple, metrics) -> dict:
        """Shared logic for train/val/test steps."""
        video, label = batch

        # Forward pass - network handles [B, C, T, H, W] input
        logits = self(video)

        # Compute loss
        loss = self.criterion(logits, label)

        # Get predictions
        pred = logits.argmax(dim=1)

        # Update metrics
        if metrics is not None:
            metrics(logits, label)

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
        """Prediction step returning class labels, probabilities, and video tensors.

        Returns dict compatible with CsvWriter and FileWriter callbacks.
        - CsvWriter: uses prediction, confidence, ground_truth
        - FileWriter: uses video tensor for mp4 output
        """
        video, label = batch if isinstance(batch, (tuple, list)) else (batch, None)

        logits = self(video)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)

        # Get top-5 predictions
        top5_probs, top5_indices = probs.topk(5, dim=1)

        result = {
            "prediction": pred.tolist(),
            "confidence": probs.max(dim=1).values.tolist(),
            "top5_classes": top5_indices.tolist(),
            "top5_probs": top5_probs.tolist(),
            "video": video,  # Include video tensor for FileWriter
        }

        if label is not None:
            result["ground_truth"] = label.tolist()

        return result
