"""EEG Regression Model for EEG 2025 Challenge.

This module implements the LighterModule for both challenges:
- Challenge 1: Cross-Task Transfer Learning (predict response time)
- Challenge 2: Externalizing Factor Prediction (predict psychopathology scores)

Both are regression tasks using EEG data from the HBN-EEG dataset.
"""

from typing import Any

from lighter import LighterModule


class EEGRegressionModel(LighterModule):
    """EEG Regression Model for the EEG 2025 Challenge.

    Handles EEG input tensors of shape [B, C, T] where:
    - B: batch size
    - C: number of EEG channels (129 for HBN-EEG)
    - T: number of time samples (200 for 2s at 100Hz)

    Supports both challenges:
    - Challenge 1: Predict response time from CCD task
    - Challenge 2: Predict externalizing factor scores

    Attributes:
        network: The neural network backbone (EEGNeX, EEGNet, etc.)
        criterion: Loss function (MSELoss or L1Loss for regression)
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler (optional)
        train_metrics: Training metrics (MAE, RMSE)
        val_metrics: Validation metrics
        test_metrics: Test metrics
    """

    def _shared_step(self, batch: tuple, metrics) -> dict[str, Any]:
        """Shared logic for train/val/test steps.

        Args:
            batch: Tuple of (EEG data, target values, crop_indices, metadata)
                   or (EEG data, target values) for simpler datasets
            metrics: MetricCollection or None

        Returns:
            Dictionary with loss, predictions, and targets
        """
        # Handle different batch formats from EEGDash
        if len(batch) == 4:
            # Full format: (X, y, crop_inds, infos)
            eeg, target = batch[0], batch[1]
        elif len(batch) == 3:
            # Format: (X, y, crop_inds)
            eeg, target = batch[0], batch[1]
        else:
            # Simple format: (X, y)
            eeg, target = batch

        # Ensure correct dtype
        eeg = eeg.float()
        target = target.float()

        # Ensure target has correct shape [B, 1]
        if target.dim() == 1:
            target = target.unsqueeze(1)

        # Forward pass
        pred = self(eeg)

        # Compute loss
        loss = self.criterion(pred, target)

        # Update metrics
        if metrics is not None:
            # Flatten for metrics
            metrics(pred.squeeze(), target.squeeze())

        return {
            "loss": loss,
            "pred": pred.squeeze(),
            "target": target.squeeze(),
        }

    def training_step(self, batch: tuple, batch_idx: int) -> dict[str, Any]:
        """Training step with loss computation and metric updates."""
        return self._shared_step(batch, self.train_metrics)

    def validation_step(self, batch: tuple, batch_idx: int) -> dict[str, Any]:
        """Validation step with loss and metrics."""
        return self._shared_step(batch, self.val_metrics)

    def test_step(self, batch: tuple, batch_idx: int) -> dict[str, Any]:
        """Test step with metrics only (no loss required)."""
        result = self._shared_step(batch, self.test_metrics)
        # Optionally remove loss for test
        return result

    def predict_step(self, batch: tuple, batch_idx: int) -> dict[str, Any]:
        """Prediction step for inference and submission.

        Returns predictions along with metadata for aggregation.
        """
        # Handle different batch formats
        if len(batch) == 4:
            eeg, target, crop_inds, infos = batch
        elif len(batch) == 3:
            eeg, target, crop_inds = batch
            infos = None
        else:
            eeg, target = batch if isinstance(batch, (tuple, list)) else (batch, None)
            infos = None

        eeg = eeg.float()

        # Forward pass
        pred = self(eeg)

        result = {
            "prediction": pred.squeeze().tolist() if pred.dim() > 0 else [pred.item()],
        }

        if target is not None:
            target = target.float()
            if target.dim() == 1:
                target = target.unsqueeze(1)
            result["target"] = target.squeeze().tolist()

        if infos is not None:
            result["subject"] = infos.get("subject", [])

        return result


class EEGWindowAggregator:
    """Aggregates predictions from multiple windows per subject.

    For the EEG 2025 Challenge, multiple windows are extracted per recording.
    This class aggregates window-level predictions to subject/recording level.

    Aggregation strategies:
    - mean: Average of all window predictions (default)
    - median: Median of all window predictions

    Args:
        strategy: Aggregation strategy ('mean', 'median')
    """

    def __init__(self, strategy: str = "mean") -> None:
        self.strategy = strategy
        self.predictions: dict[str, list[float]] = {}

    def add(self, subject: str, prediction: float) -> None:
        """Add a window prediction for a subject."""
        if subject not in self.predictions:
            self.predictions[subject] = []
        self.predictions[subject].append(prediction)

    def aggregate(self) -> dict[str, float]:
        """Aggregate all predictions by subject."""
        results = {}
        for subject, preds in self.predictions.items():
            if self.strategy == "mean":
                results[subject] = sum(preds) / len(preds)
            elif self.strategy == "median":
                sorted_preds = sorted(preds)
                mid = len(sorted_preds) // 2
                if len(sorted_preds) % 2 == 0:
                    results[subject] = (sorted_preds[mid - 1] + sorted_preds[mid]) / 2
                else:
                    results[subject] = sorted_preds[mid]
            else:
                # Default to mean
                results[subject] = sum(preds) / len(preds)
        return results

    def reset(self) -> None:
        """Clear all stored predictions."""
        self.predictions.clear()
