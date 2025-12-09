"""CIFAR10 Model implementation using LighterModule class."""

from lighter import LighterModule


class CIFAR10Model(LighterModule):
    """
    Simple classification model for CIFAR10.

    Users have full control over step logic while framework handles automatic logging.
    """

    def training_step(self, batch, batch_idx):
        """Training step with user-defined logic."""
        # Extract batch data
        x, y = batch

        # Forward pass
        pred = self(x)

        # Compute loss using criterion from config
        if self.criterion is None:
            raise RuntimeError("criterion is required for training but was not set in config")
        loss = self.criterion(pred, y)

        # Update metrics (user calls them explicitly)
        if self.train_metrics is not None:
            self.train_metrics(pred, y)

        # Return dict - framework logs automatically
        return {"loss": loss, "pred": pred, "target": y}

    def validation_step(self, batch, batch_idx):
        """Validation step with user-defined logic."""
        x, y = batch
        pred = self(x)
        if self.criterion is None:
            raise RuntimeError("criterion is required for validation but was not set in config")
        loss = self.criterion(pred, y)

        if self.val_metrics is not None:
            self.val_metrics(pred, y)

        return {"loss": loss, "pred": pred, "target": y}

    def test_step(self, batch, batch_idx):
        """Test step with user-defined logic."""
        x, y = batch
        pred = self(x)

        if self.test_metrics is not None:
            self.test_metrics(pred, y)

        # No loss required in test mode
        return {"pred": pred, "target": y}

    def predict_step(self, batch, batch_idx):
        """Prediction step - return dict for FileWriter compatibility."""
        x, y = batch
        pred = self(x)
        return {"pred": pred}
