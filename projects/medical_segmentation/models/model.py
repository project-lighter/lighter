"""Medical segmentation model using MONAI and LighterModule."""

import torch

from lighter import LighterModule


class SegmentationModel(LighterModule):
    """3D Medical image segmentation model using MONAI.

    Uses MONAI's DiceLoss and sliding window inference for volumetric data.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
        sw_batch_size: int = 4,
        roi_size: tuple[int, int, int] = (96, 96, 96),
        **kwargs,
    ) -> None:
        super().__init__(
            network=network,
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs,
        )
        self.sw_batch_size = sw_batch_size
        self.roi_size = roi_size

        # Use MONAI's DiceLoss
        from monai.losses import DiceLoss

        self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def training_step(self, batch, batch_idx):
        """Training step with Dice loss."""
        images = batch["image"]
        labels = batch["label"]

        # Forward pass
        outputs = self(images)

        # Compute Dice loss
        loss = self.dice_loss(outputs, labels)

        # Get predictions for metrics
        preds = outputs.argmax(dim=1)

        if self.train_metrics is not None:
            self.train_metrics(preds, labels.squeeze(1).long())

        self.log("train/loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Validation step with sliding window inference."""
        from monai.inferers import sliding_window_inference

        images = batch["image"]
        labels = batch["label"]

        # Use sliding window inference for full volumes
        outputs = sliding_window_inference(
            images,
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.network,
        )

        loss = self.dice_loss(outputs, labels)
        preds = outputs.argmax(dim=1)

        if self.val_metrics is not None:
            self.val_metrics(preds, labels.squeeze(1).long())

        self.log("val/loss", loss, prog_bar=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        """Test step with sliding window inference."""
        from monai.inferers import sliding_window_inference

        images = batch["image"]
        labels = batch["label"]

        outputs = sliding_window_inference(
            images,
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.network,
        )

        preds = outputs.argmax(dim=1)

        if self.test_metrics is not None:
            self.test_metrics(preds, labels.squeeze(1).long())

        return {"pred": preds, "label": labels}

    def predict_step(self, batch, batch_idx):
        """Prediction step returning segmentation masks."""
        from monai.inferers import sliding_window_inference

        images = batch["image"]

        outputs = sliding_window_inference(
            images,
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.network,
        )

        preds = outputs.argmax(dim=1)
        return {"pred": preds, "output": outputs}
