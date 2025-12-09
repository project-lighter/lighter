"""Text classification model using LighterModule."""

from lighter import LighterModule


class TextClassificationModel(LighterModule):
    """
    HuggingFace text classification model wrapper.

    The HuggingFace model computes its own loss when labels are provided,
    so we don't need a separate criterion.
    """

    def _shared_step(self, batch, metrics):
        """Shared step logic for train/val/test."""
        # HuggingFace models expect named arguments
        outputs = self.network(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        # Update metrics if available
        if metrics is not None:
            preds = outputs.logits.argmax(dim=-1)
            metrics(preds, batch["labels"])

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self._shared_step(batch, self.train_metrics)
        return {"loss": outputs.loss}

    def validation_step(self, batch, batch_idx):
        outputs = self._shared_step(batch, self.val_metrics)
        return {"loss": outputs.loss}

    def test_step(self, batch, batch_idx):
        outputs = self._shared_step(batch, self.test_metrics)
        return {"loss": outputs.loss}

    def predict_step(self, batch, batch_idx):
        """Prediction step - return predictions for CsvWriter."""
        outputs = self.network(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        preds = outputs.logits.argmax(dim=-1)
        return {"prediction": preds.tolist()}
