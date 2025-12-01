"""Vision-Language CLIP model using LighterModule."""

import torch
import torch.nn.functional as F

from lighter import LighterModule


def clip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Compute symmetric CLIP contrastive loss.

    Args:
        image_features: L2-normalized image embeddings [B, D].
        text_features: L2-normalized text embeddings [B, D].
        temperature: Temperature parameter.

    Returns:
        Scalar loss value.
    """
    # Compute similarity matrix
    logits = image_features @ text_features.t() / temperature

    # Labels: diagonal elements are positive pairs
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)

    # Symmetric loss: image->text and text->image
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)

    return (loss_i2t + loss_t2i) / 2


class CLIPLighterModel(LighterModule):
    """CLIP-style vision-language model.

    Showcases:
    - Multi-modal (image + text) inputs
    - Custom contrastive loss
    - Temperature as learnable parameter
    - Retrieval metrics (image-to-text, text-to-image)
    """

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through both encoders."""
        return self.network(image, input_ids, attention_mask)

    def _shared_step(self, batch: dict, metrics) -> dict:
        """Shared logic for train/val/test steps."""
        image = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        # Forward pass
        image_features, text_features, temperature = self(image, input_ids, attention_mask)

        # Compute CLIP loss
        loss = clip_loss(image_features, text_features, temperature)

        # Compute retrieval accuracy
        with torch.no_grad():
            logits = image_features @ text_features.t()
            batch_size = image_features.shape[0]
            labels = torch.arange(batch_size, device=image_features.device)

            # Image-to-text retrieval accuracy
            i2t_acc = (logits.argmax(dim=1) == labels).float().mean()
            # Text-to-image retrieval accuracy
            t2i_acc = (logits.argmax(dim=0) == labels).float().mean()

        # Update metrics if available
        if metrics is not None:
            # Note: CLIP typically uses retrieval metrics, not classification metrics
            pass

        return {
            "loss": loss,
            "i2t_accuracy": i2t_acc,
            "t2i_accuracy": t2i_acc,
            "temperature": temperature.detach(),
            "image_features": image_features,
            "text_features": text_features,
        }

    def training_step(self, batch, batch_idx):
        result = self._shared_step(batch, self.train_metrics)

        # Log retrieval accuracy
        self.log("train/i2t_accuracy", result["i2t_accuracy"], prog_bar=True)
        self.log("train/t2i_accuracy", result["t2i_accuracy"])
        self.log("train/temperature", result["temperature"])

        return result

    def validation_step(self, batch, batch_idx):
        result = self._shared_step(batch, self.val_metrics)

        self.log("val/i2t_accuracy", result["i2t_accuracy"], prog_bar=True)
        self.log("val/t2i_accuracy", result["t2i_accuracy"])

        return result

    def test_step(self, batch, batch_idx):
        result = self._shared_step(batch, self.test_metrics)
        del result["loss"]

        self.log("test/i2t_accuracy", result["i2t_accuracy"])
        self.log("test/t2i_accuracy", result["t2i_accuracy"])

        return result

    def predict_step(self, batch, batch_idx):
        """Prediction step returning embeddings for retrieval."""
        image = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        image_features, text_features, temperature = self(image, input_ids, attention_mask)

        # Compute similarity
        similarity = (image_features @ text_features.t()).diag()

        return {
            "image_features": image_features.tolist(),
            "text_features": text_features.tolist(),
            "similarity": similarity.tolist(),
        }
