"""SimCLR model using lightly library with LighterModule."""

import torch

from lighter import LighterModule


class SimCLRModel(LighterModule):
    """SimCLR self-supervised learning model using lightly.

    Uses lightly's NTXentLoss for contrastive learning.
    Temperature is configured via the loss function.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
        temperature: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(
            network=network,
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs,
        )
        # Use lightly's NTXentLoss
        from lightly.loss import NTXentLoss

        self.ssl_loss = NTXentLoss(temperature=temperature)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder."""
        return self.network(x)

    def training_step(self, batch, batch_idx):
        """Training step with contrastive loss.

        Lightly's LightlyDataset returns (view0, view1, label, filename).
        """
        # Unpack batch from LightlyDataset
        (view0, view1), targets, filenames = batch

        # Get projections for both views
        _, z0 = self(view0)
        _, z1 = self(view1)

        # Compute contrastive loss
        loss = self.ssl_loss(z0, z1)

        self.log("train/loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        (view0, view1), targets, filenames = batch

        _, z0 = self(view0)
        _, z1 = self(view1)

        loss = self.ssl_loss(z0, z1)

        self.log("val/loss", loss, prog_bar=True)
        return {"loss": loss}

    def predict_step(self, batch, batch_idx):
        """Extract features for downstream tasks."""
        (view0, view1), targets, filenames = batch

        # Get features (not projections) for downstream tasks
        features, _ = self(view0)

        return {
            "features": features,
            "targets": targets,
            "filenames": filenames,
        }
