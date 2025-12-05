"""SimCLR model using lightly library with LighterModule.

This module implements the SimCLR (Simple Framework for Contrastive Learning of
Visual Representations) self-supervised learning method using the lightly library.

SimCLR learns representations by maximizing agreement between differently augmented
views of the same image via a contrastive loss (NT-Xent).

Reference:
    Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations"
    https://arxiv.org/abs/2002.05709

Requirements:
    pip install lightly
"""

from typing import Any

import torch

from lighter import LighterModule


class SimCLRModel(LighterModule):
    """SimCLR self-supervised learning model using lightly.

    This model implements contrastive learning where two augmented views of
    the same image are pulled together in the embedding space while views
    from different images are pushed apart.

    The model uses NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
    from the lightly library, which is an efficient implementation of the
    InfoNCE loss used in SimCLR.

    Args:
        network: Encoder network that returns (features, projections).
            The network should output a tuple where:
            - features: Representations for downstream tasks
            - projections: Embeddings used for contrastive loss
        optimizer: Optimizer for training. Required for training.
        scheduler: Learning rate scheduler. Optional.
        temperature: Temperature parameter for NT-Xent loss. Lower values
            make the model focus more on hard negatives. Default: 0.5.
        **kwargs: Additional arguments passed to LighterModule.

    Example:
        ```yaml
        model:
          _target_: project.models.SimCLRModel
          temperature: 0.5
          network:
            _target_: project.networks.encoder.create_simclr_model
            backbone: resnet18
          optimizer:
            _target_: torch.optim.SGD
            params: $@model::network.parameters()
            lr: 0.06
        ```

    Note:
        - Use `drop_last: true` in dataloaders for contrastive learning
        - Batch size significantly impacts performance (larger is better)
        - The network must return (features, projections) tuple
    """

    def __init__(
        self,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        temperature: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            network=network,
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs,
        )
        # Import here to allow module to load even if lightly is not installed,
        # providing a clear error message when the model is actually instantiated.
        from lightly.loss import NTXentLoss

        self.ssl_loss = NTXentLoss(temperature=temperature)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder.

        Args:
            x: Input images of shape [B, C, H, W].

        Returns:
            Tuple of (features, projections) where:
            - features: Shape [B, feature_dim] for downstream tasks
            - projections: Shape [B, projection_dim] for contrastive loss
        """
        return self.network(x)

    def training_step(self, batch: tuple, batch_idx: int) -> dict[str, torch.Tensor]:
        """Training step with contrastive loss.

        Computes NT-Xent loss between projections of two augmented views.
        The loss encourages the model to produce similar representations
        for different augmentations of the same image.

        Args:
            batch: Tuple from LightlyDataset containing:
                - (view0, view1): Two augmented views of each image
                - targets: Original labels (unused in SSL)
                - filenames: Image filenames
            batch_idx: Index of the current batch.

        Returns:
            Dict with 'loss' key for automatic logging.
        """
        (view0, view1), targets, filenames = batch

        # Get projections for both views
        _, z0 = self(view0)
        _, z1 = self(view1)

        # Compute contrastive loss
        loss = self.ssl_loss(z0, z1)

        self.log("train/loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: tuple, batch_idx: int) -> dict[str, torch.Tensor]:
        """Validation step with contrastive loss.

        Same as training_step but without gradient computation.

        Args:
            batch: Tuple from LightlyDataset (see training_step).
            batch_idx: Index of the current batch.

        Returns:
            Dict with 'loss' key for automatic logging.
        """
        (view0, view1), targets, filenames = batch

        _, z0 = self(view0)
        _, z1 = self(view1)

        loss = self.ssl_loss(z0, z1)

        self.log("val/loss", loss, prog_bar=True)
        return {"loss": loss}

    def predict_step(self, batch: tuple, batch_idx: int) -> dict[str, Any]:
        """Extract features for downstream tasks.

        Returns features (not projections) which are typically used for
        downstream classification or other tasks via linear probing or
        fine-tuning.

        Args:
            batch: Tuple from LightlyDataset (see training_step).
            batch_idx: Index of the current batch.

        Returns:
            Dict containing:
            - features: Extracted feature representations
            - targets: Original labels for evaluation
            - filenames: Image filenames for identification
        """
        (view0, view1), targets, filenames = batch

        # Get features (not projections) for downstream tasks
        features, _ = self(view0)

        return {
            "features": features,
            "targets": targets,
            "filenames": filenames,
        }
