"""Focal Binary Cross-Entropy Loss with label smoothing."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalBCELoss(nn.Module):
    """BCE loss with focal weighting and optional label smoothing.

    Focal factor (1 - p_t)^gamma down-weights easy examples.

    Args:
        gamma: Focusing parameter. 0 = standard BCE.
        label_smoothing: Replace hard 0/1 labels with smoothed values.
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.005,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw logits (B, C) — NOT sigmoid-ed.
            targets: Binary labels in [0, 1], shape (B, C).

        Returns:
            Scalar loss.
        """
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
