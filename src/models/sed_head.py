"""Sound Event Detection head with attention pooling."""
from __future__ import annotations

import torch
import torch.nn as nn

class SEDHead(nn.Module):
    """Produces frame-level predictions, then attention-pools to clip level.

    Architecture:
        feature_map (B, C, H, W)
        → collapse H via avg pool → (B, C, W)   [W = time frames]
        → two parallel linear projections:
            - classifier:  (B, W, n_classes)   — frame logits
            - attention:   (B, W, n_classes)   — attention weights
        → softmax over W → weighted sum → clip logits (B, n_classes)
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, n_classes)
        self.attention = nn.Linear(in_features, n_classes)

    def forward(
        self, feature_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature_map: (B, C, H, W) — backbone output.

        Returns:
            clip_logits: (B, n_classes)
            frame_logits: (B, W, n_classes)
        """
        # Collapse frequency axis H → time-sequence of feature vectors
        x = feature_map.mean(dim=2)      # (B, C, W)
        x = x.permute(0, 2, 1)          # (B, W, C)
        x = self.dropout(x)

        frame_logits = self.classifier(x)   # (B, W, n_classes)
        attn_weights = torch.softmax(self.attention(x), dim=1)  # (B, W, n_classes)

        # Weighted aggregation
        clip_logits = (frame_logits * attn_weights).sum(dim=1)  # (B, n_classes)
        return clip_logits, frame_logits
