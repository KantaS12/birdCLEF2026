"""Full BirdCLEF classifier: backbone + SED head."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.backbone import build_backbone
from src.models.sed_head import SEDHead

class BirdClassifier(nn.Module):
    """End-to-end model for multi-label bird species classification.

    Input:  mel-spectrogram (B, 1, n_mels, time_frames)
    Output: dict with 'clip_logits' (B, n_classes) and 'frame_logits' (B, T, n_classes)

    Optionally accepts environmental context vector (month, hour sin/cos encodings).
    """

    def __init__(
        self,
        backbone_name: str = "tf_efficientnetv2_s_in21k",
        n_classes: int = 206,
        pretrained: bool = True,
        dropout: float = 0.3,
        env_dim: int = 0,
        panns_path: str | None = None,
    ) -> None:
        super().__init__()
        self.backbone, feat_dim = build_backbone(
            backbone_name, pretrained=pretrained, panns_path=panns_path
        )
        head_in = feat_dim + env_dim
        self.sed_head = SEDHead(in_features=head_in, n_classes=n_classes, dropout=dropout)
        self.env_dim = env_dim

    def forward(
        self,
        spec: torch.Tensor,
        env: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            spec: (B, 1, F, T)
            env:  (B, env_dim) optional environmental context.

        Returns:
            {'clip_logits': (B, n_classes), 'frame_logits': (B, T, n_classes)}
        """
        feat_map = self.backbone(spec)  # (B, C, H, W)

        if env is not None and self.env_dim > 0:
            B, C, H, W = feat_map.shape
            env_expanded = env.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
            feat_map = torch.cat([feat_map, env_expanded], dim=1)

        clip_logits, frame_logits = self.sed_head(feat_map)
        return {"clip_logits": clip_logits, "frame_logits": frame_logits}
