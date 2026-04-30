"""timm-based backbone factory for BirdCLEF+ 2026."""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

from src.models.panns_layers import CNN14Backbone

def build_backbone(
    name: str,
    pretrained: bool = True,
    in_channels: int = 1,
    panns_path: str | None = None,
) -> tuple[nn.Module, int]:
    """Build a backbone that accepts (B, 1, H, W) mel-spectrograms.

    Returns the backbone (without classification head) and its output feature dim.
    The backbone outputs a 4D feature map (B, C, H', W').

    Args:
        name: timm model name OR 'cnn14' for PANNs CNN14.
        pretrained: Load ImageNet pretrained weights (timm only).
        in_channels: Input channels (1 for mono spectrogram).
        panns_path: Path to PANNs CNN14 checkpoint (cnn14 only).

    Returns:
        (backbone_module, feature_dim)
    """
    if name == "cnn14":
        backbone = CNN14Backbone(pretrained_path=panns_path)
        return backbone, backbone.out_channels

    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=0,       # remove classifier head
        global_pool="",      # remove global pooling — we want feature maps
        in_chans=in_channels,
    )
    # Infer feature dim with a dummy forward pass
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, 128, 128)
        feat = model(dummy)
    feature_dim = feat.shape[1]
    return model, feature_dim
