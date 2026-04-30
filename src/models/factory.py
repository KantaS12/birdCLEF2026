"""Model factory — build BirdClassifier from config dict."""
from __future__ import annotations

from src.models.classifier import BirdClassifier

def build_model(config: dict) -> BirdClassifier:
    """Instantiate BirdClassifier from a config dict.

    Expected keys:
        backbone (str), n_classes (int), pretrained (bool),
        dropout (float, default 0.3), env_dim (int, default 0),
        panns_path (str | None, optional — for backbone='cnn14').
    """
    return BirdClassifier(
        backbone_name=config["backbone"],
        n_classes=config["n_classes"],
        pretrained=config.get("pretrained", True),
        dropout=config.get("dropout", 0.3),
        env_dim=config.get("env_dim", 0),
        panns_path=config.get("panns_path", None),
    )
