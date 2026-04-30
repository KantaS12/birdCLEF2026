"""Main training entry point for BirdCLEF+ 2026.

Usage:
    python scripts/train.py --config configs/base.yaml [overrides...]

Multiple YAML files are merged left-to-right; CLI key=value pairs override last.

Examples:
    # Stage-3 fine-tune with EfficientNetV2-S:
    python scripts/train.py \\
        --config configs/base.yaml \\
        --config configs/model/efficientnet_v2s.yaml \\
        --config configs/experiment/stage3_finetune.yaml

    # Quick smoke test with tiny data:
    python scripts/train.py \\
        --config configs/base.yaml \\
        training.epochs=1 data.batch_size=8

    # Override output dir:
    python scripts/train.py \\
        --config configs/base.yaml \\
        training.output_dir=runs/exp001
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure project root on path when running as script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.augmentations import AudioAugmenter, SpecAugmenter
from src.data.dataset import BirdDataset
from src.data.sampler import SqrtFrequencySampler
from src.losses.focal_bce import FocalBCELoss
from src.losses.soft_auc import SoftAUCLoss
from src.models.factory import build_model
from src.training.scheduler import CosineWarmupScheduler
from src.training.trainer import Trainer
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (in-place, returns base)."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base

def _set_dotted(cfg: dict, key: str, value: str) -> None:
    """Set a dotted key like 'training.epochs' to a parsed value."""
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    # Best-effort type coercion
    try:
        node[parts[-1]] = yaml.safe_load(value)
    except yaml.YAMLError:
        node[parts[-1]] = value

def load_config(config_paths: list[str], overrides: list[str]) -> dict:
    """Load and merge YAML configs, then apply CLI overrides.

    Args:
        config_paths: Ordered list of YAML file paths.
        overrides: List of 'key=value' strings.

    Returns:
        Merged config dict.
    """
    cfg: dict = {}
    for path in config_paths:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _deep_merge(cfg, data)

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be 'key=value', got: {override!r}")
        key, _, value = override.partition("=")
        _set_dotted(cfg, key.strip(), value.strip())

    return cfg

def _build_augmenter(aug_cfg: dict):
    """Build composed augmenter from config."""
    from src.data.augmentations import SpecAugmenter

    spec_aug = SpecAugmenter(
        time_mask_max=aug_cfg.get("time_mask_max", 40),
        freq_mask_max=aug_cfg.get("freq_mask_max", 8),
        n_time_masks=aug_cfg.get("n_time_masks", 2),
        n_freq_masks=aug_cfg.get("n_freq_masks", 2),
    ) if aug_cfg.get("use_spec_augment", True) else None

    noise_mixer = None
    noise_h5 = aug_cfg.get("noise_h5", "")
    if aug_cfg.get("background_noise_prob", 0.0) > 0 and noise_h5:
        from src.data.augmentations import BackgroundNoiseMixer
        noise_mixer = BackgroundNoiseMixer(
            noise_h5=noise_h5,
            snr_min_db=float(aug_cfg.get("background_noise_snr_min", 6.0)),
            snr_max_db=float(aug_cfg.get("background_noise_snr_max", 20.0)),
            p=float(aug_cfg.get("background_noise_prob", 0.5)),
        )

    if spec_aug is None and noise_mixer is None:
        return None

    class Composed:
        def __init__(self, noise_m, spec_a):
            self.noise_mixer = noise_m
            self.spec_aug = spec_a

        def __call__(self, spec):
            if self.noise_mixer is not None:
                spec = self.noise_mixer(spec)
            if self.spec_aug is not None:
                spec = self.spec_aug(spec)
            return spec

    return Composed(noise_mixer, spec_aug)

def build_loaders(cfg: dict):
    """Build train and validation DataLoaders from config."""
    from torch.utils.data import DataLoader, ConcatDataset

    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentations", {})
    spec_aug = _build_augmenter(aug_cfg)

    # Load environmental features if available
    env_lookup: dict = {}
    env_path = Path("../../data/env_features.csv")
    if env_path.exists():
        import pandas as pd
        env_df = pd.read_csv(env_path)
        # Normalize keys to match HDF5 key format (prepare_data.py uses '__' not '/')
        env_lookup = {
            row["filename"].replace("/", "__").replace("\\", "__"): np.array(
                [row["month_sin"], row["month_cos"], row["hour_sin"], row["hour_cos"]],
                dtype=np.float32
            )
            for _, row in env_df.iterrows()
        }
        logger.info("Loaded env features for %d clips", len(env_lookup))

    fold_csv = data_cfg.get("fold_csv") or None
    train_ds = BirdDataset(
        h5_path=data_cfg["train_h5"],
        augment=spec_aug,
        fold_csv=fold_csv,
        env_lookup=env_lookup,
    )

    # Optional: mix in pseudo-labeled data
    pseudo_h5 = data_cfg.get("pseudo_h5", "")
    if pseudo_h5:
        from src.data.pseudo_labels import PseudoLabelDataset
        # Compute expected spec shape from mel config so pseudo specs are
        # resampled to match the current config (A=128mel, B=64mel).
        mel_cfg = cfg.get("mel", {})
        _sr = mel_cfg.get("sample_rate", 32000)
        _hop = mel_cfg.get("hop_length", 320)
        _n_mels = mel_cfg.get("n_mels", 128)
        _T = int(_sr * 5.0) // _hop + 1  # frames for a 5-sec clip
        target_shape = (_n_mels, _T)
        pseudo_ds = PseudoLabelDataset(pseudo_h5, target_shape=target_shape)
        n_classes = train_ds.n_classes  # capture before wrapping
        train_ds = ConcatDataset([train_ds, pseudo_ds])
        train_ds.n_classes = n_classes  # re-attach for downstream use

    # Optionally mix in soundscape windows
    soundscape_h5 = data_cfg.get("soundscape_h5", "")
    if soundscape_h5 and Path(soundscape_h5).exists():
        import pandas as pd
        from src.data.soundscape_window_dataset import SoundscapeWindowDataset
        from src.data.combined_dataset import CombinedDataset
        from src.data.label_map import build_label_map

        _labeled_csv = Path("../../data/train_soundscapes_labels.csv")
        _pseudo_csv = Path("../../data/pseudo_labels/pseudo_v5.csv")
        _train_csv = Path("../../data/train.csv")

        labeled_df = pd.read_csv(str(_labeled_csv)) if _labeled_csv.exists() else pd.DataFrame()
        pseudo_df = pd.read_csv(str(_pseudo_csv)) if _pseudo_csv.exists() else pd.DataFrame()
        label_map = build_label_map(str(_train_csv)) if _train_csv.exists() else {}

        mel_cfg_dict = cfg.get("mel", {})
        mel_config = {
            "n_mels": mel_cfg_dict.get("n_mels", 128),
            "hop_length": mel_cfg_dict.get("hop_length", 320),
            "n_fft": mel_cfg_dict.get("n_fft", 1024),
            "fmin": mel_cfg_dict.get("fmin", 50.0),
            "fmax": mel_cfg_dict.get("fmax", 14000.0),
            "sample_rate": mel_cfg_dict.get("sample_rate", 32000),
        }

        n_classes_sw = cfg.get("model", {}).get("n_classes", 234)
        soundscape_ds = SoundscapeWindowDataset(
            soundscape_dir="../../data/train_soundscapes",
            labeled_df=labeled_df,
            pseudo_df=pseudo_df,
            label_map=label_map,
            n_classes=n_classes_sw,
            mel_config=mel_config,
            confidence_threshold=float(data_cfg.get("soundscape_confidence", 0.3)),
        )
        n_classes_before = getattr(train_ds, "n_classes", n_classes_sw)
        train_ds = CombinedDataset(train_ds, soundscape_ds, soundscape_ratio=0.5)
        train_ds.n_classes = n_classes_before  # re-attach for downstream use
        logger.info(
            "Combined dataset: %d samples (%d soundscape windows)",
            len(train_ds), len(soundscape_ds),
        )

    val_ds = BirdDataset(
        h5_path=data_cfg.get("val_h5", data_cfg["train_h5"]),
        fold_csv=data_cfg.get("val_fold_csv") or None,
        env_lookup=env_lookup,
    )

    sampler = None
    shuffle = True
    from src.data.combined_dataset import CombinedDataset as _CombinedDataset
    if data_cfg.get("use_sampler", False) and not isinstance(train_ds, _CombinedDataset):
        if isinstance(train_ds, ConcatDataset):
            sampler = SqrtFrequencySampler.from_concat_dataset(train_ds)
        else:
            sampler = SqrtFrequencySampler(train_ds)
        shuffle = False
    elif data_cfg.get("use_sampler", False) and isinstance(train_ds, _CombinedDataset):
        # CombinedDataset: sampler not supported; use shuffle instead
        logger.info("use_sampler disabled for CombinedDataset; using shuffle=True")

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        sampler=sampler,
        shuffle=shuffle,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )
    return train_loader, val_loader

def build_loss(cfg: dict) -> torch.nn.Module:
    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "focal_bce")
    if name == "focal_bce":
        return FocalBCELoss(
            gamma=loss_cfg.get("gamma", 2.0),
            label_smoothing=loss_cfg.get("label_smoothing", 0.005),
        )
    elif name == "soft_auc":
        return SoftAUCLoss()
    else:
        raise ValueError(f"Unknown loss: {name!r}")

def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    opt_cfg = cfg.get("optimizer", {})
    lr_head = float(opt_cfg.get("lr_head", 1e-3))
    lr_backbone = float(opt_cfg.get("lr_backbone", 1e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))

    # Split parameters: backbone vs head
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    param_groups = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    train_cfg = cfg.get("training", {})
    sched_cfg = cfg.get("scheduler", {})
    epochs = int(train_cfg.get("epochs", 30))
    warmup_epochs = int(sched_cfg.get("warmup_epochs", 1))
    min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.01))

    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    return CosineWarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BirdCLEF+ 2026 model.")
    parser.add_argument(
        "--config", "-c",
        action="append",
        dest="configs",
        default=[],
        metavar="PATH",
        help="YAML config file (may be repeated; merged left-to-right).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="key=value",
        help="Dot-notation config overrides, e.g. training.epochs=5",
    )
    return parser.parse_args()

def train_from_config(cfg: dict) -> None:
    """Run full training pipeline from a resolved config dict."""
    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info("Random seed: %d", seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_cfg = cfg.get("training", {})
    output_dir = Path(train_cfg.get("output_dir", "runs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    logger.info("Config saved to %s/config.yaml", output_dir)

    logger.info("Building data loaders…")
    train_loader, val_loader = build_loaders(cfg)
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    logger.info("Train: %d samples | Val: %d samples", n_train, n_val)
    if n_train == 0 or n_val == 0:
        raise RuntimeError(
            f"Dataset is empty (train={n_train}, val={n_val}). "
            "Run scripts/prepare_data.py first."
        )

    model_cfg = cfg["model"]
    if "n_classes" not in model_cfg:
        model_cfg["n_classes"] = train_loader.dataset.n_classes
    logger.info("Building model: %s (%d classes)", model_cfg["backbone"], model_cfg["n_classes"])
    model = build_model(model_cfg)

    # Optional: load pretrained backbone weights (non-strict, log mismatches)
    backbone_weights = model_cfg.get("backbone_weights", "")
    if backbone_weights:
        logger.info("Loading backbone weights from: %s", backbone_weights)
        ckpt = torch.load(backbone_weights, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys in backbone (%d): %s", len(missing), missing[:5])
        if unexpected:
            logger.warning("Unexpected keys in backbone (%d): %s", len(unexpected), unexpected[:5])

    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    epochs = int(train_cfg.get("epochs", 30))
    use_amp = bool(train_cfg.get("use_amp", True))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 5.0))

    frame_loss_weight = float(train_cfg.get("frame_loss_weight", 0.3))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 5))

    # Soundscape validation (uses Pantanal labeled soundscapes for early stopping)
    ss_cfg = cfg.get("soundscape_val", {})
    ss_dir = ss_cfg.get("soundscape_dir")
    ss_labels = ss_cfg.get("labels_csv")
    ss_sample_sub = ss_cfg.get("sample_sub_csv")
    mel_yaml = cfg.get("mel", {})
    mel_cfg = {
        "sr": int(mel_yaml.get("sample_rate", 32000)),
        "n_fft": int(mel_yaml.get("n_fft", 1024)),
        "hop_length": int(mel_yaml.get("hop_length", 320)),
        "n_mels": int(mel_yaml.get("n_mels", 128)),
        "fmin": float(mel_yaml.get("fmin", 50.0)),
        "fmax": float(mel_yaml.get("fmax", 14000.0)),
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        output_dir=output_dir,
        device=device,
        max_grad_norm=max_grad_norm,
        use_amp=use_amp,
        frame_loss_weight=frame_loss_weight,
        early_stopping_patience=early_stopping_patience,
        soundscape_dir=ss_dir,
        soundscape_labels_csv=ss_labels,
        sample_sub_csv=ss_sample_sub,
        mel_cfg=mel_cfg if ss_dir else None,
    )
    logger.info("Starting training for %d epochs…", epochs)
    trainer.fit(n_epochs=epochs)
    logger.info("Training complete. Outputs in: %s", output_dir)

def main() -> None:
    args = parse_args()
    if not args.configs:
        logger.error("At least one --config file must be provided.")
        sys.exit(1)
    cfg = load_config(args.configs, args.overrides)
    logger.info("Config loaded from: %s", args.configs)
    train_from_config(cfg)

if __name__ == "__main__":
    main()
