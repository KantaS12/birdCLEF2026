"""Generate pseudo-labels from a weighted ensemble over unlabeled soundscapes.

Uses adaptive per-class thresholds (lower defaults than clip-trained thresholds
because soundscape models are underconfident due to domain shift):
  - rare   (<30 samples)  → threshold 0.15 (default)
  - medium (30-199)       → threshold 0.20 (default)
  - common (>=200)        → threshold 0.25 (default)

Override via --threshold-rare / --threshold-medium / --threshold-common.

--binarize: store 0/1 labels (prob >= per-class threshold) instead of raw probs.
--min-max-confidence: skip rows where max(prob_vec) < this value (row-level filter).

Usage:
    python scripts/generate_pseudolabels.py \\
        --soundscapes-dir ../../data/train_soundscapes \\
        --weights-json runs/ensemble_weights.json \\
        --train-h5 ../../data/cache/all_config_a.h5 \\
        --output ../../data/pseudo_labels/pseudo_v1.csv \\
        --thresholds-out ../../data/pseudo_labels/thresholds.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import SoundscapeDataset
from src.utils.metrics import macro_auc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def compute_thresholds(
    train_h5: Path,
    thr_rare: float = 0.15,
    thr_medium: float = 0.20,
    thr_common: float = 0.25,
) -> np.ndarray:
    """Return per-class thresholds based on sample counts."""
    with h5py.File(train_h5, "r") as hf:
        n_classes = int(hf.attrs["n_classes"])
        counts = np.zeros(n_classes, dtype=np.int64)
        for key in hf["labels"]:
            counts += hf["labels"][key][:].astype(np.int64)
    thresholds = np.where(counts < 30, thr_rare, np.where(counts < 200, thr_medium, thr_common))
    logger.info("Thresholds: rare(%.2f) medium(%.2f) common(%.2f) | "
                "rare=%d medium=%d common=%d classes",
                thr_rare, thr_medium, thr_common,
                (counts < 30).sum(), ((counts >= 30) & (counts < 200)).sum(), (counts >= 200).sum())
    return thresholds

def load_model(model_path: str, device: str):
    """Load a checkpoint and return (model, n_classes)."""
    from src.models.factory import build_model
    ckpt = torch.load(model_path, map_location=device)
    cfg_path = Path(model_path).parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg["model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, cfg["model"]["n_classes"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--soundscapes-dir", required=True, type=Path)
    p.add_argument("--weights-json", required=True, type=Path,
                   help="ensemble_weights.json from tune_ensemble_weights.py")
    p.add_argument("--train-h5", required=True, type=Path,
                   help="Training HDF5 for computing label counts")
    p.add_argument("--output", type=Path, default=Path("../../data/pseudo_labels/pseudo_v1.csv"))
    p.add_argument("--thresholds-out", type=Path,
                   default=Path("../../data/pseudo_labels/thresholds.json"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--top-n-models", type=int, default=6,
                   help="Use top N models by weight from weights_json")
    p.add_argument("--threshold-rare", type=float, default=0.15,
                   help="Confidence threshold for rare classes (<30 samples)")
    p.add_argument("--threshold-medium", type=float, default=0.20,
                   help="Confidence threshold for medium classes (30-199 samples)")
    p.add_argument("--threshold-common", type=float, default=0.25,
                   help="Confidence threshold for common classes (>=200 samples)")
    p.add_argument("--binarize", action="store_true",
                   help="Store 0/1 labels (prob >= per-class threshold) instead of raw probabilities")
    p.add_argument("--min-max-confidence", type=float, default=0.0,
                   help="Skip rows where max(prob_vec) < this value (row-level filter)")
    p.add_argument("--runs-root", type=Path, default=None,
                   help="Override directory holding {name}_fold*/best.pt or {name}/best.pt "
                        "(default: <repo>/runs)")
    return p.parse_args()

def main():
    args = parse_args()

    with open(args.weights_json) as f:
        weights_data = json.load(f)

    # Take top-N models by descending weight
    sorted_models = sorted(weights_data["models"], key=lambda x: x["weight"], reverse=True)
    top_models = sorted_models[:args.top_n_models]
    logger.info("Using %d models (top %d by weight)", len(top_models), args.top_n_models)

    # Build runs/ root (override via --runs-root)
    runs_root = args.runs_root if args.runs_root is not None \
        else Path(__file__).parent.parent / "runs"
    logger.info("Using runs_root=%s", runs_root)

    # Each name in weights_json (e.g. "nfnet_cfgb") may have per-fold checkpoints
    # (e.g. runs/nfnet_cfgb_fold0/best.pt ... fold4/best.pt).
    # Collect all fold checkpoints for each name and treat each fold as an equal
    # sub-weight within that model's total ensemble weight.
    models_and_weights = []
    n_classes = None
    for m in top_models:
        name = m["name"]
        weight = float(m["weight"])
        # Look for per-fold dirs first, fall back to single dir
        fold_ckpts = sorted(runs_root.glob(f"{name}_fold*/best.pt"))
        if not fold_ckpts:
            single_ckpt = runs_root / name / "best.pt"
            if not single_ckpt.exists():
                raise FileNotFoundError(
                    f"No checkpoint found for model '{name}'. "
                    f"Tried {fold_ckpts} and {single_ckpt}"
                )
            fold_ckpts = [single_ckpt]
        logger.info("Model %s: loading %d fold checkpoint(s)", name, len(fold_ckpts))
        per_fold_weight = weight / len(fold_ckpts)
        for ckpt_path in fold_ckpts:
            model, nc = load_model(str(ckpt_path), args.device)
            if n_classes is None:
                n_classes = nc
            models_and_weights.append((model, per_fold_weight))

    thresholds = compute_thresholds(
        args.train_h5,
        thr_rare=args.threshold_rare,
        thr_medium=args.threshold_medium,
        thr_common=args.threshold_common,
    )
    args.thresholds_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.thresholds_out, "w") as f:
        json.dump(thresholds.tolist(), f)
    logger.info("Per-class thresholds saved to %s", args.thresholds_out)

    ds = SoundscapeDataset(audio_dir=args.soundscapes_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    logger.info("Soundscape windows: %d", len(ds))

    # Normalise weights
    total_w = sum(w for _, w in models_and_weights)
    norm_weights = [w / total_w for _, w in models_and_weights]

    rows = []
    n_batches = len(loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx % 100 == 0:
                logger.info("Batch %d/%d (%.1f%%)", batch_idx, n_batches,
                            100.0 * batch_idx / n_batches)
            spec_batch, path_stems, start_samples = batch
            spec_batch = spec_batch.to(args.device)
            ensemble_probs = None
            for (model, _), w in zip(models_and_weights, norm_weights):
                env = (torch.zeros(spec_batch.shape[0], model.env_dim, device=args.device)
                       if model.env_dim > 0 else None)
                out = model(spec_batch, env=env)
                probs = torch.sigmoid(out["clip_logits"].float()).cpu().numpy()
                if ensemble_probs is None:
                    ensemble_probs = w * probs
                else:
                    ensemble_probs += w * probs

            for i, (stem, start) in enumerate(zip(path_stems, start_samples)):
                prob_vec = ensemble_probs[i]
                if not (prob_vec >= thresholds).any():
                    continue
                if args.min_max_confidence > 0.0 and prob_vec.max() < args.min_max_confidence:
                    continue
                row = {"row_id": f"{stem}_{int(start)}"}
                if args.binarize:
                    label_vec = (prob_vec >= thresholds).astype(np.float32)
                else:
                    label_vec = prob_vec
                row.update({str(c): float(label_vec[c]) for c in range(n_classes)})
                rows.append(row)

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved %d pseudo-label rows to %s", len(df), args.output)

if __name__ == "__main__":
    main()
