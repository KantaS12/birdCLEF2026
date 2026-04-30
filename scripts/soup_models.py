"""Weight-soup (model averaging) across folds for a backbone+config group.

Averages the state_dicts of 5-fold models into a single "souped" checkpoint.
This reduces 30 models → 6 (2 configs × 3 backbones), making Kaggle CPU
inference feasible (6 × ~57ms/window × 8400 windows ≈ 48 min).

Usage:
    # Soup all fold models in a runs directory (auto-discovers groups)
    python scripts/soup_models.py --runs-root runs_mixed_pretrain --output-root runs_souped

    # Soup one specific backbone+config group (folds 0-4 must exist)
    python scripts/soup_models.py \\
        --run-pattern "runs_longtrain/effv2s_cfga_fold{fold}" \\
        --output runs_souped/effv2s_cfga

    # Dry run
    python scripts/soup_models.py --runs-root runs_mixed_pretrain --dry-run

Output:
    runs_souped/<group>/best.pt   — averaged weights (same format as single fold)
    runs_souped/<group>/config.yaml — copied from fold 0
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Regex to detect fold suffix: e.g. "effv2s_cfga_fold0" → group "effv2s_cfga"
_FOLD_RE = re.compile(r"^(.+)_fold(\d+)$")

def soup_checkpoints(ckpt_paths: list[Path]) -> dict:
    """Average model weights from multiple checkpoints.

    Args:
        ckpt_paths: Paths to best.pt files (one per fold).

    Returns:
        State dict with averaged weights.
    """
    assert len(ckpt_paths) >= 2, "Need at least 2 checkpoints to soup"

    accum: dict[str, torch.Tensor] | None = None
    for i, p in enumerate(ckpt_paths):
        ckpt = torch.load(p, map_location="cpu")
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        if accum is None:
            accum = {k: v.clone().float() for k, v in state.items()}
        else:
            for k in accum:
                accum[k] += state[k].float()

    n = len(ckpt_paths)
    souped = {k: (v / n).to(list(ckpt["model_state"].values())[0].dtype)
              for k, v in accum.items()}
    return souped

def soup_group(
    run_dirs: list[Path],
    output_dir: Path,
    dry_run: bool = False,
) -> bool:
    """Soup all folds in a group and write to output_dir."""
    ckpt_paths = [d / "best.pt" for d in run_dirs]
    missing = [p for p in ckpt_paths if not p.exists()]
    if missing:
        logger.warning("Missing checkpoints in group %s: %s", output_dir.name, missing)
        return False

    if dry_run:
        logger.info("[DRY RUN] Would soup %d folds → %s", len(run_dirs), output_dir)
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = output_dir / "best.pt"
    out_cfg = output_dir / "config.yaml"

    if out_ckpt.exists():
        logger.info("Already souped: %s — skipping", output_dir)
        return True

    logger.info("Souping %d folds → %s", len(run_dirs), output_dir)
    souped_state = soup_checkpoints(ckpt_paths)

    # Load reference ckpt to get epoch/auc metadata from best fold
    ref_ckpt = torch.load(ckpt_paths[0], map_location="cpu")
    torch.save(
        {
            "epoch": ref_ckpt.get("epoch", -1),
            "model_state": souped_state,
            "auc": ref_ckpt.get("auc", None),
            "souped_from": [str(p) for p in ckpt_paths],
            "n_folds": len(ckpt_paths),
        },
        out_ckpt,
    )
    logger.info("  Saved: %s", out_ckpt)

    # Copy config from fold 0 (remove fold-specific keys)
    cfg_path = run_dirs[0] / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        # Remove fold-specific data references
        cfg.get("data", {}).pop("fold_csv", None)
        cfg.get("data", {}).pop("val_fold_csv", None)
        cfg.get("training", {})["output_dir"] = str(output_dir)
        with open(out_cfg, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        logger.info("  Config: %s", out_cfg)

    return True

def find_groups(runs_root: Path) -> dict[str, list[Path]]:
    """Group run directories by backbone+config (strip _fold* suffix)."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        m = _FOLD_RE.match(d.name)
        if m:
            group_name = m.group(1)
            groups[group_name].append(d)
    return dict(groups)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average model weights across folds (weight soup).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--runs-root", type=Path,
        help="Root directory containing fold subdirectories (e.g. runs_mixed_pretrain/).",
    )
    src.add_argument(
        "--run-pattern", type=str,
        help="Pattern with {fold} placeholder, e.g. 'runs/effv2s_cfga_fold{fold}'.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Where to write souped checkpoints. Default: runs_souped/ next to runs-root.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output dir for --run-pattern mode.",
    )
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Number of folds (default: 5).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.run_pattern:
        # Single group mode
        run_dirs = [Path(args.run_pattern.format(fold=i)) for i in range(args.folds)]
        output = args.output
        if output is None:
            logger.error("--output is required with --run-pattern")
            sys.exit(1)
        ok = soup_group(run_dirs, output, dry_run=args.dry_run)
        sys.exit(0 if ok else 1)

    # Multi-group mode
    groups = find_groups(args.runs_root)
    if not groups:
        logger.error("No fold directories found under %s", args.runs_root)
        sys.exit(1)

    output_root = args.output_root or args.runs_root.parent / "runs_souped"
    logger.info("Found %d groups under %s → %s", len(groups), args.runs_root, output_root)

    n_ok = n_fail = 0
    for group_name, run_dirs in sorted(groups.items()):
        out_dir = output_root / group_name
        ok = soup_group(run_dirs, out_dir, dry_run=args.dry_run)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    logger.info("Done: %d souped, %d failed", n_ok, n_fail)
    if n_fail > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
