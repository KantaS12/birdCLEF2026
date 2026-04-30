"""Export trained BirdClassifier models to ONNX and OpenVINO IR (FP16).

Usage examples:
    # Export a single run directory
    python scripts/export_models.py --run-dir runs_longtrain/effv2s_cfga_fold0

    # Export all runs in a directory tree (globs for best.pt)
    python scripts/export_models.py --runs-root runs_mixed_pretrain --output-root exports/mixed_pretrain

    # Export only to ONNX (skip OpenVINO step)
    python scripts/export_models.py --run-dir runs_longtrain/effv2s_cfga_fold0 --skip-openvino

    # Dry run: just print what would be exported
    python scripts/export_models.py --runs-root runs_mixed_pretrain --dry-run

Output layout (mirrors run tree):
    exports/<run_name>/model.onnx
    exports/<run_name>/model_fp16/model.xml   (OpenVINO IR)
    exports/<run_name>/model_fp16/model.bin
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.factory import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

class ExportWrapper(nn.Module):
    """Wraps BirdClassifier for ONNX export.

    The original model returns a dict; ONNX torch.export needs tensor outputs.
    Also bakes-in env_dim so we can handle zero-env inference cleanly.

    Inputs:
        spec: (B, 1, n_mels, T)  — mel-spectrogram
        env:  (B, env_dim)       — environmental context (use zeros if unknown)

    Output:
        clip_logits: (B, n_classes)  — raw logits (apply sigmoid for probs)
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, spec: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        out = self.model(spec, env)
        return out["clip_logits"]

def _spec_shape_from_config(cfg: dict) -> tuple[int, int]:
    """Return (n_mels, T) for the 5-sec window from a run's config."""
    mel = cfg.get("mel", {})
    n_mels = int(mel.get("n_mels", 128))
    hop = int(mel.get("hop_length", 320))
    sr = int(mel.get("sample_rate", 32000))
    T = int(sr * 5.0) // hop + 1
    return n_mels, T

def _fix_bn_training_mode(onnx_path: Path) -> int:
    """Set training_mode=0 on all BatchNormalization nodes in an ONNX model.

    Some backbones (e.g. eca_nfnet_l0) export BN with training_mode=1 even
    in eval mode. OpenVINO rejects this; onnxruntime accepts it but behaves
    differently. This fix makes the graph portable to all runtimes.

    Returns the number of BN nodes fixed.
    """
    import onnx
    model = onnx.load(str(onnx_path))
    fixed = 0
    for node in model.graph.node:
        if node.op_type == "BatchNormalization":
            for attr in node.attribute:
                if attr.name == "training_mode" and attr.i != 0:
                    attr.i = 0
                    # Eval-mode BN outputs only Y; drop extra training outputs
                    while len(node.output) > 1:
                        node.output.pop()
                    fixed += 1
    if fixed:
        onnx.save(model, str(onnx_path))
    return fixed

def export_onnx(
    wrapper: ExportWrapper,
    onnx_path: Path,
    n_mels: int,
    T: int,
    env_dim: int,
) -> None:
    """Export wrapper to ONNX with dynamic batch size."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_spec = torch.zeros(1, 1, n_mels, T)
    dummy_env = torch.zeros(1, env_dim)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            wrapper,
            (dummy_spec, dummy_env),
            str(onnx_path),
            input_names=["spec", "env"],
            output_names=["clip_logits"],
            dynamic_axes={
                "spec": {0: "batch"},
                "env": {0: "batch"},
                "clip_logits": {0: "batch"},
            },
            opset_version=14,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
        )

    # Fix BN training_mode for OpenVINO compatibility
    n_fixed = _fix_bn_training_mode(onnx_path)
    if n_fixed:
        logger.info("Fixed %d BN nodes (training_mode → eval)", n_fixed)

    logger.info("ONNX saved: %s", onnx_path)

def verify_onnx(onnx_path: Path, n_mels: int, T: int, env_dim: int) -> bool:
    """Run ONNX model with onnxruntime and compare to torch output."""
    try:
        import onnxruntime as ort
        import onnx
        onnx.checker.check_model(str(onnx_path))

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        spec_np = np.zeros((1, 1, n_mels, T), dtype=np.float32)
        env_np = np.zeros((1, env_dim), dtype=np.float32)
        outputs = sess.run(None, {"spec": spec_np, "env": env_np})
        logits = outputs[0]
        assert logits.shape == (1, logits.shape[1]), f"Unexpected shape: {logits.shape}"
        logger.info("ONNX verified OK — output shape: %s", logits.shape)
        return True
    except Exception as e:
        logger.error("ONNX verification failed: %s", e)
        return False

def export_openvino(onnx_path: Path, ov_dir: Path) -> bool:
    """Convert ONNX to OpenVINO IR FP16.

    Uses the 2024+ openvino API: ov.Core().read_model() + ov.save_model().
    compress_to_fp16=True stores weights in FP16, keeping activations in FP32
    (standard for fast CPU inference without accuracy loss).
    """
    try:
        import openvino as ov
        ov_dir.mkdir(parents=True, exist_ok=True)

        core = ov.Core()
        model = core.read_model(str(onnx_path))
        xml_path = ov_dir / "model.xml"
        ov.save_model(model, str(xml_path), compress_to_fp16=True)
        logger.info("OpenVINO FP16 saved: %s", xml_path)
        return True
    except Exception as e:
        logger.error("OpenVINO export failed: %s", e)
        logger.error("Tip: make sure openvino>=2024.x is installed. ONNX model is still usable.")
        return False

def export_run(
    run_dir: Path,
    output_dir: Path,
    skip_openvino: bool = False,
    dry_run: bool = False,
) -> bool:
    """Export one trained run (run_dir must contain best.pt + config.yaml).

    Returns True on success.
    """
    ckpt_path = run_dir / "best.pt"
    cfg_path = run_dir / "config.yaml"

    if not ckpt_path.exists():
        logger.warning("No best.pt in %s — skipping", run_dir)
        return False
    if not cfg_path.exists():
        logger.warning("No config.yaml in %s — skipping", run_dir)
        return False

    onnx_path = output_dir / "model.onnx"
    ov_dir = output_dir / "model_fp16"

    # Check if already exported
    if onnx_path.exists() and (skip_openvino or (ov_dir / "model.xml").exists()):
        logger.info("Already exported: %s — skipping", output_dir)
        return True

    if dry_run:
        logger.info("[DRY RUN] Would export %s → %s", run_dir, output_dir)
        return True

    logger.info("Exporting: %s", run_dir)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    n_mels, T = _spec_shape_from_config(cfg)
    env_dim = model_cfg.get("env_dim", 0)

    logger.info(
        "  backbone=%s  n_classes=%d  n_mels=%d  T=%d  env_dim=%d",
        model_cfg["backbone"], model_cfg["n_classes"], n_mels, T, env_dim,
    )

    # Load model
    model = build_model({**model_cfg, "pretrained": False})  # don't download weights
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    wrapper = ExportWrapper(model)
    wrapper.eval()

    # ONNX
    export_onnx(wrapper, onnx_path, n_mels, T, env_dim)

    # Verify ONNX
    ok = verify_onnx(onnx_path, n_mels, T, env_dim)
    if not ok:
        logger.error("Export failed for %s (ONNX verify error)", run_dir)
        return False

    # OpenVINO
    if not skip_openvino:
        export_openvino(onnx_path, ov_dir)

    return True

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export BirdCLEF models to ONNX + OpenVINO FP16.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--run-dir", type=Path,
        help="Single run directory (contains best.pt + config.yaml).",
    )
    src_group.add_argument(
        "--runs-root", type=Path,
        help="Root of a tree of run directories. All subdirs with best.pt are exported.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help=(
            "Where to write exports. Defaults to 'exports/<runs-root-name>' "
            "or 'exports/<run-dir-name>'."
        ),
    )
    parser.add_argument(
        "--skip-openvino", action="store_true",
        help="Only export to ONNX; skip OpenVINO conversion.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be exported without doing anything.",
    )
    return parser.parse_args()

def find_run_dirs(root: Path) -> list[Path]:
    """Find all directories under root that contain best.pt."""
    run_dirs = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        if "best.pt" in filenames and "config.yaml" in filenames:
            run_dirs.append(Path(dirpath))
    run_dirs.sort()
    return run_dirs

def main() -> None:
    args = parse_args()

    if args.run_dir is not None:
        run_dirs = [args.run_dir]
        default_root = Path("exports") / args.run_dir.name
    else:
        run_dirs = find_run_dirs(args.runs_root)
        default_root = Path("exports") / args.runs_root.name
        logger.info("Found %d run directories under %s", len(run_dirs), args.runs_root)

    output_root = args.output_root or default_root

    if not run_dirs:
        logger.error("No run directories found with best.pt + config.yaml")
        sys.exit(1)

    n_ok = 0
    n_fail = 0
    for run_dir in run_dirs:
        # Mirror the relative path under output_root
        try:
            rel = run_dir.relative_to(args.runs_root or args.run_dir.parent)
        except ValueError:
            rel = Path(run_dir.name)
        out_dir = output_root / rel

        ok = export_run(
            run_dir=run_dir,
            output_dir=out_dir,
            skip_openvino=args.skip_openvino,
            dry_run=args.dry_run,
        )
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    logger.info("Done: %d succeeded, %d failed", n_ok, n_fail)
    if n_fail > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
