"""Tune ensemble weights using labeled train soundscapes (not clean-clip OOF).

This directly optimizes what Kaggle evaluates: macro-AUC on field soundscapes.
Each model in --exports-dir is run on the 66 labeled soundscape files.
Weights are found by Nelder-Mead to maximize soundscape macro-AUC.

Usage:
    python scripts/tune_soundscape_ensemble.py \
        --exports-dir exports/souped_combined_sc_aug05 \
        --exports-dir exports/souped_soundscape_r1 \
        --exports-dir exports/souped_convnext_r2 \
        --output soundscape_ensemble_weights.json
"""
import argparse, json, logging, time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE       = Path("/mnt/lustre/koa/scratch/kantas/birdCLEF2026")
SS_DIR     = BASE / "data/train_soundscapes"
LABELS_CSV = BASE / "data/train_soundscapes_labels.csv"
SAMPLE_SUB = BASE / "data/sample_submission.csv"
TARGET_SR  = 32000
WINDOW_SEC = 5.0
SLIDE_HOP  = 2.5
ENV_DIM    = 4

CFG_A = dict(n_mels=128, hop_length=320, n_fft=1024, fmin=50.0, fmax=14000.0, sr=32000)
CFG_B = dict(n_mels=64,  hop_length=160, n_fft=1024, fmin=20.0, fmax=16000.0, sr=32000)

# Map model directory name suffix to mel config
def mel_cfg_for(name: str) -> dict:
    return CFG_B if (name.endswith("cfgb") or name.startswith("cnn14")) else CFG_A

def load_audio(path) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    if sr != TARGET_SR:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * TARGET_SR / sr)).astype(np.float32)
    return audio

def split_windows(audio: np.ndarray) -> tuple[list, list]:
    ws = int(TARGET_SR * WINDOW_SEC)
    hs = int(TARGET_SR * SLIDE_HOP)
    windows, starts = [], []
    start = 0
    while start < len(audio):
        chunk = audio[start:start + ws]
        if len(chunk) < ws:
            chunk = np.pad(chunk, (0, ws - len(chunk)))
        windows.append(chunk.astype(np.float32))
        starts.append(start)
        start += hs
    return windows, starts

def compute_melspec(audio: np.ndarray, cfg: dict) -> np.ndarray:
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=cfg["sr"], n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"], n_mels=cfg["n_mels"],
        fmin=cfg["fmin"], fmax=cfg["fmax"], power=2.0,
    )
    log_mel = np.log1p(mel).astype(np.float32)
    mu, sigma = log_mel.mean(), log_mel.std()
    if sigma > 1e-6:
        log_mel = (log_mel - mu) / sigma
    return log_mel[np.newaxis]

class OVModel:
    def __init__(self, xml_path: Path):
        import openvino as ov
        core = ov.Core()
        core.set_property("CPU", {"INFERENCE_NUM_THREADS": "4"})
        self.compiled = core.compile_model(core.read_model(str(xml_path)), "CPU")
        self.req = self.compiled.create_infer_request()

    def predict(self, spec: np.ndarray, env: np.ndarray) -> np.ndarray:
        self.req.infer({"spec": spec, "env": env})
        return self.req.get_output_tensor(0).data.copy()

def build_targets(label_df: pd.DataFrame, species_cols: list,
                  files: list, n_bins_per_file: int) -> np.ndarray:
    """Build (n_windows, n_classes) target array from soundscape labels."""
    species_idx = {s: i for i, s in enumerate(species_cols)}
    n_classes = len(species_cols)
    ws = int(TARGET_SR * WINDOW_SEC)
    targets = []
    for fname in files:
        audio_path = SS_DIR / fname
        if not audio_path.exists():
            continue
        audio = load_audio(audio_path)
        _, starts = split_windows(audio)
        n_out = int(np.ceil(len(audio) / ws))
        file_rows = label_df[label_df["filename"] == fname].sort_values("start")

        for b in range(n_out):
            label_vec = np.zeros(n_classes, dtype=np.float32)
            if b < len(file_rows):
                for sp in str(file_rows.iloc[b]["primary_label"]).split(";"):
                    sp = sp.strip()
                    if sp in species_idx:
                        label_vec[species_idx[sp]] = 1.0
            targets.append(label_vec)
    return np.array(targets, dtype=np.float32)

def predict_model_on_soundscapes(xml_path: Path, mel_cfg: dict,
                                  label_df: pd.DataFrame, species_cols: list,
                                  files: list) -> np.ndarray:
    """Run one model on all soundscapes. Returns (n_output_bins, n_classes) probabilities."""
    n_classes = len(species_cols)
    ws = int(TARGET_SR * WINDOW_SEC)
    env = np.zeros((1, ENV_DIM), dtype=np.float32)
    model = OVModel(xml_path)
    all_probs = []

    for fname in files:
        audio_path = SS_DIR / fname
        if not audio_path.exists():
            continue
        audio = load_audio(audio_path)
        slide_windows, slide_starts = split_windows(audio)
        n_out = int(np.ceil(len(audio) / ws))

        # Overlapping window accumulation
        acc = np.zeros((n_out, n_classes), dtype=np.float64)
        counts = np.zeros(n_out, dtype=np.int32)
        for i, (win, ss) in enumerate(zip(slide_windows, slide_starts)):
            spec = compute_melspec(win, mel_cfg)[np.newaxis]
            logits = model.predict(spec, env)[0]
            s = ss / TARGET_SR
            e = s + WINDOW_SEC
            b_min = max(0, int(s / WINDOW_SEC))
            b_max = min(n_out - 1, int((e - 1e-9) / WINDOW_SEC))
            for b in range(b_min, b_max + 1):
                acc[b] += logits
                counts[b] += 1
        counts = np.maximum(counts, 1)
        probs = 1.0 / (1.0 + np.exp(-acc / counts[:, None]))
        all_probs.append(probs.astype(np.float32))

    return np.concatenate(all_probs, axis=0)

def macro_auc(weights: np.ndarray, preds_list: list, targets: np.ndarray) -> float:
    """Weighted ensemble AUC (negated for minimization)."""
    w = np.abs(weights)
    w = w / w.sum()
    ensemble = sum(w[i] * preds_list[i] for i in range(len(preds_list)))
    aucs = []
    for j in range(targets.shape[1]):
        if targets[:, j].sum() > 0:
            try:
                aucs.append(roc_auc_score(targets[:, j], ensemble[:, j]))
            except Exception:
                pass
    return -float(np.mean(aucs)) if aucs else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports-dir", action="append", required=True,
                        help="Export directory containing model subdirs. Repeat for multiple.")
    parser.add_argument("--output", default="soundscape_ensemble_weights.json")
    args = parser.parse_args()

    label_df = pd.read_csv(LABELS_CSV)
    sub = pd.read_csv(SAMPLE_SUB)
    species_cols = [c for c in sub.columns if c != "row_id"]
    files = sorted(label_df["filename"].unique())
    log.info("Soundscape files: %d  Species: %d  Label rows: %d",
             len(files), len(species_cols), len(label_df))

    # Collect all model XML paths across export dirs
    models = []  # (name, xml_path, mel_cfg)
    for exp_dir in args.exports_dir:
        exp = Path(exp_dir)
        for model_dir in sorted(exp.iterdir()):
            xml = model_dir / "model_fp16" / "model.xml"
            if xml.exists():
                name = f"{exp.name}/{model_dir.name}"
                models.append((name, xml, mel_cfg_for(model_dir.name)))
                log.info("  Found model: %s", name)

    if not models:
        raise RuntimeError("No model_fp16/model.xml found under any --exports-dir")
    log.info("Total models: %d", len(models))

    # Build target array
    log.info("Building target array...")
    targets = build_targets(label_df, species_cols, files, 12)
    log.info("Targets shape: %s  Positives: %d classes", targets.shape, (targets.sum(0) > 0).sum())

    # Run each model on soundscapes
    preds_list = []
    names = []
    for name, xml, cfg in models:
        log.info("Running %s...", name)
        t0 = time.time()
        preds = predict_model_on_soundscapes(xml, cfg, label_df, species_cols, files)
        # Align to targets length
        preds = preds[:targets.shape[0]]
        preds_list.append(preds)
        names.append(name)
        auc_solo, n_cls = -macro_auc(np.ones(1), [preds], targets), 0
        aucs = [roc_auc_score(targets[:, j], preds[:, j])
                for j in range(targets.shape[1]) if targets[:, j].sum() > 0]
        log.info("  %s solo AUC: %.5f  [%.0fs]", name, np.mean(aucs), time.time() - t0)

    # Equal-weight baseline
    eq_auc = -macro_auc(np.ones(len(preds_list)), preds_list, targets)
    log.info("Equal-weight soundscape AUC: %.5f", eq_auc)

    # Nelder-Mead optimisation
    log.info("Optimising ensemble weights on soundscape AUC...")
    x0 = np.ones(len(preds_list)) / len(preds_list)
    result = minimize(macro_auc, x0, args=(preds_list, targets),
                      method="Nelder-Mead",
                      options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-5})
    opt_w = np.abs(result.x)
    opt_w /= opt_w.sum()
    opt_auc = -result.fun
    log.info("Optimised soundscape AUC: %.5f", opt_auc)

    # Report top models by weight
    ranked = sorted(zip(opt_w, names), reverse=True)
    log.info("Top models by soundscape-optimised weight:")
    for w, n in ranked[:10]:
        log.info("  %.4f  %s", w, n)

    out = {
        "metric": "soundscape_macro_auc",
        "equal_weight_auc": round(eq_auc, 6),
        "optimised_auc": round(opt_auc, 6),
        "models": [{"name": n, "weight": round(float(w), 6)}
                   for w, n in sorted(zip(opt_w, names), reverse=True)],
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    log.info("Saved to %s", args.output)

if __name__ == "__main__":
    main()
