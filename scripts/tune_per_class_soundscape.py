"""Per-class ensemble weight tuner on labeled Pantanal soundscapes.

Macro-AUC = mean(per-class AUC over classes with positives), and per-class AUC
is independent across classes. So the joint weight optimisation can be split
into K independent per-class subproblems, each with n_models parameters.

Outputs:
  - {output}                  JSON summary (global, per-class top weights, AUCs)
  - {output_npy}              (n_models, n_classes) float32 weight matrix.
                              Classes WITHOUT positives in the val set fall
                              back to a globally-tuned weight vector (which
                              dominates macro-AUC computation since those
                              classes contribute 0 weight to the metric — but
                              they might still appear in test, so we don't
                              want them random).
  - {cache}                   .npz with per-model logits (avoids re-inference
                              if you want to extend the pool later).

Usage:
    python scripts/tune_per_class_soundscape.py \
        --exports-dir exports/souped_ssval_aug05 \
        --exports-dir exports/souped_ssval_aug03 \
        --output soundscape_per_class_weights.json \
        --weights-npy soundscape_per_class_weights.npy \
        --cache soundscape_logits_cache.npz
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

def mel_cfg_for(name: str) -> dict:
    return CFG_B if (name.endswith("cfgb") or name.startswith("cnn14")) else CFG_A

def load_audio(path) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    if sr != TARGET_SR:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * TARGET_SR / sr)).astype(np.float32)
    return audio

def split_windows(audio: np.ndarray):
    ws = int(TARGET_SR * WINDOW_SEC); hs = int(TARGET_SR * SLIDE_HOP)
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

def build_targets(label_df, species_cols, files):
    species_idx = {s: i for i, s in enumerate(species_cols)}
    n_classes = len(species_cols)
    ws = int(TARGET_SR * WINDOW_SEC)
    targets = []
    for fname in files:
        audio_path = SS_DIR / fname
        if not audio_path.exists():
            continue
        audio = load_audio(audio_path)
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

def predict_model(xml_path, mel_cfg, files, n_classes):
    """Returns (n_output_bins, n_classes) sigmoid probabilities and
    a parallel (n_output_bins, n_classes) logit array (pre-sigmoid)."""
    ws = int(TARGET_SR * WINDOW_SEC)
    env = np.zeros((1, ENV_DIM), dtype=np.float32)
    model = OVModel(xml_path)
    all_logits = []
    for fname in files:
        audio_path = SS_DIR / fname
        if not audio_path.exists():
            continue
        audio = load_audio(audio_path)
        slide_wins, slide_starts = split_windows(audio)
        n_out = int(np.ceil(len(audio) / ws))
        acc = np.zeros((n_out, n_classes), dtype=np.float64)
        counts = np.zeros(n_out, dtype=np.int32)
        for win, ss in zip(slide_wins, slide_starts):
            spec = compute_melspec(win, mel_cfg)[np.newaxis]
            logits = model.predict(spec, env)[0]
            s = ss / TARGET_SR; e = s + WINDOW_SEC
            b_min = max(0, int(s / WINDOW_SEC))
            b_max = min(n_out - 1, int((e - 1e-9) / WINDOW_SEC))
            for b in range(b_min, b_max + 1):
                acc[b] += logits
                counts[b] += 1
        counts = np.maximum(counts, 1)
        all_logits.append((acc / counts[:, None]).astype(np.float32))
    return np.concatenate(all_logits, axis=0)

def macro_auc_global(weights, preds_list, targets):
    w = np.abs(weights); w = w / max(w.sum(), 1e-12)
    ensemble = sum(w[i] * preds_list[i] for i in range(len(preds_list)))
    aucs = []
    for j in range(targets.shape[1]):
        if targets[:, j].sum() > 0:
            try:
                aucs.append(roc_auc_score(targets[:, j], ensemble[:, j]))
            except Exception:
                pass
    return -float(np.mean(aucs)) if aucs else 0.0

def per_class_auc(weights, class_preds_stack, y):
    """class_preds_stack: (n_models, n_windows). y: (n_windows,)."""
    w = np.abs(weights); w = w / max(w.sum(), 1e-12)
    score = (w[:, None] * class_preds_stack).sum(0)
    try:
        return -float(roc_auc_score(y, score))
    except Exception:
        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports-dir", action="append", required=True)
    parser.add_argument("--output", default="soundscape_per_class_weights.json")
    parser.add_argument("--weights-npy", default="soundscape_per_class_weights.npy")
    parser.add_argument("--cache", default="soundscape_logits_cache.npz")
    parser.add_argument("--reuse-cache", action="store_true",
                        help="If cache exists with the same model list, skip inference.")
    args = parser.parse_args()

    label_df = pd.read_csv(LABELS_CSV)
    sub = pd.read_csv(SAMPLE_SUB)
    species_cols = [c for c in sub.columns if c != "row_id"]
    n_classes = len(species_cols)
    files = sorted(label_df["filename"].unique())
    log.info("SS files: %d  Species: %d  Label rows: %d", len(files), n_classes, len(label_df))

    models = []  # (name, xml, cfg)
    for exp_dir in args.exports_dir:
        exp = Path(exp_dir)
        for model_dir in sorted(exp.iterdir()):
            xml = model_dir / "model_fp16" / "model.xml"
            if xml.exists():
                models.append((f"{exp.name}/{model_dir.name}", xml, mel_cfg_for(model_dir.name)))
                log.info("  Found %s", models[-1][0])
    if not models:
        raise RuntimeError("No models found")
    log.info("Total models: %d", len(models))
    names = [m[0] for m in models]

    targets = build_targets(label_df, species_cols, files)
    log.info("Targets: %s  Pos classes: %d", targets.shape, (targets.sum(0) > 0).sum())

    # Cache: try reuse if model list matches
    cache_path = Path(args.cache)
    preds_list = None
    if args.reuse_cache and cache_path.exists():
        c = np.load(cache_path, allow_pickle=True)
        if list(c["names"]) == names and c["targets"].shape == targets.shape:
            preds_list = [c["preds"][i] for i in range(len(names))]
            log.info("Reused cached predictions")

    if preds_list is None:
        preds_list = []
        for name, xml, cfg in models:
            log.info("Running %s ...", name); t0 = time.time()
            preds = predict_model(xml, cfg, files, n_classes)[:targets.shape[0]]
            preds_list.append(preds)
            aucs = [roc_auc_score(targets[:, j], preds[:, j])
                    for j in range(n_classes) if targets[:, j].sum() > 0]
            log.info("  solo AUC %.5f  [%.0fs]", float(np.mean(aucs)), time.time() - t0)
        # save cache
        preds_arr = np.stack(preds_list, axis=0)  # (M, W, C)
        np.savez_compressed(cache_path, names=np.array(names), targets=targets, preds=preds_arr)
        log.info("Cached predictions to %s", cache_path)

    M = len(preds_list)
    W = targets.shape[0]
    C = n_classes

    # Equal-weight + global Nelder-Mead reference
    eq_auc = -macro_auc_global(np.ones(M), preds_list, targets)
    log.info("Equal-weight macro-AUC: %.5f", eq_auc)

    log.info("Global Nelder-Mead ...")
    res = minimize(macro_auc_global, np.ones(M) / M, args=(preds_list, targets),
                   method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-5})
    global_w = np.abs(res.x); global_w /= global_w.sum()
    global_auc = -res.fun
    log.info("Global optimised macro-AUC: %.5f", global_auc)

    # Stack preds for per-class optimisation: (M, W, C)
    preds_stack = np.stack(preds_list, axis=0)

    # Per-class optimisation
    weight_matrix = np.tile(global_w[:, None], (1, C))  # default = global
    per_class_aucs = np.zeros(C, dtype=np.float64)
    pos_classes = []
    for c in range(C):
        y = targets[:, c]
        if y.sum() == 0:
            per_class_aucs[c] = np.nan
            continue
        pos_classes.append(c)
        cps = preds_stack[:, :, c]  # (M, W)
        # Init from global as warm start
        r = minimize(per_class_auc, global_w, args=(cps, y),
                     method="Nelder-Mead",
                     options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-5})
        wc = np.abs(r.x); wc /= max(wc.sum(), 1e-12)
        weight_matrix[:, c] = wc
        per_class_aucs[c] = -r.fun

    pos_n = len(pos_classes)
    pc_macro = float(np.nanmean(per_class_aucs))
    log.info("Per-class optimised macro-AUC over %d pos classes: %.5f", pos_n, pc_macro)

    # Sanity: re-evaluate macro-AUC using the per-class weight matrix end-to-end
    ensemble = np.zeros((W, C), dtype=np.float64)
    for m in range(M):
        ensemble += weight_matrix[m][None, :] * preds_stack[m]
    aucs_check = [roc_auc_score(targets[:, j], ensemble[:, j])
                  for j in range(C) if targets[:, j].sum() > 0]
    log.info("End-to-end check macro-AUC: %.5f", float(np.mean(aucs_check)))

    # Per-class weight-vector entropy (how "different" from global)
    deltas = []
    for c in pos_classes:
        delta = float(np.linalg.norm(weight_matrix[:, c] - global_w))
        deltas.append(delta)
    log.info("L2 deviation from global weights — median %.4f  max %.4f",
             float(np.median(deltas)), float(np.max(deltas)))

    # Save
    np.save(args.weights_npy, weight_matrix.astype(np.float32))
    log.info("Wrote per-class weight matrix to %s  shape=%s",
             args.weights_npy, weight_matrix.shape)

    out = {
        "metric": "soundscape_macro_auc",
        "n_models": M,
        "n_classes": C,
        "n_pos_classes": pos_n,
        "equal_weight_auc": round(eq_auc, 6),
        "global_optimised_auc": round(global_auc, 6),
        "per_class_optimised_auc": round(pc_macro, 6),
        "model_names": names,
        "global_weights": [round(float(w), 6) for w in global_w],
        "weights_npy": args.weights_npy,
        "cache": str(cache_path),
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    log.info("Saved summary to %s", args.output)

if __name__ == "__main__":
    main()
