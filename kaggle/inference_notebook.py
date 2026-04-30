"""BirdCLEF+ 2026 — Kaggle Inference Notebook (ssval per-class blend).

CPU-only OpenVINO FP16 ensemble. Reads test soundscapes, runs N souped models
on overlapping 5-sec windows (2.5s slide), aggregates to 12 non-overlapping
output bins per soundscape, and writes a submission.csv.

Soundscape-validated weights (NOT clean-clip OOF — those are anti-correlated
with LB). When a `per_class_weights.npy` + `per_class_weights.json` pair is
present alongside the model dir, per-class blending replaces global weights.

Estimated runtime: ~45 min for 700 soundscapes (5 models × 24 sliding windows).

Usage locally:
    python kaggle/inference_notebook.py \
        --test-dir /path/to/test_soundscapes \
        --sample-sub /path/to/sample_submission.csv \
        --models-dir exports/souped_ssval_aug05 \
        --output submission.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as F_audio
import torchaudio.transforms as T_audio

CFG_A = dict(n_mels=128, hop_length=320, n_fft=1024, fmin=50.0, fmax=14000.0, sr=32000)
CFG_B = dict(n_mels=64,  hop_length=160, n_fft=1024, fmin=20.0, fmax=16000.0, sr=32000)

WINDOW_SEC = 5.0
TARGET_SR = 32000
SLIDE_HOP_SEC = 2.5

# ssval_aug05 6-model pool. Top-5 deployed; effv2s_cfgb dropped (weight=0).
MODELS = [
    ("nfnet_cfgb",   CFG_B),
    ("effv2s_cfga",  CFG_A),
    ("nfnet_cfga",   CFG_A),
    ("effb0_cfga",   CFG_A),
    ("effb0_cfgb",   CFG_B),
]
# Soundscape-validated GLOBAL weights (fallback when no per-class file present).
MODEL_WEIGHTS: dict[str, float] = {
    "nfnet_cfgb":   0.328,
    "effv2s_cfga":  0.292,
    "nfnet_cfga":   0.222,
    "effb0_cfga":   0.109,
    "effb0_cfgb":   0.049,
}
ENV_DIM = 4

# Per-class weights file names (looked up under models_dir, optional).
PER_CLASS_NPY_NAME = "per_class_weights.npy"
PER_CLASS_JSON_NAME = "per_class_weights.json"

def load_audio(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        audio = F_audio.resample(torch.from_numpy(audio), sr, target_sr).numpy().astype(np.float32)
    return audio

def split_windows(audio: np.ndarray, sr: int, window_sec: float, hop_sec: float | None = None):
    ws = int(sr * window_sec)
    hs = int(sr * (hop_sec if hop_sec is not None else window_sec))
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
    waveform = torch.from_numpy(audio)
    transform = T_audio.MelSpectrogram(
        sample_rate=cfg["sr"], n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"], n_mels=cfg["n_mels"],
        f_min=cfg["fmin"], f_max=cfg["fmax"], power=2.0,
    )
    mel = transform(waveform)
    log_mel = torch.log1p(mel).numpy().astype(np.float32)
    mu = log_mel.mean()
    sigma = log_mel.std() + 1e-6
    log_mel = (log_mel - mu) / sigma
    return log_mel[np.newaxis]

class OVModel:
    def __init__(self, xml_path: str | Path) -> None:
        import openvino as ov
        core = ov.Core()
        core.set_property("CPU", {"INFERENCE_NUM_THREADS": "4"})
        model = core.read_model(str(xml_path))
        self.compiled = core.compile_model(model, "CPU")
        self.infer_req = self.compiled.create_infer_request()

    def predict(self, spec: np.ndarray, env: np.ndarray) -> np.ndarray:
        self.infer_req.infer({"spec": spec, "env": env})
        return self.infer_req.get_output_tensor(0).data.copy()

def build_class_weight_matrix(loaded_names: list[str], n_classes: int,
                              models_dir: Path) -> np.ndarray:
    """Return an (M, n_classes) float32 matrix, per-class summing to 1 across M."""
    npy_path = Path(models_dir) / PER_CLASS_NPY_NAME
    json_path = Path(models_dir) / PER_CLASS_JSON_NAME
    if npy_path.exists() and json_path.exists():
        meta = json.loads(Path(json_path).read_text())
        avail = list(meta.get("model_names", []))
        avail_short = [n.rsplit("/", 1)[-1] for n in avail]
        full = np.load(npy_path).astype(np.float32)
        if full.shape[1] != n_classes:
            print(f"  per-class npy class count {full.shape[1]} != {n_classes}; ignoring")
        else:
            rows = []
            for ln in loaded_names:
                if ln in avail_short:
                    rows.append(avail_short.index(ln))
                else:
                    rows = None
                    break
            if rows is not None:
                W = full[rows]
                col_sums = W.sum(axis=0, keepdims=True)
                col_sums = np.where(col_sums < 1e-12, 1.0, col_sums)
                W = W / col_sums
                print(f"  Using per-class weights from {npy_path.name}")
                return W
            print(f"  per-class npy missing rows for {loaded_names}; falling back to global")
    M = len(loaded_names)
    W = np.zeros((M, n_classes), dtype=np.float32)
    total = sum(MODEL_WEIGHTS.get(n, 1.0) for n in loaded_names)
    for m, n in enumerate(loaded_names):
        W[m] = MODEL_WEIGHTS.get(n, 1.0) / max(total, 1e-12)
    return W

def predict_soundscape(audio: np.ndarray,
                       ov_models: list[tuple[OVModel, dict]],
                       model_class_weights: np.ndarray,
                       n_classes: int):
    """ov_models: list of (OVModel, cfg).  model_class_weights: (M, n_classes)."""
    window_samples = int(WINDOW_SEC * TARGET_SR)
    n_output = int(np.ceil(len(audio) / window_samples))
    end_times = [str(int((i + 1) * WINDOW_SEC)) for i in range(n_output)]

    slide_windows, slide_starts = split_windows(audio, TARGET_SR, WINDOW_SEC, SLIDE_HOP_SEC)
    n_slide = len(slide_windows)

    win_logits = np.zeros((n_slide, n_classes), dtype=np.float64)
    env = np.zeros((1, ENV_DIM), dtype=np.float32)
    for m, (ov_model, mel_cfg) in enumerate(ov_models):
        wvec = model_class_weights[m]
        for i, window in enumerate(slide_windows):
            spec = compute_melspec(window, mel_cfg)
            logits = ov_model.predict(spec[np.newaxis], env)
            win_logits[i] += wvec * logits[0]

    acc = np.zeros((n_output, n_classes), dtype=np.float64)
    counts = np.zeros(n_output, dtype=np.int32)
    for i, start_samples in enumerate(slide_starts):
        start_sec = start_samples / TARGET_SR
        end_sec = start_sec + WINDOW_SEC
        b_min = max(0, int(start_sec / WINDOW_SEC))
        b_max = min(n_output - 1, int((end_sec - 1e-9) / WINDOW_SEC))
        for b in range(b_min, b_max + 1):
            acc[b] += win_logits[i]
            counts[b] += 1

    counts = np.maximum(counts, 1)
    avg_logits = acc / counts[:, np.newaxis]
    probs = 1.0 / (1.0 + np.exp(-avg_logits))
    return end_times, probs.astype(np.float32)

def run_inference(test_dir: Path, sample_sub: Path,
                  models_dir: Path, output: Path) -> None:
    print("Loading sample submission…")
    sub = pd.read_csv(str(sample_sub))
    species_cols = [c for c in sub.columns if c != "row_id"]
    n_classes = len(species_cols)
    print(f"  {n_classes} species, {len(sub)} rows in sample submission")

    print("Loading OpenVINO models…")
    t_load = time.time()
    ov_models: list[tuple[OVModel, dict]] = []
    loaded_names: list[str] = []
    for name, mel_cfg in MODELS:
        xml_path = models_dir / name / "model_fp16" / "model.xml"
        if not xml_path.exists():
            print(f"  WARNING: {xml_path} not found — skipping")
            continue
        print(f"  Loading {name}…")
        ov_models.append((OVModel(xml_path), mel_cfg))
        loaded_names.append(name)
    if not ov_models:
        raise RuntimeError("No models loaded — check models_dir path")
    print(f"  Loaded {len(ov_models)} models in {time.time() - t_load:.1f}s")

    model_class_weights = build_class_weight_matrix(loaded_names, n_classes, models_dir)

    test_files = sorted(p for p in test_dir.glob("*.ogg") if p.stem != "readme")
    if not test_files:
        test_files = sorted(test_dir.glob("*.wav"))
    print(f"Found {len(test_files)} test soundscapes")

    rows: list[dict] = []
    t_start = time.time()
    for i, audio_path in enumerate(test_files):
        t0 = time.time()
        audio = load_audio(audio_path)
        soundscape_id = audio_path.stem
        end_times, probs = predict_soundscape(audio, ov_models, model_class_weights, n_classes)
        for et, prob_vec in zip(end_times, probs):
            row = {"row_id": f"{soundscape_id}_{et}"}
            for j, col in enumerate(species_cols):
                row[col] = float(prob_vec[j])
            rows.append(row)
        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        eta = total_elapsed / (i + 1) * (len(test_files) - i - 1)
        print(f"  [{i+1}/{len(test_files)}] {soundscape_id} {elapsed:.1f}s  ETA {eta/60:.1f}min")

    print(f"Inference complete in {(time.time() - t_start)/60:.1f} min")

    if rows:
        result = pd.DataFrame(rows).set_index("row_id")
    else:
        print("WARNING: No soundscapes processed — returning sample submission as placeholder")
        sub.to_csv(str(output), index=False)
        return

    sub_indexed = sub.set_index("row_id")
    result = result.reindex(index=sub_indexed.index, columns=sub_indexed.columns)
    result = result.fillna(1.0 / n_classes).reset_index()
    result.to_csv(str(output), index=False)
    print(f"Saved submission: {output}  ({len(result)} rows)")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BirdCLEF+ 2026 inference")
    parser.add_argument("--test-dir", type=Path,
        default=Path("/kaggle/input/birdclef-2026/test_soundscapes"))
    parser.add_argument("--sample-sub", type=Path,
        default=Path("/kaggle/input/birdclef-2026/sample_submission.csv"))
    parser.add_argument("--models-dir", type=Path,
        default=Path("/kaggle/input/birdclef2026-exports"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    run_inference(test_dir=args.test_dir, sample_sub=args.sample_sub,
                  models_dir=args.models_dir, output=args.output)
