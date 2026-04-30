"""Cache pseudo-labels CSV into an HDF5 file with the same schema as train_config_a.h5.

Reads pseudo_v1.csv (row_id, species_0..N-1) + the corresponding soundscape audio
to extract mel-spectrograms, then writes a pseudo_v1.h5 that PseudoLabelDataset
can read directly.

Usage:
    python scripts/cache_pseudolabels.py \
        --pseudo-csv data/pseudo_labels/pseudo_v1.csv \
        --soundscapes-dir data/test_soundscapes \
        --train-h5 data/cache/train_config_a.h5 \
        --output data/cache/pseudo_v1.h5

Validation:
    Raises ValueError if n_classes in pseudo CSV != n_classes in train_config_a.h5.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import h5py
import numpy as np
import pandas as pd
import torch
import torchaudio

from src.data.preprocessing import CONFIG_A, CONFIG_B, compute_melspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 32000
WINDOW_SAMPLES = int(5.0 * SAMPLE_RATE)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache pseudo-labels to HDF5.")
    p.add_argument("--pseudo-csv", required=True, type=Path)
    p.add_argument("--soundscapes-dir", required=True, type=Path)
    p.add_argument("--train-h5", required=True, type=Path,
                   help="Reference HDF5 to validate n_classes consistency.")
    p.add_argument("--output", type=Path, default=Path("data/cache/pseudo_v1.h5"))
    p.add_argument(
        "--mel-config",
        choices=["a", "b"],
        default="a",
        help="Mel-spectrogram config to use: 'a' (128 mel, hop=320) or 'b' (64 mel, hop=160).",
    )
    return p.parse_args()

def _validate_n_classes(pseudo_n_classes: int, train_h5_path: Path) -> None:
    """Raise ValueError if n_classes does not match the reference HDF5."""
    with h5py.File(train_h5_path, "r") as hf:
        train_n_classes = int(hf.attrs["n_classes"])
    if pseudo_n_classes != train_n_classes:
        raise ValueError(
            f"n_classes mismatch: pseudo CSV has {pseudo_n_classes} species columns "
            f"but {train_h5_path} has n_classes={train_n_classes}. "
            "Make sure generate_pseudolabels.py and train_config_a.h5 use the same label space."
        )

def _load_window(audio_path: Path, start_sample: int) -> torch.Tensor:
    """Load a 5-second window from an audio file and return a mono waveform tensor."""
    waveform, sr = torchaudio.load(
        str(audio_path),
        frame_offset=start_sample,
        num_frames=WINDOW_SAMPLES,
    )
    waveform = waveform.mean(0)  # mono, shape (T,)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    # Pad if last window is short
    if waveform.shape[0] < WINDOW_SAMPLES:
        pad = torch.zeros(WINDOW_SAMPLES - waveform.shape[0])
        waveform = torch.cat([waveform, pad])
    return waveform

def main() -> None:
    args = parse_args()

    mel_cfg = CONFIG_A if args.mel_config == "a" else CONFIG_B
    logger.info("Using mel config %s: %s", args.mel_config.upper(), mel_cfg)

    df = pd.read_csv(args.pseudo_csv)
    logger.info("Read %d pseudo-label rows from %s", len(df), args.pseudo_csv)

    species_cols = [c for c in df.columns if c != "row_id"]
    pseudo_n_classes = len(species_cols)
    logger.info("Pseudo CSV has %d species columns.", pseudo_n_classes)

    _validate_n_classes(pseudo_n_classes, args.train_h5)
    logger.info("n_classes validation passed: %d", pseudo_n_classes)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.output, "w") as hf:
        hf.attrs["n_classes"] = pseudo_n_classes
        specs_grp = hf.create_group("spectrograms")
        labels_grp = hf.create_group("labels")

        for i, row in df.iterrows():
            row_id: str = row["row_id"]
            # row_id format: "{soundscape_stem}_{start_sample}"
            parts = row_id.rsplit("_", 1)
            if len(parts) != 2:
                logger.warning("Skipping malformed row_id: %s", row_id)
                continue
            stem, start_str = parts
            start_sample = int(start_str)

            # Find the audio file (try .ogg first, then .wav)
            audio_path: Path | None = None
            for ext in (".ogg", ".wav", ".flac"):
                candidate = args.soundscapes_dir / f"{stem}{ext}"
                if candidate.exists():
                    audio_path = candidate
                    break
            if audio_path is None:
                logger.warning("Audio not found for row_id=%s, skipping.", row_id)
                continue

            waveform = _load_window(audio_path, start_sample)
            spec = compute_melspec(waveform, mel_cfg)  # (1, n_mels, T)

            label_vec = row[species_cols].to_numpy(dtype=np.float32)

            specs_grp.create_dataset(row_id, data=spec.numpy())
            labels_grp.create_dataset(row_id, data=label_vec)

            if (i + 1) % 100 == 0:
                logger.info("Cached %d/%d entries.", i + 1, len(df))

    logger.info("Wrote pseudo-label HDF5 to %s", args.output)

if __name__ == "__main__":
    main()
