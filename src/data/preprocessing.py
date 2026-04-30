"""Audio → mel-spectrogram conversion and HDF5 caching."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as T
from tqdm import tqdm

from src.utils.audio import load_audio, pad_or_crop

logger = logging.getLogger(__name__)

SAMPLE_RATE = 32000
WINDOW_SAMPLES = int(5.0 * SAMPLE_RATE)  # 5-second clips

@dataclass
class MelConfig:
    """Configuration for mel-spectrogram computation."""
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 320
    fmin: float = 50.0
    fmax: float = 14000.0
    sample_rate: int = SAMPLE_RATE
    name: str = "config_a"

CONFIG_A = MelConfig(
    n_mels=128, n_fft=1024, hop_length=320,
    fmin=50.0, fmax=14000.0, name="config_a"
)
CONFIG_B = MelConfig(
    n_mels=64, n_fft=1024, hop_length=160,
    fmin=20.0, fmax=16000.0, name="config_b"
)

def compute_melspec(waveform: torch.Tensor, config: MelConfig) -> torch.Tensor:
    """Compute normalized log-mel spectrogram from waveform.

    Args:
        waveform: 1D float tensor, shape (T,).
        config: Mel configuration.

    Returns:
        Spectrogram tensor, shape (1, n_mels, time_frames).
    """
    transform = T.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        f_min=config.fmin,
        f_max=config.fmax,
        power=2.0,
    )
    spec = transform(waveform)  # (n_mels, T)
    spec = torch.log1p(spec)
    # Normalize per clip
    mean = spec.mean()
    std = spec.std() + 1e-6
    spec = (spec - mean) / std
    return spec.unsqueeze(0)  # (1, n_mels, T)

def build_hdf5_cache(
    metadata_csv: Path,
    audio_dir: Path,
    output_path: Path,
    config: MelConfig = CONFIG_A,
    window_sec: float = 5.0,
) -> None:
    """Precompute mel-spectrograms and write to HDF5.

    HDF5 layout:
        /spectrograms/<row_id>  — float32 array (1, n_mels, time_frames)
        /labels/<row_id>        — int8 array (n_classes,)
        /metadata               — JSON-encoded DataFrame

    Args:
        metadata_csv: Path to train_metadata.csv.
        audio_dir: Root directory for audio files.
        output_path: Destination .h5 file.
        config: Mel configuration to use.
        window_sec: Length of audio window in seconds.
    """
    df = pd.read_csv(metadata_csv)
    window_samples = int(window_sec * config.sample_rate)

    species = sorted(df["primary_label"].unique())
    species_to_idx = {s: i for i, s in enumerate(species)}
    n_classes = len(species)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as hf:
        specs_grp = hf.create_group("spectrograms")
        labels_grp = hf.create_group("labels")
        hf.attrs["species"] = str(species)
        hf.attrs["n_classes"] = n_classes
        hf.attrs["mel_config"] = config.name

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Caching spectrograms"):
            row_id = str(row.get("filename", row.name))
            audio_path = audio_dir / row["filename"]

            try:
                waveform = load_audio(audio_path)
                waveform = pad_or_crop(waveform, window_samples)
                spec = compute_melspec(waveform, config)
            except Exception as e:
                logger.warning("Failed to process %s: %s", audio_path, e)
                continue

            specs_grp.create_dataset(row_id, data=spec.numpy(), dtype="float32")

            label_vec = np.zeros(n_classes, dtype=np.int8)
            primary = row.get("primary_label", "")
            if primary in species_to_idx:
                label_vec[species_to_idx[primary]] = 1
            secondary = str(row.get("secondary_labels", ""))
            for s in secondary.strip("[]'").split(","):
                s = s.strip().strip("'")
                if s in species_to_idx:
                    label_vec[species_to_idx[s]] = 1
            labels_grp.create_dataset(row_id, data=label_vec)
