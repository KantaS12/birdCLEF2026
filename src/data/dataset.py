"""PyTorch Datasets for BirdCLEF+ 2026."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from src.data.preprocessing import MelConfig, CONFIG_A, compute_melspec
from src.utils.audio import load_audio, pad_or_crop

WINDOW_SAMPLES = 32000 * 5

class BirdDataset(Dataset):
    """Dataset backed by a precomputed HDF5 cache.

    Returns (spectrogram, env, label) triples.
    env is a float32 tensor of shape (4,) containing month_sin, month_cos,
    hour_sin, hour_cos. Zeros when no env_lookup provided.
    """

    def __init__(
        self,
        h5_path: Path | str,
        transform=None,
        augment=None,
        fold_csv: Optional[Path | str] = None,
        env_lookup: Optional[dict] = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.transform = transform
        self.augment = augment
        self.env_lookup = env_lookup or {}
        with h5py.File(self.h5_path, "r") as hf:
            all_keys = list(hf["spectrograms"].keys())
            # Determine n_classes from actual label dimension (more reliable than stale attrs)
            label_keys = list(hf["labels"].keys())
            if label_keys:
                self.n_classes: int = int(hf["labels"][label_keys[0]][:].shape[0])
            else:
                self.n_classes = int(hf.attrs["n_classes"])

        if fold_csv is not None:
            import pandas as pd
            # HDF5 keys use '__' where filenames use '/' (see prepare_data.py key encoding)
            allowed = set(
                n.replace("/", "__").replace("\\", "__")
                for n in pd.read_csv(fold_csv)["filename"].astype(str)
            )
            self.keys = [k for k in all_keys if k in allowed]
        else:
            self.keys = all_keys
        self._hf: Optional[h5py.File] = None

    def _get_hf(self) -> h5py.File:
        if self._hf is None:
            self._hf = h5py.File(self.h5_path, "r")
        return self._hf

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = self.keys[idx]
        hf = self._get_hf()
        spec = torch.from_numpy(hf["spectrograms"][key][:])
        label = torch.from_numpy(hf["labels"][key][:].astype(np.float32))
        if self.augment is not None:
            spec = self.augment(spec)
        if self.transform is not None:
            spec = self.transform(spec)
        # Sanitize AFTER augmentation — augmentations (e.g. BackgroundNoiseMixer)
        # can produce inf/NaN via power-domain overflow; clamp here so nothing
        # non-finite ever reaches the model or loss function.
        spec = torch.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
        env_vec = self.env_lookup.get(key, np.zeros(4, dtype=np.float32))
        env = torch.from_numpy(np.asarray(env_vec, dtype=np.float32))
        return spec, env, label

    def get_label_counts(self) -> np.ndarray:
        """Return per-class sample counts for sampler construction.

        Opens a temporary file handle so self._hf stays None — safe to fork
        DataLoader workers afterwards without sharing an h5py file object.
        """
        counts = np.zeros(self.n_classes, dtype=np.int64)
        with h5py.File(self.h5_path, "r") as hf:
            for key in self.keys:
                counts += hf["labels"][key][:].astype(np.int64)
        return counts

class SoundscapeDataset(Dataset):
    """Streams 5-second windows from a directory of soundscape audio files.

    Used for inference — no labels returned.
    """

    def __init__(
        self,
        audio_dir: Path | str,
        mel_config: MelConfig = CONFIG_A,
        window_sec: float = 5.0,
        step_sec: float = 5.0,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.mel_config = mel_config
        self.window_samples = int(window_sec * mel_config.sample_rate)
        self.step_samples = int(step_sec * mel_config.sample_rate)
        self._index = self._build_index()

    def _build_index(self) -> list[tuple[Path, int]]:
        """Build list of (audio_path, start_sample) pairs."""
        index = []
        for path in sorted(self.audio_dir.glob("*.ogg")):
            info = torchaudio.info(str(path))
            total_samples = info.num_frames
            start = 0
            while start + self.window_samples <= total_samples:
                index.append((path, start))
                start += self.step_samples
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, int]:
        path, start = self._index[idx]
        waveform, sr = torchaudio.load(str(path), frame_offset=start, num_frames=self.window_samples)
        waveform = waveform.mean(0)  # mono
        spec = compute_melspec(waveform, self.mel_config)
        return spec, path.stem, start
