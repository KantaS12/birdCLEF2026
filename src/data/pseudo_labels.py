"""PseudoLabelDataset — reads pseudo-label HDF5 caches produced by cache_pseudolabels.py.

HDF5 schema (same as train_config_a.h5):
  spectrograms/<key>  float32 (1, n_mels, T)
  labels/<key>        float32 (n_classes,)
  attrs["n_classes"]  int
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class PseudoLabelDataset(Dataset):
    """Dataset backed by a pseudo-label HDF5 cache.

    Returns (spectrogram, env, label) triples — same interface as BirdDataset,
    so it can be combined with it via torch.utils.data.ConcatDataset.

    If target_shape=(n_mels, T) is given and the stored spec has a different
    shape, the spec is bilinearly interpolated to match.  This allows a single
    pseudo HDF5 (cached at config-A resolution) to be used with config-B folds.
    """

    def __init__(
        self,
        h5_path: Path | str,
        target_shape: Optional[tuple[int, int]] = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.target_shape = target_shape  # (n_mels, T) or None
        with h5py.File(self.h5_path, "r") as hf:
            self.keys: list[str] = sorted(hf["spectrograms"].keys())
            self._n_classes: int = int(hf.attrs["n_classes"])
        self._hf: Optional[h5py.File] = None  # open lazily per worker

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def _get_hf(self) -> h5py.File:
        if self._hf is None:
            self._hf = h5py.File(self.h5_path, "r")
        return self._hf

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = self.keys[idx]
        hf = self._get_hf()
        spec = torch.from_numpy(hf["spectrograms"][key][:])  # (1, n_mels, T)
        label = torch.from_numpy(hf["labels"][key][:].astype(np.float32))  # (n_classes,)
        env = torch.zeros(4, dtype=torch.float32)  # no env metadata for pseudo samples
        if self.target_shape is not None and tuple(spec.shape[1:]) != self.target_shape:
            spec = F.interpolate(
                spec.unsqueeze(0),  # (1, 1, n_mels, T)
                size=self.target_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # (1, n_mels, T)
        return spec, env, label
