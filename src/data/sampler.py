"""Balanced samplers for class imbalance in BirdCLEF."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler

from src.data.dataset import BirdDataset

class SqrtFrequencySampler(Sampler):
    """Sample indices with probability proportional to sqrt(class_frequency).

    Upweights rare classes, downweights dominant ones.
    """

    def __init__(self, dataset: BirdDataset, num_samples: int | None = None) -> None:
        if len(dataset) == 0:
            raise ValueError(
                f"Dataset is empty (0 samples). "
                f"Check that the HDF5 cache at '{dataset.h5_path}' was built correctly "
                f"and that the fold CSV keys match the HDF5 keys."
            )
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        counts = dataset.get_label_counts()  # (C,) — uses local handle, leaves _hf=None
        # Assign each sample weight = max sqrt(count) of its positive classes
        sqrt_counts = np.sqrt(counts + 1e-6)
        import h5py
        sample_weights = []
        # Use a local context manager — must NOT call dataset._get_hf() here because
        # DataLoader will fork workers immediately after this constructor returns, and
        # an open h5py handle inherited across fork causes corrupted/NaN reads.
        with h5py.File(dataset.h5_path, "r") as hf:
            for key in dataset.keys:
                label = hf["labels"][key][:]
                pos_classes = np.where(label > 0)[0]
                if len(pos_classes) == 0:
                    sample_weights.append(1.0)
                else:
                    sample_weights.append(float(np.max(sqrt_counts[pos_classes])))
        total = sum(sample_weights)
        self.weights = torch.tensor([w / total for w in sample_weights], dtype=torch.float64)

    @classmethod
    def from_concat_dataset(
        cls,
        concat_ds: "torch.utils.data.ConcatDataset",
        num_samples: int | None = None,
    ) -> "SqrtFrequencySampler":
        """Build a SqrtFrequencySampler from a ConcatDataset of BirdDataset / PseudoLabelDataset.

        Computes global class counts across all sub-datasets, then assigns per-sample
        weights using the same sqrt-frequency logic as __init__.
        """
        import h5py

        global_counts: np.ndarray | None = None
        for sub_ds in concat_ds.datasets:
            with h5py.File(sub_ds.h5_path, "r") as hf:
                for key in sub_ds.keys:
                    label = hf["labels"][key][:]
                    if global_counts is None:
                        global_counts = np.zeros(len(label), dtype=np.float64)
                    global_counts += (label > 0).astype(np.float64)

        if global_counts is None or global_counts.sum() == 0:
            raise ValueError("ConcatDataset is empty or has no positive labels.")

        sqrt_counts = np.sqrt(global_counts + 1e-6)

        sample_weights: list[float] = []
        for sub_ds in concat_ds.datasets:
            with h5py.File(sub_ds.h5_path, "r") as hf:
                for key in sub_ds.keys:
                    label = hf["labels"][key][:]
                    pos_classes = np.where(label > 0)[0]
                    if len(pos_classes) == 0:
                        sample_weights.append(1.0)
                    else:
                        sample_weights.append(float(np.max(sqrt_counts[pos_classes])))

        total = sum(sample_weights)
        n = num_samples if num_samples is not None else len(sample_weights)
        inst = cls.__new__(cls)
        inst.num_samples = n
        inst.weights = torch.tensor([w / total for w in sample_weights], dtype=torch.float64)
        return inst

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self) -> int:
        return self.num_samples
