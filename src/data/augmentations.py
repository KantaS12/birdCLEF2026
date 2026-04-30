"""Audio and spectrogram augmentations for BirdCLEF+ 2026."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T

class AudioAugmenter(nn.Module):
    """Time-domain augmentations applied to raw waveform."""

    def __init__(
        self,
        sample_rate: int = 32000,
        time_shift_max: float = 0.1,
        noise_snr_db: float = 20.0,
        volume_range: tuple[float, float] = (0.8, 1.2),
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.time_shift_max = time_shift_max
        self.noise_snr_db = noise_snr_db
        self.volume_range = volume_range
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to waveform (T,)."""
        if torch.rand(1).item() < self.p:
            max_shift = int(self.time_shift_max * self.sample_rate)
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            waveform = torch.roll(waveform, shift)

        if torch.rand(1).item() < self.p:
            signal_power = waveform.pow(2).mean()
            noise_power = signal_power / (10 ** (self.noise_snr_db / 10))
            noise = torch.randn_like(waveform) * noise_power.sqrt()
            waveform = waveform + noise

        if torch.rand(1).item() < self.p:
            lo, hi = self.volume_range
            gain = lo + torch.rand(1).item() * (hi - lo)
            waveform = waveform * gain

        return waveform

class SpecAugmenter(nn.Module):
    """SpecAugment: frequency and time masking on mel-spectrograms."""

    def __init__(
        self,
        freq_mask_max: int = 8,
        time_mask_max: int = 40,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ) -> None:
        super().__init__()
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_max)
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_max)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply masks to spectrogram (B, 1, F, T) or (1, F, T)."""
        for _ in range(self.n_freq_masks):
            spec = self.freq_masking(spec)
        for _ in range(self.n_time_masks):
            spec = self.time_masking(spec)
        return spec

def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply mixup to a batch of spectrograms and soft labels.

    Args:
        x: Spectrogram batch (B, C, F, T).
        y: Label batch (B, n_classes).
        alpha: Beta distribution parameter.

    Returns:
        Mixed (x, y) tensors.
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    batch_size = x.shape[0]
    idx = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = lam * y + (1 - lam) * y[idx]
    return mixed_x, mixed_y

class BackgroundNoiseMixer(nn.Module):
    """Mix soundscape background noise into log1p-normalized mel spectrograms.

    The noise bank HDF5 must contain LINEAR-POWER (pre-log1p) spectrograms,
    built by ``prepare_data.py --noise-bank``. SNR mixing is performed in the
    linear-power domain to keep the SNR formula mathematically correct.

    Args:
        noise_h5: Path to noise bank HDF5.
        snr_min_db: Minimum SNR in dB (higher = less noise).
        snr_max_db: Maximum SNR in dB.
        p: Probability of applying the augmentation.
    """

    def __init__(
        self,
        noise_h5: str,
        snr_min_db: float = 6.0,
        snr_max_db: float = 20.0,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.noise_h5 = noise_h5
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db
        self.p = p
        self._keys: list[str] | None = None

    def _get_keys(self) -> list[str]:
        if self._keys is None:
            import h5py
            with h5py.File(self.noise_h5, "r") as hf:
                self._keys = list(hf["spectrograms"].keys())
        return self._keys

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: (1, F, T) log1p-normalized mel spectrogram.
        Returns:
            Noise-mixed spectrogram of same shape, re-normalized.
        """
        if torch.rand(1).item() >= self.p:
            return spec

        import h5py
        keys = self._get_keys()
        key = keys[torch.randint(len(keys), (1,)).item()]

        with h5py.File(self.noise_h5, "r") as hf:
            noise = torch.from_numpy(hf["spectrograms"][key][:].copy())

        # Guard: skip this sample if noise contains NaN/inf (corrupted soundscape)
        if not torch.all(torch.isfinite(noise)):
            return spec

        # Match frequency dimension via interpolation if needed
        if noise.shape[-2] != spec.shape[-2]:
            noise = torch.nn.functional.interpolate(
                noise.unsqueeze(0), size=(spec.shape[-2], noise.shape[-1]), mode="bilinear", align_corners=False
            ).squeeze(0)

        # Trim or pad noise to match spec time dimension
        T = spec.shape[-1]
        if noise.shape[-1] > T:
            start = torch.randint(0, noise.shape[-1] - T + 1, (1,)).item()
            noise = noise[..., start:start + T]
        elif noise.shape[-1] < T:
            noise = torch.nn.functional.pad(noise, (0, T - noise.shape[-1]))

        # Convert input from log1p-normalized to linear power
        foreground = torch.expm1(spec.clamp(min=0))

        signal_power = foreground.pow(2).mean().clamp(min=1e-10)
        noise_power = noise.pow(2).mean().clamp(min=1e-10)

        # Scale noise to target SNR: signal_power / noise_scaled_power = 10^(snr_db/10)
        snr_db = self.snr_min_db + torch.rand(1).item() * (self.snr_max_db - self.snr_min_db)
        noise_scaled = noise * (signal_power / noise_power).sqrt() / (10 ** (snr_db / 20))

        mixed = (foreground + noise_scaled).clamp(min=0)
        mixed = torch.log1p(mixed)

        # Re-normalize to zero mean, unit variance
        mean = mixed.mean()
        std = mixed.std() + 1e-6
        return (mixed - mean) / std
