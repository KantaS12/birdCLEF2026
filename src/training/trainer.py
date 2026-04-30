"""Training loop for BirdCLEF+ 2026."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.transforms as T_audio
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.utils.metrics import macro_auc
from src.utils.logger import CSVLogger

logger = logging.getLogger(__name__)

class Trainer:
    """Manages training loop, validation, checkpointing, and logging."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: nn.Module,
        output_dir: Path,
        device: torch.device,
        max_grad_norm: float = 5.0,
        use_amp: bool = True,
        frame_loss_weight: float = 0.3,
        early_stopping_patience: int = 5,
        soundscape_dir: Optional[Path] = None,
        soundscape_labels_csv: Optional[Path] = None,
        sample_sub_csv: Optional[Path] = None,
        mel_cfg: Optional[dict] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.output_dir = output_dir
        self.device = device
        self.max_grad_norm = max_grad_norm
        # bfloat16 has float32 range — no overflow, no GradScaler needed.
        # float16 overflows on large activations (pretrained BN stats vs mel spec data).
        self.amp_dtype = torch.bfloat16 if use_amp else None
        self.scaler = GradScaler(device=device.type, enabled=False)  # not needed with bf16
        self.use_amp = use_amp
        self.frame_loss_weight = frame_loss_weight
        self.clip_loss_weight = 1.0 - frame_loss_weight
        self.early_stopping_patience = early_stopping_patience
        output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_logger = CSVLogger(output_dir / "metrics.csv")
        self.best_auc: float = 0.0
        self._no_improve_count: int = 0
        self._best_oof_preds: Optional[np.ndarray] = None
        self._best_oof_targets: Optional[np.ndarray] = None

        # Soundscape validation (Pantanal-domain, not clean-clip OOF)
        self.soundscape_enabled = bool(soundscape_dir and soundscape_labels_csv and sample_sub_csv and mel_cfg)
        if self.soundscape_enabled:
            self._init_soundscape_val(Path(soundscape_dir), Path(soundscape_labels_csv),
                                      Path(sample_sub_csv), mel_cfg)
        self.best_soundscape_auc: float = 0.0

    def _init_soundscape_val(self, ss_dir: Path, labels_csv: Path,
                             sample_sub: Path, mel_cfg: dict) -> None:
        """Pre-load soundscape validation data once at start of training."""
        logger.info("Setting up soundscape validation (mel cfg: %s)", mel_cfg)
        self.ss_mel_cfg = mel_cfg
        TARGET_SR = mel_cfg.get("sr", 32000)
        WINDOW_SEC = 5.0

        sub = pd.read_csv(str(sample_sub))
        species_cols = [c for c in sub.columns if c != "row_id"]
        self.ss_species_idx = {s: i for i, s in enumerate(species_cols)}
        self.ss_n_classes = len(species_cols)

        label_df = pd.read_csv(str(labels_csv))
        files = sorted(label_df["filename"].unique())

        # Pre-load all audio + build window tensors and target matrix
        windows_list, targets_list = [], []
        ws = int(TARGET_SR * WINDOW_SEC)
        for fname in files:
            audio_path = ss_dir / fname
            if not audio_path.exists():
                continue
            audio, sr = sf.read(str(audio_path), always_2d=True)
            audio = audio.mean(axis=1).astype(np.float32)
            if sr != TARGET_SR:
                import torchaudio.functional as F_audio
                audio = F_audio.resample(torch.from_numpy(audio), sr, TARGET_SR).numpy().astype(np.float32)

            file_rows = label_df[label_df["filename"] == fname].sort_values("start")
            for i in range(int(np.ceil(len(audio) / ws))):
                chunk = audio[i*ws:(i+1)*ws]
                if len(chunk) < ws:
                    chunk = np.pad(chunk, (0, ws - len(chunk)))
                windows_list.append(chunk)
                label_vec = np.zeros(self.ss_n_classes, dtype=np.float32)
                if i < len(file_rows):
                    for sp in str(file_rows.iloc[i]["primary_label"]).split(";"):
                        sp = sp.strip()
                        if sp in self.ss_species_idx:
                            label_vec[self.ss_species_idx[sp]] = 1.0
                targets_list.append(label_vec)

        self.ss_windows = torch.from_numpy(np.stack(windows_list)).to(self.device)
        self.ss_targets = np.stack(targets_list)
        self.ss_mel_transform = T_audio.MelSpectrogram(
            sample_rate=mel_cfg["sr"], n_fft=mel_cfg["n_fft"],
            hop_length=mel_cfg["hop_length"], n_mels=mel_cfg["n_mels"],
            f_min=mel_cfg["fmin"], f_max=mel_cfg["fmax"], power=2.0,
        ).to(self.device)
        logger.info("Soundscape val: %d windows, %d classes (%d with positives)",
                    len(self.ss_windows), self.ss_n_classes, int((self.ss_targets.sum(0) > 0).sum()))

    @torch.no_grad()
    def _validate_soundscapes(self, batch_size: int = 64) -> float:
        """Compute macro-AUC on labeled Pantanal soundscapes."""
        self.model.eval()
        all_logits = []
        env = torch.zeros((1, 4), dtype=torch.float32, device=self.device)
        for i in range(0, len(self.ss_windows), batch_size):
            chunk = self.ss_windows[i:i+batch_size]
            mel = self.ss_mel_transform(chunk)
            log_mel = torch.log1p(mel)
            mu = log_mel.mean(dim=(1, 2), keepdim=True)
            sigma = log_mel.std(dim=(1, 2), keepdim=True) + 1e-6
            spec = ((log_mel - mu) / sigma).unsqueeze(1)
            env_b = env.expand(spec.shape[0], -1)
            with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                out = self.model(spec, env=env_b if env_b.shape[-1] > 0 else None)
            all_logits.append(torch.sigmoid(out["clip_logits"].float()).cpu().numpy())
        preds = np.concatenate(all_logits, axis=0)
        aucs = []
        for j in range(self.ss_n_classes):
            if self.ss_targets[:, j].sum() > 0:
                try:
                    aucs.append(roc_auc_score(self.ss_targets[:, j], preds[:, j]))
                except Exception:
                    pass
        return float(np.mean(aucs)) if aucs else 0.0

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)
        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch} train")):
            specs, envs, labels = batch
            specs = specs.to(self.device)
            envs = envs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                out = self.model(specs, env=envs if envs.shape[-1] > 0 else None)
                clip_loss = self.loss_fn(out["clip_logits"], labels)
                frame_loss = self.loss_fn(out["frame_logits"].mean(dim=1), labels)
                loss = self.clip_loss_weight * clip_loss + self.frame_loss_weight * frame_loss
            if torch.isnan(loss) or torch.isinf(loss):
                spec_ok = torch.all(torch.isfinite(specs)).item()
                env_ok = torch.all(torch.isfinite(envs)).item()
                clip_ok = torch.all(torch.isfinite(out["clip_logits"])).item()
                frame_ok = torch.all(torch.isfinite(out["frame_logits"])).item()
                logger.error(
                    "NaN/inf loss at epoch %d step %d — specs_finite=%s envs_finite=%s "
                    "clip_logits_finite=%s frame_logits_finite=%s. Skipping batch.",
                    epoch, step, spec_ok, env_ok, clip_ok, frame_ok,
                )
                self.optimizer.zero_grad()
                continue
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += loss.item()
        return total_loss / n_batches

    @torch.no_grad()
    def _validate(self) -> tuple[float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0
        for batch in tqdm(self.val_loader, desc="Validating"):
            specs, envs, labels = batch
            specs = specs.to(self.device)
            envs = envs.to(self.device)
            labels = labels.to(self.device)
            with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                out = self.model(specs, env=envs if envs.shape[-1] > 0 else None)
                loss = self.loss_fn(out["clip_logits"], labels)
            total_loss += loss.item()
            all_logits.append(torch.sigmoid(out["clip_logits"].float()).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        preds = np.concatenate(all_logits, axis=0)
        targets = np.concatenate(all_labels, axis=0)
        auc = macro_auc(targets, preds)
        return total_loss / len(self.val_loader), auc, preds, targets

    def save_checkpoint(self, epoch: int, auc: float, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "auc": auc,
        }
        torch.save(state, self.output_dir / "last.pt")
        if is_best:
            torch.save(state, self.output_dir / "best.pt")
            logger.info("New best AUC: %.4f — saved best.pt", auc)

    def fit(self, n_epochs: int) -> None:
        """Run training loop with early stopping.

        If soundscape validation is enabled, uses soundscape AUC for best-model
        selection and early stopping (the metric Kaggle actually evaluates).
        Otherwise falls back to clean-clip val AUC.
        """
        for epoch in range(1, n_epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss, val_auc, preds, targets = self._validate()

            ss_auc = self._validate_soundscapes() if self.soundscape_enabled else None
            selection_metric = ss_auc if self.soundscape_enabled else val_auc
            best_field = self.best_soundscape_auc if self.soundscape_enabled else self.best_auc

            is_best = selection_metric > best_field
            if is_best:
                if self.soundscape_enabled:
                    self.best_soundscape_auc = ss_auc
                self.best_auc = val_auc
                self._no_improve_count = 0
                self._best_oof_preds = preds
                self._best_oof_targets = targets
            else:
                self._no_improve_count += 1
            self.save_checkpoint(epoch, val_auc, is_best)
            metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_auc": round(val_auc, 5),
                "lr": self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]["lr"],
            }
            if self.soundscape_enabled:
                metrics["ss_auc"] = round(ss_auc, 5)
            self.csv_logger.log(metrics)
            ss_str = f" ss_auc={ss_auc:.4f}" if self.soundscape_enabled else ""
            logger.info("Epoch %d | train_loss=%.4f val_loss=%.4f val_auc=%.4f%s",
                        epoch, train_loss, val_loss, val_auc, ss_str)
            if self._no_improve_count >= self.early_stopping_patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, self.early_stopping_patience)
                break
        # Save OOF predictions from best epoch
        if self._best_oof_preds is not None:
            np.savez(self.output_dir / "oof.npz",
                     preds=self._best_oof_preds, targets=self._best_oof_targets)
            logger.info("OOF predictions saved to %s/oof.npz", self.output_dir)
        self.csv_logger.close()
