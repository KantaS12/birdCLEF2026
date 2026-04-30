"""Learning rate scheduler: cosine annealing with linear warmup."""
from __future__ import annotations

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

class CosineWarmupScheduler(LRScheduler):
    """Linear warmup then cosine annealing.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.
        min_lr_ratio: Final LR as fraction of base LR.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            min_lr = base_lr * self.min_lr_ratio
            if step < self.warmup_steps:
                lr = base_lr * (step + 1) / max(1, self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs
