import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from typing import Optional


class WarmupCosineLR(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 base_lr: float,
                 total_steps: int,
                 start_lr: float = 0,
                 end_lr: float = 0,
                 warmup_steps: int = 0,
                 last_epoch: int = -1):

        self.base_lr = base_lr
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            factor = self.start_lr + ((self.base_lr - self.start_lr) * (self.last_epoch + 1) / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            factor = self.end_lr + 0.5 * (self.base_lr - self.end_lr) * (1 + math.cos(math.pi * progress))

        factor /= self.base_lr
        return [base_lr * factor for base_lr in self.base_lrs]
