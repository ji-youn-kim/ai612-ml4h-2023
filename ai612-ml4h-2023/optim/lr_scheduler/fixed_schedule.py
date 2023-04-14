# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, List
from argparse import Namespace

from . import LRScheduler, register_lr_scheduler

@register_lr_scheduler("fixed")
class FixedLRSchedule(LRScheduler):
    """Decay the LR on a fixed schedule."""
    
    def __init__(self, cfg: Namespace, optimizer, **kwargs):
        super().__init__(cfg, optimizer, **kwargs)

        self.lr = cfg.lr
        if cfg.warmup_updates > 0:
            self.warmup_factor = 1.0 / cfg.warmup_updates
        else:
            self.warmup_factor = 1

    def state_dict(self):
        return {"lr": self.lr}

    def load_state_dict(self, state_dict):
        if "lr" in state_dict:
            self.lr = state_dict["lr"]

    def get_next_lr(self, epoch):
        lrs = self.cfg.lr
        if self.cfg.force_anneal is None or epoch < self.cfg.force_anneal:
            # use fixed LR schedule
            next_lr = lrs[min(epoch - 1, len(lrs) - 1)]
        else:
            # anneal based on lr_shrink
            next_lr = lrs[-1] * self.cfg.lr_shrink ** (
                epoch + 1 - self.cfg.force_anneal
            )
        return next_lr

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        self.lr = self.get_next_lr(epoch)
        self.optimizer.set_lr(self.warmup_factor * self.lr)
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.cfg.warmup_updates > 0 and num_updates < self.cfg.warmup_updates:
            self.warmup_factor = (num_updates + 1) / float(self.cfg.warmup_updates)
            self.optimizer.set_lr(self.warmup_factor * self.lr)
        else:
            self.optimizer.set_lr(self.lr)
        return self.optimizer.get_lr()