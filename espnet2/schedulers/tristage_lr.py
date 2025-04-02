"""Warm up learning rate scheduler module."""

from typing import Union

import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import typechecked

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class TristageLR(_LRScheduler, AbsBatchStepScheduler):
    """The tri-stage lr scheduler

    refer to:
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/lr_scheduler/tri_stage_lr_scheduler.py

    max_steps = max_epoch * num_iters_per_epoch

    """

    @typechecked
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_steps: Union[int, float] = 25000,
        warmup_ratio: float = 0.1, 
        hold_ratio: float = 0.4, 
        decay_ratio: float = 0.5,
        init_lr_scale: float = 0.01, 
        final_lr_scale: float = 0.01, 
        last_epoch: int = -1,
    ):
        self.max_steps = max_steps
        assert (
            warmup_ratio > 0 and hold_ratio > 0 and decay_ratio > 0, 
            "The warmup_ratio, hold_ratio, and decay_ratio must be greater than 0."
        )
        assert (
            warmup_ratio + hold_ratio + decay_ratio == 1, 
            "The sum of warmup_ratio, hold_ratio, and decay_ratio must be 1."
        )
        self.warmup_steps = int(max_steps * warmup_ratio)
        assert self.warmup_steps > 0, "The warmup_steps must be greater than 0."
        self.hold_steps = int(max_steps * hold_ratio)
        self.decay_steps = int(max_steps * decay_ratio)
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        super().__init__(optimizer, last_epoch)
    
        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        # super().__init__(optimizer, last_epoch)
        

    def __repr__(self):
        express = f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"
        express += f"(hold_steps={self.hold_steps})"
        express += f"(decay_steps={self.decay_steps})"
        express += f"(init_lr_scale={self.init_lr_scale})"
        express += f"(final_lr_scale={self.final_lr_scale})"
        express += f"(decay_factor={self.decay_factor})"
        return express


    def get_lr(self):

        step_num = self.last_epoch + 1
        if step_num < self.warmup_steps:
            return [
                self.init_lr_scale * base_lr + 
                (base_lr - self.init_lr_scale * base_lr) /  self.warmup_steps * 
                step_num
                for base_lr in self.base_lrs
            ]
        elif step_num < self.warmup_steps + self.hold_steps:
            return [base_lr for base_lr in self.base_lrs]
        else:
            return [
                base_lr * math.exp(-self.decay_factor * (step_num - self.warmup_steps - self.hold_steps)) 
                for base_lr in self.base_lrs
            ]
