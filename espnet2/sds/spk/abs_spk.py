from abc import ABC, abstractmethod

import torch


class AbsSPK(torch.nn.Module, ABC):
    @abstractmethod
    def warmup(self):
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        xs_pad,
    ):
        raise NotImplementedError
