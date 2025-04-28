# Copyright 2023 Jee-weon Jung
# Apache 2.0

"""RawNet3 Encoder"""

import torch

from espnet2.asr.encoder.abs_encoder import AbsEncoder


class ProjectorEncoder(AbsEncoder):
    """
    This is used for implementing MMS lid, 
    it uses a projector prior to the pooling. 
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1024,
    ):
        super().__init__()
        self._output_size = output_size
        self.projector = torch.nn.Linear(input_size, output_size, bias=True)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor):
        x = self.projector(x) 
        return x.transpose(1, 2) # (B, D, T) -> (B, T, D)
