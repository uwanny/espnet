import torch

from espnet2.spk.projector.abs_projector import AbsProjector


class IdentityProjector(AbsProjector):
    """
    Used for implementing MMS lid, it doesn't use a projector after pooling. 
    """
    def __init__(self, input_size):
        super().__init__()
        self._output_size = input_size

    def output_size(self):
        return self._output_size

    def forward(self, x):
        return x
