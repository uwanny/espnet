import torch

from espnet2.spk.pooling.abs_pooling import AbsPooling


class MeanPooling(AbsPooling):
    """Average frame-level features to a single utterance-level feature.

    args:
        input_size: dimensionality of the input frame-level embeddings.
            Determined by encoder hyperparameter.
    """

    def __init__(self, input_size: int = 1536):
        super().__init__()
        self._output_size = input_size

    def output_size(self):
        return self._output_size

    def forward(self, x, task_tokens: torch.Tensor = None, feat_lengths: torch.Tensor = None):
        if task_tokens is not None:
            raise ValueError("MeanPooling is not adequate for task_tokens")
        if feat_lengths is not None:
            x = torch.stack([torch.mean(x[i, :, :l], dim=-1) for i, l in enumerate(feat_lengths)], dim=0)
        else:
            x = torch.mean(x, dim=-1)

        return x
