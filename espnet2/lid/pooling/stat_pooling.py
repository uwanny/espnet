import torch

from espnet2.spk.pooling.abs_pooling import AbsPooling


class StatsPooling(AbsPooling):
    """Aggregates frame-level features to single utterance-level feature.

    Proposed in D. Snyder et al., "X-vectors: Robust dnn embeddings for speaker
    recognition"

    args:
        input_size: dimensionality of the input frame-level embeddings.
            Determined by encoder hyperparameter.
            For this pooling layer, the output dimensionality will be double of
            the input_size
    """

    def __init__(self, input_size: int = 1536):
        super().__init__()
        self._output_size = input_size * 2

    def output_size(self):
        return self._output_size

    def forward(self, x, task_tokens: torch.Tensor = None, feat_lengths: torch.Tensor = None):
        if task_tokens is not None:
            raise ValueError("StatisticsPooling is not adequate for task_tokens")
        
        if feat_lengths is not None:
            mu = torch.stack(
                [torch.mean(x[i, :, :l.item()], dim=-1) for i, l in enumerate(feat_lengths)],
                dim=0,
            )
            st = torch.stack(
                [torch.std(x[i, :, :l.item()], dim=-1, unbiased=False) for i, l in enumerate(feat_lengths)],
                dim=0,
            ) # unbiased=False, refer to https://pytorch.org/docs/stable/generated/torch.std.html
        else:
            mu = torch.mean(x, dim=-1)
            st = torch.std(x, dim=-1, unbiased=False)

        x = torch.cat((mu, st), dim=1)

        return x
