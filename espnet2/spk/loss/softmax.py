import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.spk.loss.abs_loss import AbsLoss


class Softmax(AbsLoss):
    """
    softmax loss
    """

    def __init__(
        self, nout, nclasses, **kwargs
    ):
        super().__init__(nout)

        self.test_normalize = True

        self.in_feats = nout
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(nclasses, nout), requires_grad=True
        )
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

    def forward(self, x, label=None):
        if len(label.size()) == 2:
            label = label.squeeze(1)

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        logits = F.linear(F.normalize(x), F.normalize(self.weight))
        loss = self.ce(logits, label)
        pred_lids = torch.argmax(logits, dim=1)
        accuracy = (pred_lids == label).float().mean()
        return loss, accuracy