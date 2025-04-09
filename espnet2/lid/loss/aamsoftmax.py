#! /usr/bin/python
# -*- encoding: utf-8 -*-
# code from https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)
# 暂时还没改好（整体改成lid，不留一点spk），还不能用

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.spk.loss.abs_loss import AbsLoss


class AAMSoftmax(AbsLoss):
    """Additive angular margin softmax.

    Paper: Deng, Jiankang, et al. "Arcface: Additive angular margin loss for
    deep face recognition." Proceedings of the IEEE/CVF conference on computer
    vision and pattern recognition. 2019.

    args:
        nout    : dimensionality of speaker embedding
        nclases: number of speakers in the training set
        margin  : margin value of AAMSoftmax
        scale   : scale value of AAMSoftmax
    """

    def __init__(
        self, nout, nclasses, margin=0.3, scale=15, easy_margin=False, **kwargs
    ):
        super().__init__(nout)

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nout
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(nclasses, nout), requires_grad=True
        )
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print("Initialised AAMSoftmax margin %.3f scale %.3f" % (self.m, self.s))

    def forward(self, x, label=None):
        if len(label.size()) == 2:
            label = label.squeeze(1)

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # cos(theta)
        # the normalized vector mul, the results is the cosine value
        cosine = F.linear(F.normalize(x), F.normalize(self.weight)) # F.normalize is L2 norm

        pred_lids = torch.argmax(cosine, dim=1)
        if label is not None: # train or dev
            accuracy = (pred_lids == label).float().mean()
        else:
            accuracy = None

        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m # cos(theta + margin)

        if self.easy_margin:
            # to avoid 
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        # one_hot * phi is for the correct class, which needs margin, 
        # while (1 - one_hot) * cosine is for the incorrect classes, which don't need margin
        # output is the added margin logits, for softmax
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output = output * self.s
        
        loss = self.ce(output, label)
        return loss, accuracy, pred_lids
