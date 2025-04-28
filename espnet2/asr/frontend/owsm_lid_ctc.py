import copy
import logging
import argparse
from typing import List, Tuple, Union
from typing import Optional, Tuple, Union

import humanfriendly
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet2.tasks.s2t_ctc import S2TTask
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.s2t.espnet_ctc_model import ESPnetS2TCTCModel

class Featurizer(torch.nn.Module):

    def __init__(
        self,
        ssl_model: AbsESPnetModel,
        ssl_model_args: argparse.Namespace,
        layer_selections: List[int] = None,
        normalize: bool = False,
    ):
        super().__init__()
        self.normalize = normalize
        self.num_layers = len(layer_selections) if layer_selections is not None else ssl_model_args.encoder_conf["num_blocks"]

        if self.num_layers > 1:
            if layer_selections is not None:
                assert self.num_layers >= len(layer_selections)
                self.layer_selections = sorted(layer_selections)
            else:
                self.layer_selections = list(range(self.num_layers))
            self.weights = torch.nn.Parameter(torch.zeros(len(self.layer_selections)))

    def _weighted_sum(self, all_hs, all_lens):
        assert len(all_hs) == len(all_lens) > 1
        stacked_hs = torch.stack(all_hs, dim=0)

        if self.normalize:
            stacked_hs = F.layer_norm(stacked_hs, (stacked_hs.shape[-1],))

        _, *origin_shape = stacked_hs.shape
        stacked_hs = stacked_hs.view(len(self.layer_selections), -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_hs = (norm_weights.unsqueeze(-1) * stacked_hs).sum(dim=0)
        weighted_hs = weighted_hs.view(*origin_shape)

        return weighted_hs, all_lens[0]

    def forward(
        self, all_hs: List[torch.FloatTensor], all_lens: List[torch.LongTensor]
    ):
        if len(all_hs) == 1:
            return all_hs[0], all_lens[0]

        hs, hs_len = self._weighted_sum(all_hs, all_lens)
        return hs, hs_len

class OWSMLIDCTCFrontend(AbsFrontend):
    """Speech Pretrained Representation frontend structure for ASR."""

    @typechecked
    def __init__(
        self,
        fs: Union[int, str], 
        checkpoint: str, 
        config: str, 
        layer_selections: Optional[List[int]] = None,
        multilayer_feature: bool = False,
        tile_factor: int = 1, 
        use_all_layer_outs: bool = True, 
    ): 
        super().__init__()

        self.multilayer_feature = multilayer_feature
        self.tile_factor = tile_factor
        self.fs = fs
        self.use_all_layer_outs = use_all_layer_outs

        if use_all_layer_outs: 
            assert layer_selections is None, "layer_selections must be None when use_all_layer_outs is True"
        self.layer_selections = layer_selections

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.owsm_model, self.owsm_train_args = S2TTask.build_model_from_file(
            config,
            checkpoint,
            device
        )
        self.owsm_model.eval()

        # Weighted (trainable) sum of the features from the selected layers.
        self.featurizer = Featurizer(
            self.owsm_model, self.owsm_train_args, layer_selections=self.layer_selections
        )

    def output_size(self) -> int:
        return self.owsm_train_args.encoder_conf["output_size"]

    def _tile_representations(self, feature):
        """Tile up the representations by `tile_factor`.

        Input - sequence of representations
                shape: (batch_size, seq_len, feature_dim)

        Output - sequence of tiled representations
                 shape: (batch_size, seq_len * factor, feature_dim)
        """
        assert (
            len(feature.shape) == 3
        ), "Input argument `feature` has invalid shape: {}".format(feature.shape)
        tiled_feature = feature.repeat(1, 1, self.tile_factor)
        tiled_feature = tiled_feature.reshape(
            feature.size(0), feature.size(1) * self.tile_factor, feature.size(2)
        )
        return tiled_feature

    def forward(
        self, 
        input: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, feats_lens, all_layer_outs = self.owsm_model.encode_only_encoder(
            input, 
            input_lengths,
        ) 

        # feats: List[Tensor (batch, seq_len, feature_dim), ...] len(feats) == num_layers
        if isinstance(feats, tuple): 
            last_layer_out, intermediate_out = feats # intermediate_out [(index, encoder_out), ...]
            intermediate_out = [x[1] for x in intermediate_out]
            if self.use_all_layer_outs:
                feats = all_layer_outs
            else: 
                feats = [last_layer_out] + intermediate_out
            feats_lens = [feats_lens] * len(feats) # len(feats_lens) == num_layers
        else: 
            feats = all_layer_outs
            feats_lens = [feats_lens] * len(feats)

        if self.multilayer_feature:
            feats, feats_lens = self.featurizer(feats, feats_lens)
        else:
            feats, feats_lens = self.featurizer(feats[-1:], feats_lens[-1:])

        if self.tile_factor != 1:
            feats = self._tile_representations(feats)

        return feats, feats_lens, all_layer_outs
