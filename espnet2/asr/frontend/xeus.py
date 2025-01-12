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
from espnet2.tasks.ssl import SSLTask
from espnet2.train.abs_espnet_model import AbsESPnetModel

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
        self.num_layers = ssl_model_args.encoder_conf["num_blocks"]

        if self.num_layers > 1:
            if layer_selections is not None:
                assert self.num_layers >= len(layer_selections)
                self.layer_selections = sorted(layer_selections)
            else:
                self.layer_selections = list(range(self.num_layers))
            self.weights = torch.nn.Parameter(torch.zeros(len(self.layer_selections)))

    def _weighted_sum(self, all_hs, all_lens):
        assert len(all_hs) == len(all_lens) > 1
        for l in all_lens[1:]:
            torch.allclose(all_lens[0], l)
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

        all_hs = [h for idx, h in enumerate(all_hs) if idx in self.layer_selections]
        all_lens = [l for idx, l in enumerate(all_lens) if idx in self.layer_selections]
        hs, hs_len = self._weighted_sum(all_hs, all_lens)
        return hs, hs_len

class XEUSFrontend(AbsFrontend):
    """Speech Pretrained Representation frontend structure for ASR."""

    @typechecked
    def __init__(
        self,
        fs: Union[int, str], 
        checkpoint: str, 
        config: str, 
        use_mask: bool = False, 
        use_final_output: bool = False, 
        layer: int = -1,
        multilayer_feature: bool = False,
        tile_factor: int = 1
    ): 
        super().__init__()

        self.use_mask = use_mask
        self.use_final_output = use_final_output
        self.layer = layer
        self.multilayer_feature = multilayer_feature
        self.tile_factor = tile_factor
        self.fs = fs

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.xeus_model, self.xeus_train_args = SSLTask.build_model_from_file(
            None,
            checkpoint,
            device
        )
        self.xeus_model.eval()

        if layer != -1:
            layer_selections = [layer]
            assert (
                not multilayer_feature
            ), "multilayer feature will be deactivated, when specific layer used"
        else:
            layer_selections = None
        # Weighted (trainable) sum of the features from the selected layers.
        self.featurizer = Featurizer(
            self.xeus_model, self.xeus_train_args, layer_selections=layer_selections
        )

    def output_size(self) -> int:
        return self.xeus_train_args.encoder_conf["output_size"]

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
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, _, _, feats_lens = self.xeus_model.encode(
            input, input_lengths, self.use_mask, self.use_final_output
        ) # feats: List[Tensor (batch, seq_len, feature_dim), ...] len(feats) == num_layers
        feats_lens = [feats_lens] * len(feats) # len(feats_lens) == num_layers
        
        if self.layer != -1:
            layer = self.layer
            feats, feats_lens = feats[layer], feats_lens[layer]
            return feats, feats_lens

        if self.multilayer_feature:
            feats, feats_lens = self.featurizer(feats, feats_lens)
        else:
            feats, feats_lens = self.featurizer(feats[-1:], feats_lens[-1:])

        if self.tile_factor != 1:
            feats = self._tile_representations(feats)

        return feats, feats_lens
