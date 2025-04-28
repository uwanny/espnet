# The original code is copied from espnet2/asr/encoder/linear_encoder.py

from typing import Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.attention import (  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoderLayer
from espnet2.asr.layers.fastformer import FastSelfAttention
from espnet2.asr.layers.cgmlp import ConvolutionalGatingMLP

class AttentionPooling(torch.nn.Module):
    def __init__(self, embed_dim):
        super(AttentionPooling, self).__init__()
        self.query = torch.nn.Parameter(torch.randn(1, 1, embed_dim))
        self.scale = embed_dim ** 0.5

    def forward(self, h):
        """
        h: (B, D, L)
        Returns:
            pooled_representation: (B, D)
        """
        attn_weights = torch.matmul(self.query, h) / self.scale
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        pooled_representation = torch.matmul(attn_weights, h.permute(0, 2, 1)).squeeze(1)
        return pooled_representation


class LIDEncoderEBranchformer(AbsEncoder):
    """Linear encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        linear_units: the number of units of position-wise feed forward
        dropout_rate: dropout rate
        input_layer: input layer type
        normalize_before: whether to use layer_norm before the first block
        padding_idx: padding_idx for input_layer=embed
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        dropout_rate: float = 0.1,
        input_layer: Optional[str] = None,
        normalize_before: bool = False,
        padding_idx: int = -1,
        pooling_mode: str = "avg_pooling", # avg_pooling, attn_pooling
        vocab_size: int = 0,
        num_blocks: int = 2,
        attention_heads: int = 8, 
        attention_dropout_rate: float = 0.1,
        use_flash_attn: bool = False,
        positionwise_layer_type: str = "linear",
        linear_units: int = 2048,
        positionwise_conv_kernel_size: int = 1,
        layer_drop_rate: float = 0.1,
        attention_layer_type: str = "rel_selfattn",
        pos_enc_layer_type: str = "rel_pos",
        cgmlp_linear_units: int = 2048,
        cgmlp_conv_kernel: int = 31,
        use_linear_after_conv: bool = False,
        gate_activation: str = "identity",
        zero_triu: bool = False,
        use_ffn: bool = False,
        macaron_ffn: bool = False,
        merge_conv_kernel: int = 3,
        activation_ckpt=False,
    ):
        super().__init__()
        self._output_size = output_size

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, output_size, dropout_rate)
        elif input_layer == "embed":
            self.embed = (
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
            )
        elif input_layer is None:
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
    
        if attention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                False,
                use_flash_attn,
                False,
                False,
            )
        elif attention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif attention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        elif attention_layer_type == "fast_selfattn":
            assert pos_enc_layer_type in ["abs_pos", "scaled_abs_pos"]
            encoder_selfattn_layer = FastSelfAttention
            encoder_selfattn_layer_args = (
                output_size,
                attention_heads,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + attention_layer_type)

        cgmlp_layer = ConvolutionalGatingMLP
        cgmlp_layer_args = (
            output_size,
            cgmlp_linear_units,
            cgmlp_conv_kernel,
            dropout_rate,
            use_linear_after_conv,
            gate_activation,
            activation_ckpt,
        )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EBranchformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                cgmlp_layer(*cgmlp_layer_args),
                positionwise_layer(*positionwise_layer_args) if use_ffn else None,
                (
                    positionwise_layer(*positionwise_layer_args)
                    if use_ffn and macaron_ffn
                    else None
                ),
                dropout_rate,
                merge_conv_kernel,
                activation_ckpt,
            ),
            layer_drop_rate,
        )
        
        self.pooling_mode = pooling_mode
        if pooling_mode == "avg_pooling":
            self.pooling = torch.nn.AdaptiveAvgPool1d(output_size=1)
        elif pooling_mode == "attn_pooling":
            self.pooling = AttentionPooling(embed_dim=output_size) 
        
        self.head = torch.nn.Sequential(
            torch.nn.Linear(output_size, vocab_size),
            # torch.nn.Softmax(dim=-1)
        )

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            cls_out: (B, Vocab_size)
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if self.embed is None:
            xs_pad = xs_pad
        elif (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)
        
        for layer_idx, encoder_layer in enumerate(self.encoders):
            xs_pad, masks = encoder_layer(xs_pad, masks)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        # xs_pad: [B, L, D] -> [B, D, L]
        xs_pad = xs_pad.permute(0, 2, 1)

        if self.pooling_mode == "avg_pooling":
            pooled_representation = self.pooling(xs_pad).squeeze(-1) # [B, D]
        elif self.pooling_mode == "attn_pooling":
            pooled_representation = self.pooling(xs_pad)

        cls_out = self.head(pooled_representation)


        return cls_out
