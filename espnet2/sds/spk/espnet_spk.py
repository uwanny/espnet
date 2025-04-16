import librosa  # noqa
import numpy as np
import torch
from typeguard import typechecked


from espnet2.bin.spk_inference import Speech2Embedding
from espnet2.sds.spk.abs_spk import AbsSPK


class ESPnetSPKModel(AbsSPK):
    """ESPnet SPK"""

    @typechecked
    def __init__(
        self,
        tag: str = ("espnet/" "voxcelebs12_rawnet3"),
        device: str = "cuda",
        dtype: str = "float16",
    ):
        """Initializer method.

        Args:
        tag (str, optional):
            The pre-trained model tag (on Hugging Face).
            Defaults to:
            "espnet/voxcelebs12_rawnet3".
        device (str, optional):
            The computation device for running inference.
            Defaults to "cuda".
            Common options include "cuda" or "cpu".
        dtype (str, optional):
            The floating-point precision to use.
            Defaults to "float16".
        """
        super().__init__()
        self.spk = Speech2Embedding.from_pretrained(
            model_tag=tag,
            device=device,
            batch_size=1,
        )
        self.device = device
        self.dtype = dtype

    def warmup(self):
        """Perform a single forward pass with dummy input to

        pre-load and warm up the model.
        """
        with torch.no_grad():
            dummy_input = (
                torch.randn(
                    (3000),
                    dtype=getattr(torch, self.dtype),
                    device="cpu",
                )
                .cpu()
                .numpy()
            )
            _ = self.spk(dummy_input)

    def forward(self, array: np.ndarray) -> str:
        """Perform a forward pass on the given audio data,

        returning the speaker embedding.

        Args:
            array (np.ndarray):
                The input audio data to be converted to speaker embedding.
                Typically a NumPy array.

        Returns:
            tensor:
                The speaker embedding from the audio input,
                as returned by the speech embedding model.
        """
        with torch.no_grad():
            embedding = self.spk(array)
            return embedding
