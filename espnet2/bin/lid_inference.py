#!/usr/bin/env python3
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union
from torch.multiprocessing.spawn import ProcessContext

import numpy as np
import torch
from typeguard import typechecked
from espnet2.train.distributed_utils import (
    DistributedOption,
    free_port,
    get_master_port,
    get_node_rank,
    get_num_nodes,
    resolve_distributed_mode,
)
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.tasks.lid import LIDTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args
from tqdm import tqdm


class Speech2LID:
    """
    Speech to language embeddings and ids (like iso3 codes). 

    """

    @typechecked
    def __init__(
        self,
        train_config: Union[Path, str, None] = None,
        model_file: Union[Path, str, None] = None,
        device: str = "cpu",
        dtype: str = "float32",
        batch_size: int = 1,
    ):

        lid_model, lid_train_args = LIDTask.build_model_from_file(
            train_config, model_file, device
        )
        self.lid_model = lid_model.eval()
        self.lid_train_args = lid_train_args
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

    @torch.no_grad()
    @typechecked
    def __call__(self, speech: Union[torch.Tensor, np.ndarray], lid_labels) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        """Inference

        Args:
            speech: Input speech data

        Returns:
            lang_embd, lid

        """

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        if len(speech.size()) == 1:
            speech = speech.unsqueeze(0).to(getattr(torch, self.dtype)) # batch size 1
            lid_labels = lid_labels.unsqueeze(0).to(torch.int64)
        elif len(speech.size()) == 2:
            speech = speech.to(getattr(torch, self.dtype)) # batch size not 1
            lid_labels = lid_labels.to(torch.int64)
        logging.info("speech length: " + str(speech.size(1)))
        batch = {"speech": speech, "lid_labels": lid_labels, "extract_embd": True}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward the model embedding extraction
        lang_embd, lid = self.lid_model(**batch)

        return lang_embd, lid

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build Speech2LID instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.

        Returns:
            Speech2Text: Speech2LID instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return Speech2LID(**kwargs)


@typechecked
def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    model_tag: Optional[str],
):
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2embedding
    speech2lid_kwargs = dict(
        batch_size=batch_size,
        dtype=dtype,
        train_config=train_config,
        model_file=model_file,
    )

    speech2lid = Speech2LID.from_pretrained(
        model_tag=model_tag,
        **speech2lid_kwargs,
    )

    # 3. Build data-iterator
    loader = LIDTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=LIDTask.build_preprocess_fn(
            speech2lid.lid_train_args, False
        ),
        collate_fn=LIDTask.build_collate_fn(speech2lid.lid_train_args, False),
        inference=True,
    )

    # 4. Create idx2lang dict, like {0: "eng", 1: "fra", ...}
    with open(speech2lid.lid_train_args.spk2utt, "r") as f:
        spk2utt = f.readlines()
    lang_idx = 0
    lang2idx = {}
    for line in spk2utt:
        lang = line.strip().split()[0]
        lang2idx[lang] = lang_idx
        lang_idx += 1
    idx2lang = {v: k for k, v in lang2idx.items()}

    # 5. Start for-loop
    with NpyScpWriter(os.path.join(output_dir, "embed"), os.path.join(output_dir, "embed.scp")) as embed_writer, \
         open(os.path.join(output_dir, "lid"), "w") as lid_file:
        for keys, batch in tqdm(loader):
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}
            lang_embds, lids = speech2lid(**batch)
            lids = [idx2lang[lid.cpu().item()] for lid in lids]

            for key, lang_embd, lid in zip(keys, lang_embds, lids):
                # embed_writer[key] = lang_embd.cpu().numpy()
                lid_file.write(f"{key} {lid}\n")


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Language emebedding extraction and language id inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--train_config",
        type=str,
        help="Speaker model training configuration",
    )
    group.add_argument(
        "--model_file",
        type=str,
        help="Speaker model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
        "*_file will be overwritten",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
