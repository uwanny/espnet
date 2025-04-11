"""
Trainer module for language identification and language embedding extraction.
"""

from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import os
import logging
from tqdm import tqdm

from typeguard import typechecked
from espnet2.torch_utils.device_funcs import to_device
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import SubReporter
from espnet2.train.trainer import Trainer, TrainerOptions

if torch.distributed.is_available():
    from torch.distributed import ReduceOp


class LIDTrainer(Trainer):
    """Trainer designed for LID, adapted from spk_trainer.py
    """

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    @torch.no_grad()
    @typechecked
    def extract_embed_lid(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
        distributed_option: DistributedOption,
        output_dir: str,
        custom_bs: int,
        idx2lang: Dict[int, str],
        extract_embd: bool = False,
        save_every: int = 1000,
        resume: bool = True,
    ) -> None: 
        # Extract language embedding and lids. 
        ngpu = options.ngpu
        distributed = distributed_option.distributed

        model.eval()
        if extract_embd:
            lang_embd_dic = {}
        lang_id_dic = {} # {utt_id: lang_id}

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        # iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        # fill dictionary with speech samples
        utt_id_list = []
        utt_id_whole_list = []
        speech_list = []
        speech_length_list = []
        task_token = None
        if distributed:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        idx = 0
        step = 0 # for save middle results
        num_recheck = 0
        if resume:
            skip_utts = set()
            if os.path.exists(f"{output_dir}/lids{rank}"):
                with open(f"{output_dir}/lids{rank}", "r") as f:
                    for line in f:
                        utt_id, lid = line.strip().split()
                        skip_utts.add(utt_id)
            logging.info(f"[Rank {rank}] Resume: {len(skip_utts)} utterances found in {output_dir}/lids{rank}")
        for utt_id, batch in iterator:
            if "task_tokens" in batch:
                task_token = batch["task_tokens"][0]

            assert isinstance(batch, dict), type(batch)
            for _utt_id, _speech, _speech_length in zip(
                utt_id, batch["speech"], batch["speech_lengths"]
            ):
                if resume:
                    if _utt_id in skip_utts:
                        if num_recheck % save_every == 0 and num_recheck > 0:
                            logging.info(f"[Rank {rank}] Skip utterance {num_recheck - save_every}-{num_recheck}.")
                        num_recheck += 1
                        continue
                if _utt_id not in utt_id_whole_list:
                    utt_id_whole_list.append(_utt_id)
                    if idx % world_size == rank:
                        utt_id_list.append(_utt_id)
                        speech_list.append(_speech)
                        speech_length_list.append(_speech_length)
                    idx += 1

                    if len(utt_id_list) == custom_bs:
                        try:
                            speech_list = torch.stack(speech_list, dim=0) # (bs, t), t is the length of the speech
                            speech_length_list = torch.stack(speech_length_list, dim=0) # (bs,)
                        except RuntimeError as e: # for last few batches, pad to the same length
                            max_len = max(s.size(0) for s in speech_list)
                            speech_list = torch.stack(
                                [F.pad(s, (0, max_len - s.size(0))) for s in speech_list],
                                dim=0
                            )
                            speech_length_list = torch.stack(speech_length_list, dim=0)
                        speech_list = to_device(
                            speech_list, "cuda" if ngpu > 0 else "cpu"
                        )
                        speech_length_list = to_device(
                            speech_length_list, "cuda" if ngpu > 0 else "cpu"
                        )
                        if task_token is None:
                            task_tokens = None
                        else:
                            task_tokens = to_device(
                                task_token.repeat(speech_list.size(0)),
                                "cuda" if ngpu > 0 else "cpu",
                            ).unsqueeze(1)
                        lang_embds, pred_lids = model(
                            speech=speech_list,
                            speech_lengths=speech_length_list,
                            lid_labels=None,
                            task_tokens=task_tokens,
                            extract_embd=True,
                        ) # [batch_size, dim], [batch_size]
                        if extract_embd:
                            lang_embds = F.normalize(lang_embds, p=2, dim=1)
                        pred_lids = [idx2lang[lid.item()] for lid in pred_lids]

                        for uid, _lang_embd, _pred_lid in zip(utt_id_list, lang_embds, pred_lids):
                            if extract_embd:
                                lang_embd_dic[uid] = _lang_embd.detach().cpu().numpy()
                            lang_id_dic[uid] = _pred_lid
                        
                        # save middle results
                        if len(lang_id_dic) >= save_every:
                            if extract_embd:
                                # save each middle step results to different files
                                np.savez(output_dir + f"/embeddings{rank + world_size * step}", **lang_embd_dic)
                            with open(f"{output_dir}/lids{rank}", "a") as f:
                                # save all middle step results to the same file
                                for uid, lid in lang_id_dic.items():
                                    f.write(f"{uid} {lid}\n")
                            logging.info(f"[Rank {rank}] Saved {len(lang_id_dic)} utts at step {step}")
                            if extract_embd:
                                lang_embd_dic.clear()
                            lang_id_dic.clear()
                            step += 1

                        utt_id_list = []
                        speech_list = []
                        speech_length_list = []

        if len(utt_id_list) != 0:
            try:
                speech_list = torch.stack(speech_list, dim=0)
                speech_length_list = torch.stack(speech_length_list, dim=0) # (bs,)
            except RuntimeError as e: # for last few batches, pad to the same length
                max_len = max(s.size(0) for s in speech_list)
                speech_list = torch.stack(
                    [F.pad(s, (0, max_len - s.size(0))) for s in speech_list],
                    dim=0
                )
                speech_length_list = torch.stack(speech_length_list, dim=0)
            speech_list = to_device(
                speech_list, "cuda" if ngpu > 0 else "cpu"
            )
            speech_length_list = to_device(
                            speech_length_list, "cuda" if ngpu > 0 else "cpu"
                        )
            if task_token is None:
                task_tokens = None
            else:
                task_tokens = to_device(
                    task_token.repeat(speech_list.size(0)),
                    "cuda" if ngpu > 0 else "cpu",
                ).unsqueeze(1)
            lang_embds, pred_lids = model(
                speech=speech_list,
                speech_lengths=speech_length_list,
                lid_labels=None,
                task_tokens=task_tokens,
                extract_embd=True,
            ) # [batch_size, dim], [batch_size]
            lang_embds = F.normalize(lang_embds, p=2, dim=1)
            pred_lids = [idx2lang[lid.item()] for lid in pred_lids]

            for uid, _lang_embd, _pred_lid in zip(utt_id_list, lang_embds, pred_lids):
                if extract_embd:
                    lang_embd_dic[uid] = _lang_embd.detach().cpu().numpy()
                lang_id_dic[uid] = _pred_lid

        if len(lang_id_dic) != 0:
            # save the last results
            if extract_embd:
                np.savez(output_dir + f"/embeddings{rank + world_size * step}", **lang_embd_dic)
            with open(f"{output_dir}/lids{rank}", "a") as f:
                # save all middle step results to the same file
                for uid, lid in lang_id_dic.items():
                    f.write(f"{uid} {lid}\n")

