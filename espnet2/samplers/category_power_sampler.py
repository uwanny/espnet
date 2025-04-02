import random
from collections import defaultdict
from typing import Iterator, Optional, Tuple, Union, List
import numpy as np

from typeguard import typechecked
from espnet2.fileio.read_text import read_2columns_text
from espnet2.samplers.abs_sampler import AbsSampler
from espnet2.fileio.read_text import load_num_sequence_text


class CategoryPowerSampler(AbsSampler):
    """
    If use this batch sampler, please denote `batch_type` as `catpow`

    This is a sampler used in MMS for balanced sampling of 
    languages in one dataset to form a batch. 

    The basic idea is as follows:
    
    l is the language category, l = {1, 2, ..., L}
    n_l is the number of utterances in category l
    N is the total number of utterances in the dataset

    The probability of sampling a category l is:
    P(l) = (n_l / N)^β, 
    where β is the upsampling factor.

    The probability of sampling an utterance from category l is:
    P(x | l) = 1 / n_l

    Then, the probability of sampling an utterance x is:
    P(x) = P(l) * P(x | l) = (n_l / N)^β * (1 / n_l)

    Zen of upsampling factor setting:
    - β -> 0, upsample the low-resource languages, downsample the high-resource languages
    - β -> 1, more and more like uniform sampling, that sample more from high-resource languages, less from low-resource languages

    The batch form style is like LengthBatchSampler, use batch_bins to control the batch size.
    """
    @typechecked
    def __init__(
        self,
        batch_bins: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        min_batch_size: int = 1,
        upsampling_factor: float = 1.0,
        drop_last: bool = False,
        category2utt_file: Optional[str] = None,
        seed: int = 1,
        **kwargs,
    ):
        assert batch_bins > 0
        assert category2utt_file is not None
        random.seed(seed)
        np.random.seed(seed)

        self.batch_bins = batch_bins
        self.drop_last = drop_last
        self.min_batch_size = min_batch_size
        self.upsampling_factor = upsampling_factor

        utt2sizes = [
            load_num_sequence_text(s, loader_type="text_int") for s in shape_files
        ] # return a dict, key is utt id, value is speech size (length of audio, or # samples in audio)

        # load category -> list of utterances
        category2utt_raw = read_2columns_text(category2utt_file)
        self.category2utt = {k: v.split(" ") for k, v in category2utt_raw.items()}
        self.categories = list(self.category2utt.keys())

        # 1. compute n_l (the number of utterances in each category) and N (total number of utterances)
        self.category_counts = {cat: len(utts) for cat, utts in self.category2utt.items()}
        total_count = sum(self.category_counts.values())

        # 2. compute sampling prob of each category: p_l ∝ (n_l / N)^β
        probs = np.array([
            (self.category_counts[cat] / total_count) ** upsampling_factor
            for cat in self.categories
        ])
        probs /= probs.sum()  # normalize
        self.category_probs = probs

        # 3. flatten all utts by category
        self.all_utts_by_category = defaultdict(list)
        for cat, utts in self.category2utt.items():
            self.all_utts_by_category[cat].extend(utts)
            random.shuffle(self.all_utts_by_category[cat]) # shuffle, hint: P(x | l) = 1 / n_l, shuffle is like uniform sampling

        # 4. make batches
        self.batch_list = []
        current_batch = []
        current_batch_bins = 0
        while True:
            # sample a category
            cat = np.random.choice(self.categories, p=self.category_probs)

            if not self.all_utts_by_category[cat]:
                continue

            utt = self.all_utts_by_category[cat].pop()
            current_batch.append(utt)
            current_batch_bins += utt2sizes[0][utt][0]

            if current_batch_bins > self.batch_bins and current_batch_bins >= self.min_batch_size:
                self.batch_list.append(current_batch)
                current_batch = []
                current_batch_bins = 0
            
            if all(len(v) == 0 for v in self.all_utts_by_category.values()):
                break

        if not self.drop_last and len(current_batch) >= 1:
            self.batch_list.append(current_batch)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_bins={self.batch_bins}, "
            f"upsampling_factor={self.upsampling_factor})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
