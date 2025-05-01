import random
from collections import defaultdict
from typing import Iterator, Optional, Tuple, Union, List
import numpy as np

from typeguard import typechecked
from espnet2.fileio.read_text import read_2columns_text
from espnet2.samplers.abs_sampler import AbsSampler
from espnet2.fileio.read_text import load_num_sequence_text
import logging


class CategoryPowerSampler_Origin_Back_Because_Slow(AbsSampler):
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

    During upsampling, since the low-resource languages are upsampled,
    it is like "repeating", but we wish to make the repeat happened cross
    the batch, not in the same batch. 

    So we use a dataset_scaling_factor to control the finally used utterance number. 

    Why use batch_bins instead of batch_size?
    To keep same with MMS-LID. 

    Args:
        shape_files: the shape files must be a list or tuple, in this sampler, 
        we only use one shape file, but to be campatible with other samplers, we
        still reserve the design of input with a shape file list style. 
        min_batch_size: the sampler will at least sample min_batch_size utterances
        max_batch_size: the sampler will at most sample max_batch_size utterances, recommended to tune according to GPU to avoid OOM. 
        upsampling_factor: the beta in the formula above
        dataset_scaling_factor: if 1, then the number of utterances used is just the
        original dataset size, but the sampling rate of each language is changed. 
        If > 1, then the number of utterances used is dataset_scaling_factor times.
        Remember, upsample is like repeat low-resource, but we wish to make it happened
        cross batch, not within batch. And we want to make low-resource language data
        happen more times in the final batch list. 
        dataset_scaling_factor must >= 1
    
    """
    @typechecked
    def __init__(
        self,
        batch_bins: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        upsampling_factor: float = 1.0,
        dataset_scaling_factor: float = 1.2, 
        drop_last: bool = False,
        category2utt_file: Optional[str] = None,
        epoch: int = 1,
        **kwargs,
    ):
        assert batch_bins > 0
        assert category2utt_file is not None
        assert dataset_scaling_factor >= 1, "dataset_scaling_factor must >= 1"
        # set random seed as epoch, since we want to make the the sample different in each epoch
        random.seed(epoch)
        np.random.seed(epoch)

        self.batch_bins = batch_bins
        self.drop_last = drop_last
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.upsampling_factor = upsampling_factor

        assert len(shape_files) == 1, "only one shape file is supported"
        utt2sizes = [
            load_num_sequence_text(s, loader_type="text_int") for s in shape_files
        ] # return a dict, key is utt id, value is speech size (length of audio, or # samples in audio)

        # load category -> list of utterances
        category2utt_raw = read_2columns_text(category2utt_file)
        self.category2utt = {k: v.split(" ") for k, v in category2utt_raw.items()}
        self.categories = list(self.category2utt.keys())

        # 1. compute n_l (the number of utterances in each category) and N (total number of utterances)
        # self.category_counts = {cat: len(utts) for cat, utts in self.category2utt.items()}
        self.category_bins = {cat: sum(utt2sizes[0][utt][0] for utt in utts) for cat, utts in self.category2utt.items()}
        # total_count = sum(self.category_counts.values())
        total_bins = sum(self.category_bins.values())
        assert dataset_scaling_factor >= 1, "dataset_scaling_factor must >= 1"
        # scaling_count = int(total_count * dataset_scaling_factor)
        scaling_bins = int(total_bins * dataset_scaling_factor)

        # 2. compute sampling prob of each category: p_l ∝ (n_l / N)^β
        probs = np.array([
            (self.category_bins[cat] / total_bins) ** upsampling_factor
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
        current_batch_samples = set()
        current_batch_bins = 0
        current_batch_size = 0
        sample_bins = 0
        while True:
            # sample a category
            cat = np.random.choice(self.categories, p=self.category_probs) # P(l)

            if not self.all_utts_by_category[cat]:
                continue

            utt = np.random.choice(self.all_utts_by_category[cat]) # P(x|l) = 1 / n_l, choice is uniform sampling
            if utt in current_batch_samples: # avoid repeat in the same batch
                continue
            current_batch.append(utt)
            current_batch_bins += utt2sizes[0][utt][0]
            current_batch_samples.add(utt)
            current_batch_size += 1
            sample_bins += utt2sizes[0][utt][0]

            if (
                current_batch_bins > self.batch_bins and current_batch_size >= self.min_batch_size
                or self.max_batch_size is not None and current_batch_size >= self.max_batch_size
            ):
                self.batch_list.append(current_batch)
                current_batch = []
                current_batch_bins = 0
                current_batch_samples = set()
                current_batch_size = 0
            
            if sample_bins >= scaling_bins:
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


class CategoryPowerSampler(AbsSampler):
    """
    A fast version of CategoryPowerSampler,

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

    During upsampling, since the low-resource languages are upsampled,
    it is like "repeating", but we wish to make the repeat happened cross
    the batch, not in the same batch. 

    So we use a dataset_scaling_factor to control the finally used utterance number. 

    Why use batch_bins instead of batch_size?
    To keep same with MMS-LID. 

    Compared to the original version, this would be faster because the original version
    did too many judges and jumps when sampling, and it samples one by one, this avoids 
    that by pre estimate the total number of utterances after upsampling the whole dataset, 
    and then sample the utterances according to the category_probs, and then patch them into batches.
    This reduces the original time on Voxlingua107 from 1 hour to 1 minute.

    Args:
        shape_files: the shape files must be a list or tuple, in this sampler, 
        we only use one shape file, but to be campatible with other samplers, we
        still reserve the design of input with a shape file list style. 
        min_batch_size: the sampler will at least sample min_batch_size utterances
        max_batch_size: the sampler will at most sample max_batch_size utterances, recommended to tune according to GPU to avoid OOM. 
        upsampling_factor: the beta in the formula above
        dataset_scaling_factor: if 1, then the number of utterances used is just the
        original dataset size, but the sampling rate of each language is changed. 
        If > 1, then the number of utterances used is dataset_scaling_factor times.
        Remember, upsample is like repeat low-resource, but we wish to make it happened
        cross batch, not within batch. And we want to make low-resource language data
        happen more times in the final batch list. 
        dataset_scaling_factor must >= 1
    
    """
    @typechecked
    def __init__(
        self,
        batch_bins: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        upsampling_factor: float = 1.0,
        dataset_scaling_factor: float = 1.2, 
        drop_last: bool = False,
        category2utt_file: Optional[str] = None,
        epoch: int = 1,
        **kwargs,
    ):
        assert batch_bins > 0
        assert category2utt_file is not None
        assert dataset_scaling_factor >= 1, "dataset_scaling_factor must >= 1"
        # set random seed as epoch, since we want to make the the sample different in each epoch
        random.seed(epoch)
        np.random.seed(epoch)

        self.batch_bins = batch_bins
        self.drop_last = drop_last
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.upsampling_factor = upsampling_factor

        assert len(shape_files) == 1, "only one shape file is supported"
        utt2sizes = [
            load_num_sequence_text(s, loader_type="text_int") for s in shape_files
        ] # return a dict, key is utt id, value is speech size (length of audio, or # samples in audio)

        # load category -> list of utterances
        category2utt_raw = read_2columns_text(category2utt_file)
        self.category2utt = {k: v.split(" ") for k, v in category2utt_raw.items()}
        self.categories = list(self.category2utt.keys())

        # 1. compute n_l (the number of utterances in each category) and N (total number of utterances)
        # self.category_counts = {cat: len(utts) for cat, utts in self.category2utt.items()}
        self.category_bins = {cat: sum(utt2sizes[0][utt][0] for utt in utts) for cat, utts in self.category2utt.items()}
        # total_count = sum(self.category_counts.values())
        total_bins = sum(self.category_bins.values())
        assert dataset_scaling_factor >= 1, "dataset_scaling_factor must >= 1"
        # scaling_count = int(total_count * dataset_scaling_factor)
        scaling_bins = int(total_bins * dataset_scaling_factor)

        # 2. compute sampling prob of each category: p_l ∝ (n_l / N)^β
        probs = np.array([
            (self.category_bins[cat] / total_bins) ** upsampling_factor
            for cat in self.categories
        ])
        probs /= probs.sum()  # normalize
        self.category_probs = probs

        # 3. flatten all utts by category
        self.all_utts_by_category = defaultdict(list)
        for cat, utts in self.category2utt.items():
            self.all_utts_by_category[cat].extend(utts)
            random.shuffle(self.all_utts_by_category[cat]) # shuffle, hint: P(x | l) = 1 / n_l, shuffle is like uniform sampling
        
        # 4. estimate the total number of utterances after upsampling the whole dataset
        utt_avg_size = np.mean([utt2sizes[0][utt][0] for utt in utt2sizes[0].keys()])
        total_num_samples = int(scaling_bins / utt_avg_size)

        # 5. first sample utterances according to category_probs
        # use cat_ptr to avoid repeatedly sampling the same utterance from the same category to form a batch
        # the pointer to the utterance in the category update after each sampling
        logging.info(f"++++++++++++++Begin sampling utterances")
        cat_ptr = {cat: 0 for cat in self.categories}
        sampled_utts = []
        for _ in range(total_num_samples):
            cat = np.random.choice(self.categories, p=self.category_probs) # P(l)
            idx = cat_ptr[cat] % len(self.all_utts_by_category[cat]) # the index might be back to the beginning, but it will not cause repeat, because they definetly cannot be patched into one batch
            utt = self.all_utts_by_category[cat][idx]
            cat_ptr[cat] += 1
            sampled_utts.append(utt)
        logging.info(f"++++++++++++++Finish sampling utterances, total sampled utterances: {len(sampled_utts)}")
        
        # 6. patch sampled utterances into batches
        logging.info(f"++++++++++++++Begin patching sampled utterances into batches")
        self.batch_list = []
        current_batch = []
        current_batch_bins = 0
        for utt in sampled_utts:
            utt_size = utt2sizes[0][utt][0]

            if (
                current_batch_bins > self.batch_bins and len(current_batch) >= self.min_batch_size
                or self.max_batch_size is not None and len(current_batch) >= self.max_batch_size
            ): 
                self.batch_list.append(current_batch)
                current_batch = []
                current_batch_bins = 0
            
            current_batch.append(utt)
            current_batch_bins += utt_size
        
        # 7. if the last batch is not empty, append it to the batch list
        if not self.drop_last and len(current_batch) >= 1:
            self.batch_list.append(current_batch)
        logging.info(f"++++++++++++++Finish patching sampled utterances into batches, total batches: {len(self.batch_list)}")

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

