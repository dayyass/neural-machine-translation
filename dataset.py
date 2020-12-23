from typing import List, Tuple, Union

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def load_data(
    data_path: str,
    tokenizer: spm.SentencePieceProcessor,
    add_bos: bool = False,
    add_eos: bool = False,
    verbose: bool = True,
):
    """
    Load data and apply tokenizer to each sentence.
    """

    data_list = []
    with open(data_path, mode="r") as fp:
        if verbose:
            fp = tqdm(fp)  # add tqdm
        for line in fp:
            data_list.append(
                tokenizer.encode(line, add_bos=add_bos, add_eos=add_eos),  # tokenize
            )
    return data_list


def bucket_sequencing(
    seq_list: List[List[int]],
    percentile: Union[int, float],
    pad_id: int,
):
    """
    Bucket sequencing for variable-size sentences.
    """

    max_len = int(
        np.percentile(
            [len(seq) for seq in seq_list],
            percentile,
        )
    )

    for i in range(len(seq_list)):
        seq = seq_list[i][:max_len]  # clip
        seq += [pad_id] * (max_len - len(seq))  # pad
        seq_list[i] = seq

    return seq_list


class WMTDataset(Dataset):
    """
    WMT Dataset with tokenized sentences.
    """

    def __init__(
        self,
        from_lang_data_path: str,
        to_lang_data_path: str,
        from_lang_tokenizer: spm.SentencePieceProcessor,
        to_lang_tokenizer: spm.SentencePieceProcessor,
        verbose: bool = True,
    ):
        self.from_lang_data_path = from_lang_data_path
        self.to_lang_data_path = to_lang_data_path

        # tokenizers
        self.from_lang_tokenizer = from_lang_tokenizer
        self.to_lang_tokenizer = to_lang_tokenizer

        # load "from" language
        self.from_lang_list = load_data(
            data_path=self.from_lang_data_path,
            tokenizer=self.from_lang_tokenizer,
            add_bos=False,
            add_eos=False,
            verbose=verbose,
        )

        # load "to" language
        self.to_lang_list = load_data(
            data_path=self.to_lang_data_path,
            tokenizer=self.to_lang_tokenizer,
            add_bos=True,
            add_eos=True,
            verbose=verbose,
        )

        # filter empty lines
        good_idx = []
        for i in range(len(self.from_lang_list)):
            if (len(self.from_lang_list[i]) != 0) and (len(self.to_lang_list[i]) != 0):
                good_idx.append(i)

        self.from_lang_list = [self.from_lang_list[i] for i in good_idx]
        self.to_lang_list = [self.to_lang_list[i] for i in good_idx]

    def __len__(self):
        assert len(self.from_lang_list) == len(self.to_lang_list)

        return len(self.from_lang_list)

    def __getitem__(self, idx):
        return self.from_lang_list[idx], self.to_lang_list[idx]


class WMTCollator(object):
    """
    Collator that handles variable-size sentences.
    """

    def __init__(
        self,
        from_lang_pad_id: int = 3,
        to_lang_pad_id: int = 3,
        percentile: Union[int, float] = 100,
    ):
        self.from_lang_pad_id = from_lang_pad_id
        self.to_lang_pad_id = to_lang_pad_id
        self.percentile = percentile

    def __call__(
        self,
        batch: List[Tuple[List[int], List[int]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        from_lang_tuple, to_lang_tuple = zip(*batch)
        from_lang_list, to_lang_list = list(from_lang_tuple), list(to_lang_tuple)

        # bucket sequencing
        from_lang_list = bucket_sequencing(
            from_lang_list,
            percentile=self.percentile,
            pad_id=self.from_lang_pad_id,
        )

        to_lang_list = bucket_sequencing(
            to_lang_list,
            percentile=self.percentile,
            pad_id=self.to_lang_pad_id,
        )

        return torch.tensor(from_lang_list), torch.tensor(to_lang_list)
