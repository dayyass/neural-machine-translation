from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from language import Language


def load_data(
    data_path: str,
    language: Language,
    add_bos: bool = False,
    add_eos: bool = False,
    verbose: bool = True,
) -> List[List[int]]:
    """
    Load data and apply word2idx to each sentence.
    """

    data_list = []
    with open(data_path, mode="r") as fp:
        if verbose:
            fp = tqdm(fp)
        for line in fp:
            data_list.append(
                language.encode_sentence(line, add_bos=add_bos, add_eos=add_eos),
            )
    return data_list


def bucket_sequencing(
    seq_list: List[List[int]],
    percentile: Union[int, float],
    pad_id: int,
) -> List[List[int]]:
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
        input_lang_data_path: str,
        output_lang_data_path: str,
        input_language: Language,
        output_language: Language,
        reverse_source_lang: bool = True,
        verbose: bool = True,
    ):
        self.input_lang_data_path = input_lang_data_path
        self.output_lang_data_path = output_lang_data_path

        # language
        self.input_language = input_language
        self.output_language = output_language

        # load input language
        self.input_lang_list = load_data(
            data_path=self.input_lang_data_path,
            language=self.input_language,
            add_bos=False,
            add_eos=False,
            verbose=verbose,
        )

        if reverse_source_lang:
            self.input_lang_list = [
                list(reversed(seq)) for seq in self.input_lang_list
            ]  # reverse

        # load output language
        self.output_lang_list = load_data(
            data_path=self.output_lang_data_path,
            language=self.output_language,
            add_bos=True,
            add_eos=True,
            verbose=verbose,
        )

        # filter empty lines
        good_idx = []
        for i in range(len(self.input_lang_list)):
            if (len(self.input_lang_list[i]) != 0) and (
                len(self.output_lang_list[i]) != 0
            ):
                good_idx.append(i)

        self.input_lang_list = [self.input_lang_list[i] for i in good_idx]
        self.output_lang_list = [self.output_lang_list[i] for i in good_idx]

    def __len__(self) -> int:
        assert len(self.input_lang_list) == len(self.output_lang_list)

        return len(self.input_lang_list)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.input_lang_list[idx], self.output_lang_list[idx]


class WMTCollator(object):
    """
    Collator that handles variable-size sentences.
    """

    def __init__(
        self,
        input_lang_pad_id: int = 3,
        output_lang_pad_id: int = 3,
        percentile: Union[int, float] = 100,
    ):
        self.input_lang_pad_id = input_lang_pad_id
        self.output_lang_pad_id = output_lang_pad_id
        self.percentile = percentile

    def __call__(
        self,
        batch: List[Tuple[List[int], List[int]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        input_lang_tuple, output_lang_tuple = zip(*batch)
        input_lang_list, output_lang_list = list(input_lang_tuple), list(
            output_lang_tuple
        )

        # bucket sequencing
        input_lang_list = bucket_sequencing(
            input_lang_list,
            percentile=self.percentile,
            pad_id=self.input_lang_pad_id,
        )

        output_lang_list = bucket_sequencing(
            output_lang_list,
            percentile=self.percentile,
            pad_id=self.output_lang_pad_id,
        )

        return torch.tensor(input_lang_list), torch.tensor(output_lang_list)
