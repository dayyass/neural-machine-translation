import json
from typing import List

from mosestokenizer import MosesTokenizer


class Language:
    """
    Language abstraction to handle tokenizer, word2idx, idx2word.
    """

    def __init__(
        self,
        language: str,
        path_to_word2idx: str,
        unk_id: int,
        bos_id: int,
        eos_id: int,
    ):
        self.language = language
        self.path_to_word2idx = path_to_word2idx
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.tokenizer = MosesTokenizer(language)
        with open(path_to_word2idx, mode="r") as fp:
            self.word2idx = json.load(fp)
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode_sentence(
        self,
        sentence: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        seq = [
            self.word2idx.get(word, self.unk_id) for word in self.tokenizer(sentence)
        ]
        if add_bos:
            seq.insert(0, self.bos_id)
        if add_eos:
            seq.append(self.eos_id)
        return seq
