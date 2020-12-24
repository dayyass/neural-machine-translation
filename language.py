import json

from mosestokenizer import MosesTokenizer


class Language:
    """
    Language abstraction to handle tokenizer, word2idx, idx2word.
    """

    def __init__(self, language: str, path_to_word2idx: str):
        self.language = language
        self.tokenizer = MosesTokenizer(language)
        with open(path_to_word2idx, mode="r") as fp:
            self.word2idx = json.load(fp)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
