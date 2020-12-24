import json
from collections import Counter
from typing import Counter as CounterType
from typing import Dict

from mosestokenizer import MosesTokenizer
from tqdm import tqdm


def get_word2idx(
    path: str,
    tokenizer: MosesTokenizer,
    min_count: int = 5,
    add_unk: bool = True,
    add_bos: bool = True,
    add_eos: bool = True,
    add_pad: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Get mapping from words to indices to use with Embedding layer.
    """

    word2idx: Dict[str, int] = {}
    counter: CounterType[str] = Counter()

    with open(path, mode="r") as fp:
        if verbose:
            fp = tqdm(fp)
        for line in fp:
            counter.update(tokenizer(line))

    if add_unk:
        word2idx["<unk>"] = len(word2idx)
    if add_bos:
        word2idx["<bos>"] = len(word2idx)
    if add_eos:
        word2idx["<eos>"] = len(word2idx)
    if add_pad:
        word2idx["<pad>"] = len(word2idx)

    for word, cnt in counter.most_common():
        if cnt >= min_count:
            word2idx[word] = len(word2idx)

    return word2idx


if __name__ == "__main__":

    # path
    INPUT_LANG = "en"  # moses language
    OUTPUT_LANG = "vi"  # moses language
    INPUT_LANG_DATA_PATH = "../data/IWSLT15_English_Vietnamese/train.en"
    OUTPUT_LANG_DATA_PATH = "../data/IWSLT15_English_Vietnamese/train.vi"
    INPUT_LANG_VOCAB_SAVE_PATH = f"{INPUT_LANG}_vocab.json"
    OUTPUT_LANG_VOCAB_SAVE_PATH = f"{OUTPUT_LANG}_vocab.json"

    # hyper-parameters
    MIN_COUNT = 5
    ADD_UNK = True
    ADD_BOS = True
    ADD_EOS = True
    ADD_PAD = True
    VERBOSE = True

    if VERBOSE:
        print("### PARAMETERS ###")
        print()
        print(f"INPUT_LANG: {INPUT_LANG}")
        print(f"OUTPUT_LANG: {OUTPUT_LANG}")
        print(f"INPUT_LANG_DATA_PATH: {INPUT_LANG_DATA_PATH}")
        print(f"OUTPUT_LANG_DATA_PATH: {OUTPUT_LANG_DATA_PATH}")
        print(f"INPUT_LANG_VOCAB_SAVE_PATH: {INPUT_LANG_VOCAB_SAVE_PATH}")
        print(f"OUTPUT_LANG_VOCAB_SAVE_PATH: {OUTPUT_LANG_VOCAB_SAVE_PATH}")
        print()
        print(f"MIN_COUNT: {MIN_COUNT}")
        print(f"ADD_UNK: {ADD_UNK}")
        print(f"ADD_BOS: {ADD_BOS}")
        print(f"ADD_EOS: {ADD_EOS}")
        print(f"ADD_PAD: {ADD_PAD}")
        print()

    # vocab
    input_tokenizer = MosesTokenizer(INPUT_LANG)
    output_tokenizer = MosesTokenizer(OUTPUT_LANG)

    input_lang_word2idx = get_word2idx(
        path=INPUT_LANG_DATA_PATH,
        tokenizer=input_tokenizer,
        min_count=MIN_COUNT,
        add_unk=ADD_UNK,
        add_bos=ADD_BOS,
        add_eos=ADD_EOS,
        add_pad=ADD_PAD,
        verbose=VERBOSE,
    )

    output_lang_word2idx = get_word2idx(
        path=OUTPUT_LANG_DATA_PATH,
        tokenizer=output_tokenizer,
        min_count=MIN_COUNT,
        add_unk=ADD_UNK,
        add_bos=ADD_BOS,
        add_eos=ADD_EOS,
        add_pad=ADD_PAD,
        verbose=VERBOSE,
    )

    # save
    with open(f"{INPUT_LANG_VOCAB_SAVE_PATH}", mode="w") as fp:
        json.dump(input_lang_word2idx, fp)

    with open(f"{OUTPUT_LANG_VOCAB_SAVE_PATH}", mode="w") as fp:
        json.dump(output_lang_word2idx, fp)
