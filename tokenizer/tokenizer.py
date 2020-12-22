import sentencepiece as spm

# path
FROM_LANG_DATA_PATH = "../data/IWSLT15_English_Vietnamese/train.en"
TO_LANG_DATA_PATH = "../data/IWSLT15_English_Vietnamese/train.vi"
FROM_LANG_TOKENIZER_PREFIX = "en"
TO_LANG_TOKENIZER_PREFIX = "vi"

# hyper-parameters
FROM_LANG_VOCAB_SIZE = 29756  # max possible value
TO_LANG_VOCAB_SIZE = 14427  # max possible value
CHARACTER_COVERAGE = 1
MODEL_TYPE = "unigram"
PAD_ID = 3
INPUT_SENTENCE_SIZE = (
    1000000  # if not enough RAM (https://github.com/google/sentencepiece/issues/341)
)


# train "from" lang
spm.SentencePieceTrainer.Train(
    input=FROM_LANG_DATA_PATH,
    model_prefix=FROM_LANG_TOKENIZER_PREFIX,
    vocab_size=FROM_LANG_VOCAB_SIZE,
    character_coverage=CHARACTER_COVERAGE,
    model_type=MODEL_TYPE,
    pad_id=PAD_ID,
    # input_sentence_size=INPUT_SENTENCE_SIZE,
)

# train "to" lang
spm.SentencePieceTrainer.Train(
    input=TO_LANG_DATA_PATH,
    model_prefix=TO_LANG_TOKENIZER_PREFIX,
    vocab_size=TO_LANG_VOCAB_SIZE,
    character_coverage=CHARACTER_COVERAGE,
    model_type=MODEL_TYPE,
    pad_id=PAD_ID,
    # input_sentence_size=INPUT_SENTENCE_SIZE,
)
