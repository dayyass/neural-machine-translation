import sentencepiece as spm

# train English
spm.SentencePieceTrainer.Train(
    input="../data/WMT14_English_German/train.en",
    model_prefix="en",
    vocab_size=50000,
    character_coverage=1.0,
    model_type="unigram",
)

# train German
spm.SentencePieceTrainer.Train(
    input="../data/WMT14_English_German/train.de",
    model_prefix="de",
    vocab_size=50000,
    character_coverage=1.0,
    model_type="unigram",
)
