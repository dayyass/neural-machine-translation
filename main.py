import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import WMTCollator, WMTDataset
from network import Seq2SeqRNN
from train import train_epoch
from utils import set_global_seed

# path
FROM_LANG_TRAIN_DATA_PATH = "data/IWSLT15_English_Vietnamese/train.en"
TO_LANG_TRAIN_DATA_PATH = "data/IWSLT15_English_Vietnamese/train.vi"
FROM_LANG_TEST_DATA_PATH = "data/IWSLT15_English_Vietnamese/tst2012.en"
TO_LANG_TEST_DATA_PATH = "data/IWSLT15_English_Vietnamese/tst2012.vi"
FROM_LANG_TOKENIZER_PATH = "tokenizer/en.model"
TO_LANG_TOKENIZER_PATH = "tokenizer/vi.model"

# hyper-parameters
SEED = 42
DEVICE = "cpu"
VERBOSE = True

PAD_ID = 3
BATCH_SIZE = 1  # TODO: validate batch_size > 1
BUCKET_SEQUENCING_PERCENTILE = 100

ENCODER_EMBEDDING_DIM = 100
ENCODER_HIDDEN_SIZE = 100
ENCODER_NUM_LAYERS = 1  # TODO: validate num_layers > 1
ENCODER_DROPOUT = 0

DECODER_EMBEDDING_DIM = 100
DECODER_HIDDEN_SIZE = 100
DECODER_NUM_LAYERS = 1  # TODO: validate num_layers > 1
DECODER_DROPOUT = 0

LEARNING_RATE = 1e-3


# seed and device
set_global_seed(SEED)
device = torch.device(DEVICE)


# tokenizer
from_lang_tokenizer = spm.SentencePieceProcessor(
    model_file=FROM_LANG_TOKENIZER_PATH,
)
to_lang_tokenizer = spm.SentencePieceProcessor(
    model_file=TO_LANG_TOKENIZER_PATH,
)


# data
train_dataset = WMTDataset(
    from_lang_data_path=FROM_LANG_TRAIN_DATA_PATH,
    to_lang_data_path=TO_LANG_TRAIN_DATA_PATH,
    from_lang_tokenizer=from_lang_tokenizer,
    to_lang_tokenizer=to_lang_tokenizer,
    verbose=VERBOSE,
)
test_dataset = WMTDataset(
    from_lang_data_path=FROM_LANG_TEST_DATA_PATH,
    to_lang_data_path=TO_LANG_TEST_DATA_PATH,
    from_lang_tokenizer=from_lang_tokenizer,
    to_lang_tokenizer=to_lang_tokenizer,
    verbose=VERBOSE,
)

train_collator = WMTCollator(
    from_lang_padding_value=PAD_ID,
    to_lang_padding_value=PAD_ID,
    percentile=BUCKET_SEQUENCING_PERCENTILE,
)
test_collator = WMTCollator(
    from_lang_padding_value=PAD_ID,
    to_lang_padding_value=PAD_ID,
    percentile=100,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=train_collator,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,  # TODO: fix batch_size
    shuffle=False,
    collate_fn=test_collator,
)


# model
model = Seq2SeqRNN(
    encoder_num_embeddings=train_dataset.from_lang_tokenizer.vocab_size(),
    encoder_embedding_dim=ENCODER_EMBEDDING_DIM,
    encoder_hidden_size=ENCODER_HIDDEN_SIZE,
    encoder_num_layers=ENCODER_NUM_LAYERS,
    encoder_dropout=ENCODER_DROPOUT,
    decoder_num_embeddings=train_dataset.to_lang_tokenizer.vocab_size(),
    decoder_embedding_dim=DECODER_EMBEDDING_DIM,
    decoder_hidden_size=DECODER_HIDDEN_SIZE,
    decoder_num_layers=DECODER_NUM_LAYERS,
    decoder_dropout=DECODER_DROPOUT,
).to(device)

if VERBOSE:
    print(f"model number of parameters: {sum(p.numel() for p in model.parameters())}")


# criterion and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# train
train_epoch(
    model=model,
    dataloader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    verbose=VERBOSE,
)
