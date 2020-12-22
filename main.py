import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import WMTCollator, WMTDataset
from network import Seq2SeqRNN
from train import train
from utils import set_global_seed

# path
FROM_LANG_TRAIN_DATA_PATH = "data/IWSLT15_English_Vietnamese/train.en"
TO_LANG_TRAIN_DATA_PATH = "data/IWSLT15_English_Vietnamese/train.vi"
FROM_LANG_VAL_DATA_PATH = "data/IWSLT15_English_Vietnamese/tst2012.en"
TO_LANG_VAL_DATA_PATH = "data/IWSLT15_English_Vietnamese/tst2012.vi"
FROM_LANG_TEST_DATA_PATH = "data/IWSLT15_English_Vietnamese/tst2013.en"
TO_LANG_TEST_DATA_PATH = "data/IWSLT15_English_Vietnamese/tst2013.vi"

FROM_LANG_TOKENIZER_PATH = "tokenizer/en.model"
TO_LANG_TOKENIZER_PATH = "tokenizer/vi.model"

SAVE_MODEL_PATH = "models/seq2seq.pth"

# hyper-parameters
SEED = 42
DEVICE = "cuda"
VERBOSE = True

PAD_ID = 3
BATCH_SIZE = 128
BUCKET_SEQUENCING_PERCENTILE = 95

ENCODER_EMBEDDING_DIM = DECODER_EMBEDDING_DIM = 128
ENCODER_HIDDEN_SIZE = DECODER_HIDDEN_SIZE = 256
ENCODER_NUM_LAYERS = DECODER_NUM_LAYERS = 2
ENCODER_DROPOUT = DECODER_DROPOUT = 0.2

N_EPOCH = 15
LEARNING_RATE = 1e-3
TRAIN_EVAL_FREQ = 50  # number of batches


# print params
if VERBOSE:
    print("### PARAMETERS ###")
    print()
    print(f"FROM_LANG_TRAIN_DATA_PATH: {FROM_LANG_TRAIN_DATA_PATH}")
    print(f"TO_LANG_TRAIN_DATA_PATH: {TO_LANG_TRAIN_DATA_PATH}")
    print(f"FROM_LANG_VAL_DATA_PATH: {FROM_LANG_VAL_DATA_PATH}")
    print(f"TO_LANG_VAL_DATA_PATH: {TO_LANG_VAL_DATA_PATH}")
    print(f"FROM_LANG_TEST_DATA_PATH: {FROM_LANG_TEST_DATA_PATH}")
    print(f"TO_LANG_TEST_DATA_PATH: {TO_LANG_TEST_DATA_PATH}")
    print()
    print(f"FROM_LANG_TOKENIZER_PATH: {FROM_LANG_TOKENIZER_PATH}")
    print(f"TO_LANG_TOKENIZER_PATH: {TO_LANG_TOKENIZER_PATH}")
    print()
    print(f"SEED: {SEED}")
    print(f"DEVICE: {DEVICE}")
    print()
    print(f"PAD_ID: {PAD_ID}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"BUCKET_SEQUENCING_PERCENTILE: {BUCKET_SEQUENCING_PERCENTILE}")
    print()
    print(f"ENCODER/DECODER_EMBEDDING_DIM: {ENCODER_EMBEDDING_DIM}")
    print(f"ENCODER/DECODER_HIDDEN_SIZE: {ENCODER_HIDDEN_SIZE}")
    print(f"ENCODER/DECODER_NUM_LAYERS: {ENCODER_NUM_LAYERS}")
    print(f"ENCODER/DECODER_DROPOUT: {ENCODER_DROPOUT}")
    print()
    print(f"N_EPOCH: {N_EPOCH}")
    print(f"LEARNING_RATE: {LEARNING_RATE}")
    print(f"TRAIN_EVAL_FREQ: {TRAIN_EVAL_FREQ}")
    print()


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
val_dataset = WMTDataset(
    from_lang_data_path=FROM_LANG_VAL_DATA_PATH,
    to_lang_data_path=TO_LANG_VAL_DATA_PATH,
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
    from_lang_pad_id=PAD_ID,
    to_lang_pad_id=PAD_ID,
    percentile=BUCKET_SEQUENCING_PERCENTILE,
)
test_collator = WMTCollator(
    from_lang_pad_id=PAD_ID,
    to_lang_pad_id=PAD_ID,
    percentile=100,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=train_collator,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=test_collator,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
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
train(
    model=model,
    trainloader=train_loader,
    valloader=val_loader,
    testloader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    n_epoch=N_EPOCH,
    train_eval_freq=TRAIN_EVAL_FREQ,
    verbose=VERBOSE,
)


# save
torch.save(model.state_dict(), SAVE_MODEL_PATH)
