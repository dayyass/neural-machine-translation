import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import WMTCollator, WMTDataset
from language import Language
from network import Seq2SeqModel
from train_utils import train
from utils import set_global_seed

# path
INPUT_LANG_TRAIN_DATA_PATH = "data/IWSLT15_English_Vietnamese/train.en"
OUTPUT_LANG_TRAIN_DATA_PATH = "data/IWSLT15_English_Vietnamese/train.vi"
INPUT_LANG_VAL_DATA_PATH = "data/IWSLT15_English_Vietnamese/tst2012.en"
OUTPUT_LANG_VAL_DATA_PATH = "data/IWSLT15_English_Vietnamese/tst2012.vi"
INPUT_LANG_TEST_DATA_PATH = "data/IWSLT15_English_Vietnamese/tst2013.en"
OUTPUT_LANG_TEST_DATA_PATH = "data/IWSLT15_English_Vietnamese/tst2013.vi"

INPUT_LANG = "en"
OUTPUT_LANG = "vi"
INPUT_LANG_WORD2IDX_PATH = f"vocab/{INPUT_LANG}_vocab.json"
OUTPUT_LANG_WORD2IDX_PATH = f"vocab/{OUTPUT_LANG}_vocab.json"

SAVE_MODEL_PATH = "models/seq2seq.pth"

# hyper-parameters
SEED = 42
DEVICE = "cuda"
VERBOSE = True

UNK_ID = 0
BOS_ID = 1
EOS_ID = 2
PAD_ID = 3

BATCH_SIZE = 32
REVERSE_SOURCE_LANG = True
BUCKET_SEQUENCING_PERCENTILE = 100

ENCODER_EMBEDDING_DIM = DECODER_EMBEDDING_DIM = 500
ENCODER_HIDDEN_SIZE = DECODER_HIDDEN_SIZE = 500
ENCODER_NUM_LAYERS = DECODER_NUM_LAYERS = 2
ENCODER_DROPOUT = DECODER_DROPOUT = 0.2

N_EPOCH = 12
LEARNING_RATE = 1
TRAIN_EVAL_FREQ = 50  # number of batches


# print params
if VERBOSE:
    print("### PARAMETERS ###")
    print()
    print(f"INPUT_LANG_TRAIN_DATA_PATH: {INPUT_LANG_TRAIN_DATA_PATH}")
    print(f"OUTPUT_LANG_TRAIN_DATA_PATH: {OUTPUT_LANG_TRAIN_DATA_PATH}")
    print(f"INPUT_LANG_VAL_DATA_PATH: {INPUT_LANG_VAL_DATA_PATH}")
    print(f"OUTPUT_LANG_VAL_DATA_PATH: {OUTPUT_LANG_VAL_DATA_PATH}")
    print(f"INPUT_LANG_TEST_DATA_PATH: {INPUT_LANG_TEST_DATA_PATH}")
    print(f"OUTPUT_LANG_TEST_DATA_PATH: {OUTPUT_LANG_TEST_DATA_PATH}")
    print()
    print(f"INPUT_LANG: {INPUT_LANG}")
    print(f"OUTPUT_LANG: {OUTPUT_LANG}")
    print(f"INPUT_LANG_WORD2IDX_PATH: {INPUT_LANG_WORD2IDX_PATH}")
    print(f"OUTPUT_LANG_WORD2IDX_PATH: {OUTPUT_LANG_WORD2IDX_PATH}")
    print()
    print(f"SAVE_MODEL_PATH: {SAVE_MODEL_PATH}")
    print()
    print(f"SEED: {SEED}")
    print(f"DEVICE: {DEVICE}")
    print()
    print(f"UNK_ID: {UNK_ID}")
    print(f"BOS_ID: {BOS_ID}")
    print(f"EOS_ID: {EOS_ID}")
    print(f"PAD_ID: {PAD_ID}")
    print()
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"REVERSE_SOURCE_LANG: {REVERSE_SOURCE_LANG}")
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


# language
input_language = Language(
    language=INPUT_LANG,
    path_to_word2idx=INPUT_LANG_WORD2IDX_PATH,
    unk_id=UNK_ID,
    bos_id=BOS_ID,
    eos_id=EOS_ID,
)
output_language = Language(
    language=OUTPUT_LANG,
    path_to_word2idx=OUTPUT_LANG_WORD2IDX_PATH,
    unk_id=UNK_ID,
    bos_id=BOS_ID,
    eos_id=EOS_ID,
)


# data
train_dataset = WMTDataset(
    input_lang_data_path=INPUT_LANG_TRAIN_DATA_PATH,
    output_lang_data_path=OUTPUT_LANG_TRAIN_DATA_PATH,
    input_language=input_language,
    output_language=output_language,
    reverse_source_lang=REVERSE_SOURCE_LANG,
    verbose=VERBOSE,
)
val_dataset = WMTDataset(
    input_lang_data_path=INPUT_LANG_VAL_DATA_PATH,
    output_lang_data_path=OUTPUT_LANG_VAL_DATA_PATH,
    input_language=input_language,
    output_language=output_language,
    reverse_source_lang=REVERSE_SOURCE_LANG,
    verbose=VERBOSE,
)
test_dataset = WMTDataset(
    input_lang_data_path=INPUT_LANG_TEST_DATA_PATH,
    output_lang_data_path=OUTPUT_LANG_TEST_DATA_PATH,
    input_language=input_language,
    output_language=output_language,
    reverse_source_lang=REVERSE_SOURCE_LANG,
    verbose=VERBOSE,
)

train_collator = WMTCollator(
    input_lang_pad_id=PAD_ID,
    output_lang_pad_id=PAD_ID,
    percentile=BUCKET_SEQUENCING_PERCENTILE,
)
test_collator = WMTCollator(  # same for val_loader
    input_lang_pad_id=PAD_ID,
    output_lang_pad_id=PAD_ID,
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
model = Seq2SeqModel(
    encoder_num_embeddings=len(input_language.word2idx),
    encoder_embedding_dim=ENCODER_EMBEDDING_DIM,
    encoder_hidden_size=ENCODER_HIDDEN_SIZE,
    encoder_num_layers=ENCODER_NUM_LAYERS,
    encoder_dropout=ENCODER_DROPOUT,
    decoder_num_embeddings=len(output_language.word2idx),
    decoder_embedding_dim=DECODER_EMBEDDING_DIM,
    decoder_hidden_size=DECODER_HIDDEN_SIZE,
    decoder_num_layers=DECODER_NUM_LAYERS,
    decoder_dropout=DECODER_DROPOUT,
).to(device)

if VERBOSE:
    print(f"model number of parameters: {sum(p.numel() for p in model.parameters())}")


# criterion, optimizer, scheduler
criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_ID,
)
optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
)
scheduler = optim.lr_scheduler.MultiStepLR(  # hardcoded
    optimizer,
    milestones=[8, 9, 10, 11],
    gamma=0.5,
)


# train
train(
    model=model,
    trainloader=train_loader,
    valloader=val_loader,
    testloader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    n_epoch=N_EPOCH,
    train_eval_freq=TRAIN_EVAL_FREQ,
    device=device,
    verbose=VERBOSE,
)


# save
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), SAVE_MODEL_PATH)
