import torch
from torch.utils.data import DataLoader

from dataset import WMTCollator, WMTDataset
from network import Seq2SeqRNNEncoder
from utils import set_global_seed

# hyper-parameters
SEED = 42
DEVICE = "cpu"

BATCH_SIZE = 1  # TODO: validate batch_size > 1
BUCKET_SEQUENCING_PERCENTILE = 100

ENCODER_EMBEDDING_DIM = 300
ENCODER_HIDDEN_SIZE = 300
ENCODER_NUM_LAYERS = 1  # TODO: validate num_layers > 1
ENCODER_DROPOUT = 0


device = torch.device(DEVICE)
set_global_seed(SEED)

# data
dataset = WMTDataset(
    from_lang_data_path="data/WMT14_English_German/train_tmp.en",
    to_lang_data_path="data/WMT14_English_German/train_tmp.de",
    from_lang_tokenizer_path="tokenizer/en.model",
    to_lang_tokenizer_path="tokenizer/de.model",
    verbose=True,
)

collator = WMTCollator(
    from_lang_padding_value=3,
    to_lang_padding_value=3,
    percentile=BUCKET_SEQUENCING_PERCENTILE,
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
)

# model
model = Seq2SeqRNNEncoder(
    encoder_num_embeddings=dataset.from_lang_tokenizer.vocab_size(),
    encoder_embedding_dim=ENCODER_EMBEDDING_DIM,
    encoder_hidden_size=ENCODER_HIDDEN_SIZE,
    encoder_num_layers=ENCODER_NUM_LAYERS,
    encoder_dropout=ENCODER_DROPOUT,
).to(device)

# inference
from_seq, to_seq = next(iter(dataloader))
from_seq, to_seq = from_seq.to(device), to_seq.to(device)
enc_last = model(from_seq)
