from typing import Union

import torch.nn as nn


class Seq2SeqRNNEncoder(nn.Module):
    def __init__(
        self,
        encoder_num_embeddings: int,
        encoder_embedding_dim: int,
        encoder_hidden_size: int,
        encoder_num_layers: int,
        encoder_dropout: Union[int, float],
    ):
        super().__init__()
        self.encoder_embedding = nn.Embedding(
            num_embeddings=encoder_num_embeddings,
            embedding_dim=encoder_embedding_dim,
            padding_idx=3,
        )
        self.encoder = nn.GRU(
            input_size=encoder_embedding_dim,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            dropout=encoder_dropout,
            bidirectional=False,
            batch_first=True,
        )

    def forward(self, x):
        enc_emb = self.encoder_embedding(x)
        _, enc_last = self.encoder(enc_emb)
        return enc_last
