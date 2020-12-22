from typing import Union

import torch.nn as nn


class Seq2SeqRNNEncoder(nn.Module):
    """
    Seq2seq rnn encoder.
    """

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


class Seq2SeqRNNDecoder(nn.Module):
    """
    Seq2seq rnn decoder.
    """

    def __init__(
        self,
        decoder_num_embeddings: int,
        decoder_embedding_dim: int,
        decoder_hidden_size: int,
        decoder_num_layers: int,
        decoder_dropout: Union[int, float],
    ):
        super().__init__()
        self.decoder_embedding = nn.Embedding(
            num_embeddings=decoder_num_embeddings,
            embedding_dim=decoder_embedding_dim,
            padding_idx=3,
        )
        self.decoder = nn.GRU(
            input_size=decoder_embedding_dim,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
            dropout=decoder_dropout,
            bidirectional=False,
            batch_first=True,
        )
        self.linear = nn.Linear(
            in_features=decoder_hidden_size,
            out_features=decoder_num_embeddings,
        )

    def forward(self, x, enc_last):
        dec_emb = self.decoder_embedding(x)
        dec_out, _ = self.decoder(dec_emb, enc_last)
        logits = self.linear(dec_out)
        return logits


class Seq2SeqRNN(nn.Module):
    """
    Seq2seq rnn model.
    """

    def __init__(
        self,
        encoder_num_embeddings: int,
        encoder_embedding_dim: int,
        encoder_hidden_size: int,
        encoder_num_layers: int,
        encoder_dropout: Union[int, float],
        decoder_num_embeddings: int,
        decoder_embedding_dim: int,
        decoder_hidden_size: int,
        decoder_num_layers: int,
        decoder_dropout: Union[int, float],
    ):
        super().__init__()
        self.encoder = Seq2SeqRNNEncoder(
            encoder_num_embeddings=encoder_num_embeddings,
            encoder_embedding_dim=encoder_embedding_dim,
            encoder_hidden_size=encoder_hidden_size,
            encoder_num_layers=encoder_num_layers,
            encoder_dropout=encoder_dropout,
        )
        self.decoder = Seq2SeqRNNDecoder(
            decoder_num_embeddings=decoder_num_embeddings,
            decoder_embedding_dim=decoder_embedding_dim,
            decoder_hidden_size=decoder_hidden_size,
            decoder_num_layers=decoder_num_layers,
            decoder_dropout=decoder_dropout,
        )

    def forward(self, from_seq, to_seq):
        enc_last = self.encoder(from_seq)
        logits = self.decoder(to_seq, enc_last)
        return logits
