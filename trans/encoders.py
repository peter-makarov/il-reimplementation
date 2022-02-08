"""Encoder classes used by the Transducer model."""

from typing import Any, Dict, List, Optional, TextIO, Union
import abc
import math


import torch


class LSTMEncoder(torch.nn.LSTM):
    def __init__(self, dargs: dict):
        super().__init__(
            input_size=dargs['char_dim'],
            hidden_size=dargs['enc_hidden_dim'],
            num_layers=dargs['enc_layers'],
            bidirectional=dargs['enc_bidirectional'],
            dropout=dargs['enc_dropout'],
            device=dargs['device']
        )

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument("--enc-hidden-dim", type=int, default=200,
                            help="Encoder LSTM state dimension.")
        parser.add_argument("--enc-layers", type=int, default=1,
                            help="Number of encoder LSTM layers.")
        parser.add_argument("--enc-bidirectional", type=bool, default=True,
                            help="If LSTM is bidirectional.")
        parser.add_argument("--enc-dropout", type=float, default=0.,
                            help="Dropout probability after each LSTM layer"
                                 "(except the last layer).")


    @property
    def output_size(self):
        return self.hidden_size * 2 if self.bidirectional else self.hidden_size


class TransformerEncoder(torch.nn.Module):
    def __init__(self, dargs: dict):
        super().__init__()
        self.d_model = dargs['char_dim']
        self.pos_encoding = PositionalEncoding(
            d_model=dargs['char_dim'],
            dropout=dargs['enc_dropout']
        )
        self.encoder_layers = torch.nn.TransformerEncoderLayer(
            d_model=dargs['char_dim'],
            nhead=dargs['enc_nhead'],
            dim_feedforward=dargs['enc_dim_feedforward'],
            dropout=dargs['enc_dropout'],
            device=dargs['device']
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer=self.encoder_layers,
            num_layers=dargs['enc_layers']
        )

    def forward(self, src, mask=None, src_key_padding_mask=None):
        pos_encoded = self.pos_encoding(src * math.sqrt(self.d_model))
        return self.transformer_encoder(pos_encoded, mask, src_key_padding_mask), None

    @property
    def output_size(self):
        return self.d_model

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument("--enc-layers", type=int, default=4,
                            help="Number of Transformer encoder layers.")
        parser.add_argument("--enc-nhead", type=int, default=4,
                            help="Number of Transformer heads.")
        parser.add_argument("--enc-dim-feedforward", type=int, default=1024,
                            help="Number of Transformer heads.")
        parser.add_argument("--enc-dropout", type=float, default=0.1,
                            help="Dropout probability.")


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 150):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


ENCODER_MAPPING = {
    'lstm': LSTMEncoder,
    'transformer': TransformerEncoder,
}
