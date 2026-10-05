import math
from typing import Any
import torch
import torch.nn as nn
from .positional import PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1, max_len=5000, pad_id=0, norm_first=True):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.src_embed = nn.Embedding(
            src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(
            tgt_vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout,norm_first)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout,norm_first)

        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def make_src_mask(self, src):
        # (B, S_src) -> (B, 1, 1, S_src)
        # from all heads, queries, no attention to pad
        return (src == self.pad_id).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        # (B, S_tgt) -> (B, 1, S_tgt, S_tgt)
        B, S = tgt.shape
        pad_mask = (tgt == self.pad_id).unsqueeze(
            1).unsqueeze(2)  # (B, 1, 1, S_tgt)
        causal_mask = torch.triu(torch.ones(
            S, S, device=tgt.device), diagonal=1).bool()  # (S_tgt,S_tgt)
        return pad_mask | causal_mask  # (B,1,S_tgt,S_tgt)

    def encode(self, src):
        # src: (B, S_src)
        src_mask = self.make_src_mask(src)  # (B, 1, 1, S_src)
        x = self.src_embed(src)*math.sqrt(self.d_model)  # (B, S_src, d_model)
        x = self.pos_encoding(x)

        # (B, S_src, d_model), (B, 1, 1, S_src)
        return self.encoder(x, src_mask), src_mask

    def decode(self, tgt, memory, src_mask):
        tgt_mask = self.make_tgt_mask(tgt)  # (B, 1, S_tgt, S_tgt)
        x = self.tgt_embed(tgt)*math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        # (B, S_tgt, d_model)
        return self.decoder(x, memory, src_mask, tgt_mask)

    def forward(self, src, tgt):
        memory, src_mask = self.encode(src)
        out = self.decode(tgt, memory, src_mask)  # (B, S_tgt, d_model)
        return self.generator(out)  # (B, S_tgt, tgt_vocab_size)


# model = Transformer(src_vocab_size=8000, tgt_vocab_size=5000, num_layers=2)
# src = torch.randint(0, 8000, (2, 10))
# tgt = torch.randint(0, 5000, (2, 8))
# logits = model(src, tgt)
# print(logits.shape)  # (2, 8, 5000)
