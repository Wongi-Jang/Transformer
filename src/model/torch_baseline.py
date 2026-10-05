import math
import torch
import torch.nn as nn
from .positional import PositionalEncoding


class TorchTransformer(nn.Module):
    # same embed / PE / generator as Transformer; encoder+decoder from nn.Transformer (Pre-LN)
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1, max_len=5000, pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.src_embed = nn.Embedding(
            src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(
            tgt_vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads,
                                          num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                                          dim_feedforward=d_ff, dropout=dropout,
                                          batch_first=True, norm_first=True)
        # my MultiHeadAttention has no dropout on attention weights; match it
        for m in self.transformer.modules():
            if isinstance(m, nn.MultiheadAttention):
                m.dropout = 0.0
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src):
        src_pad = src == self.pad_id  # (B, S_src), True = ignore
        x = self.pos_encoding(self.src_embed(src)*math.sqrt(self.d_model))
        return self.transformer.encoder(x, src_key_padding_mask=src_pad), src_pad

    def decode(self, tgt, memory, src_pad):
        S = tgt.shape[1]
        causal = torch.triu(torch.ones(
            S, S, dtype=torch.bool, device=tgt.device), diagonal=1)
        x = self.pos_encoding(self.tgt_embed(tgt)*math.sqrt(self.d_model))
        return self.transformer.decoder(x, memory, tgt_mask=causal,
                                        tgt_key_padding_mask=tgt == self.pad_id,
                                        memory_key_padding_mask=src_pad)

    def forward(self, src, tgt):
        memory, src_pad = self.encode(src)
        return self.generator(self.decode(tgt, memory, src_pad))
