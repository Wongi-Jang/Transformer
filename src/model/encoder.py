import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .layers import FeedForward, SublayerConnection


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1,norm_first=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout,norm_first)
        self.sublayer2 = SublayerConnection(d_model, dropout,norm_first)

    def forward(self, x, src_mask=None):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, src_mask)[0])
        x = self.sublayer2(x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1,norm_first=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout,norm_first) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# enc = Encoder(d_model=512, num_heads=8, d_ff=2048, num_layers=6)
# x = torch.randn(2, 10, 512)
# out = enc(x)
# print(out.shape)
