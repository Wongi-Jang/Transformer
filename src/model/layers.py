import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, S, d_model)
        # (B, S, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x+self.dropout(sublayer(self.norm(x)))


# ffn = FeedForward(d_model=512, d_ff=2048)
# sublayer = SublayerConnection(d_model=512)
# x = torch.randn(2, 10, 512)
# out = sublayer(x, ffn)
# print(out.shape)
