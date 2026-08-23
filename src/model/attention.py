import math
import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v: (B, num_heads, S, d_k)
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # (B, num_heads, S_q, S_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)  # (B, num_heads, S_q, S_k)
    output = torch.matmul(attn_weights, v)  # (B, num_heads, S_q, d_k)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model//num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        # (B, S, d_model) -> (B, num_heads, S, d_k)
        B, S, _ = x.shape
        return x.reshape(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        # (B, num_heads, S, d_k) -> (B, S, d_model)
        B, H, S, d_k = x.shape
        return x.transpose(1, 2).reshape(B, S, H*d_k)

    def forward(self, query, key, value, mask=None):
        q = self._split_heads(self.w_q(query))  # (B, num_heads, S, d_k)
        k = self._split_heads(self.w_k(key))  # (B, num_heads, S, d_k)
        v = self._split_heads(self.w_v(value))  # (B, num_heads, S, d_k)

        out, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)
        out = self._merge_heads(out)
        return self.w_out(out), attn_weights


# mha = MultiHeadAttention(d_model=512, num_heads=8)
# x = torch.rand(2, 10, 512)
# out, attn = mha(x, x, x)
# print(out.shape) (2, 10, 512)
# print(attn.shape) (2, 8, 10, 10)
