import torch.nn as nn
from Sublayers import FeedForward, MultiHeadAttention_S


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.norm_1 = nn.GroupNorm(1, d_model)
        self.norm_2 = nn.GroupNorm(1, d_model)
        self.attn_1 = MultiHeadAttention_S(heads, d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x):
        b, s, c, h, w = x.size()

        x = x + self.attn_1(x, x, x)
        x = self.norm_1(x.view(-1, c, h, w)).view(b, s, c, h, w)

        x = x + self.ff(x)
        x = self.norm_2(x.view(-1, c, h, w)).view(b, s, c, h, w)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.norm_1 = nn.GroupNorm(1, d_model)
        self.norm_2 = nn.GroupNorm(1, d_model)
        self.norm_3 = nn.GroupNorm(1, d_model)

        self.attn_1 = MultiHeadAttention_S(heads, d_model)
        self.attn_2 = MultiHeadAttention_S(heads, d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x, e_outputs):
        b, s, c, h, w = x.size()

        x = x + self.attn_1(x, x, x)
        x = self.norm_1(x.view(-1, c, h, w)).view(b, s, c, h, w)

        x = x + self.attn_2(x, e_outputs, e_outputs)
        x = self.norm_2(x.view(-1, c, h, w)).view(b, s, c, h, w)

        x = x + self.ff(x)
        x = self.norm_3(x.view(-1, c, h, w)).view(b, s, c, h, w)

        return x
