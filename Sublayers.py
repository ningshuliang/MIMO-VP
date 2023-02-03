import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def attention_s(q, k, v):
    scores_s = torch.matmul(q.view(q.size(0), q.size(1), -1), k.view(k.size(0), k.size(1), -1).permute(0, 2, 1)) \
             / math.sqrt(q.size(2) * q.size(3) * q.size(4))
    # if q.size(1) > k.size(1):
    #     for i in range(k.size(1)):
    #         scores_s[:, i, i] = 0

    scores_s = F.softmax(scores_s, dim=-1)

    v_s = torch.matmul(scores_s, v.reshape(v.size(0), v.size(1), -1))

    output = v_s.reshape(q.size())
    return output

class MultiHeadAttention_S(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        self.d_model = d_model
        self.h = heads

        self.q_Conv = nn.Sequential(nn.Conv2d(self.d_model, self.d_model, 1, 1, 0, bias=False),
                                    nn.GroupNorm(1, self.d_model),
                                    )
        self.v_Conv = nn.Sequential(nn.Conv2d(self.d_model, self.d_model, 1, 1, 0, bias=False),
                                    nn.GroupNorm(1, self.d_model),
                                    )
        self.k_Conv = nn.Sequential(nn.Conv2d(self.d_model, self.d_model, 1, 1, 0, bias=False),
                                    nn.GroupNorm(1, self.d_model),
                                    )
        self.v_post_f = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, 1, 0, bias=False),
            nn.GroupNorm(1, d_model),
            nn.SiLU(inplace=True),
        )

    def forward(self, q, k, v):
        b_q, s_q, c_q, h_q, w_q = q.size()
        b_k, s_k, c_k, h_k, w_k = k.size()
        b_v, s_v, c_v, h_v, w_v = v.size()
        q = self.q_Conv(q.reshape(q.size(0) * q.size(1), *q.shape[2:])).reshape(q.size(0)*q.size(1), self.h,
                                                                                self.d_model // self.h, h_q, w_q)
        q = q.reshape(b_q, s_q, self.h, self.d_model // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        q = q.reshape(q.size(0)*q.size(1), *q.shape[2:])

        k = self.k_Conv(k.reshape(k.size(0) * k.size(1), *k.shape[2:])).reshape(k.size(0) * k.size(1), self.h,
                                                                                self.d_model // self.h, h_q, w_q)
        k = k.reshape(b_k, s_k, self.h, self.d_model // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        k = k.reshape(k.size(0) * k.size(1), *k.shape[2:])

        v = self.v_Conv(v.reshape(v.size(0) * v.size(1), *v.shape[2:])).reshape(v.size(0) * v.size(1), self.h,
                                                                                self.d_model // self.h, h_q, w_q)
        v = v.reshape(b_v, s_v, self.h, self.d_model // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        v = v.reshape(v.size(0) * v.size(1), *v.shape[2:])

        output = attention_s(q, k, v).reshape(b_q, self.h, s_q, self.d_model // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        output = self.v_post_f(output.reshape(b_q*s_q, self.h, self.d_model // self.h,
                                              h_q, w_q).reshape(b_q*s_q, self.d_model, h_q, w_q)).view(b_q, s_q, c_q, h_q, w_q)

        return output

class FeedForward11(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.ff1 = nn.Sequential(
            nn.Conv3d(d_model, 2 * d_model, [3,3,3], [2,1,1], 1, bias=False),
            nn.GroupNorm(1, 2 * d_model),
            nn.SiLU(inplace=True),
        )
        self.ff2 = nn.Sequential(
            nn.Conv3d(2 * d_model, d_model, 3, 1, 1, bias=False),
            nn.GroupNorm(1, d_model),
            nn.SiLU(inplace=True),
        )

    def forward(self, x, e_outputs, e_outputs1):
        XX = torch.cat([x, e_outputs], dim=1)
        x = self.ff2(self.ff1(XX.permute(0, 2, 1, 3, 4))).permute(0, 2, 1, 3, 4)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.ff1 = nn.Sequential(
            nn.Conv3d(d_model, 2 * d_model, 3, 1, 1, bias=False),
            nn.GroupNorm(1, 2 * d_model),
            nn.SiLU(inplace=True),
        )
        self.ff2 = nn.Sequential(
            nn.Conv3d(2 * d_model, d_model, 3, 1, 1, bias=False),
            nn.GroupNorm(1, d_model),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x = self.ff2(self.ff1(x.permute(0, 2, 1, 3, 4))).permute(0, 2, 1, 3, 4)
        return x

