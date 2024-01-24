import torch
import torch.nn as nn
from Layers import EncoderLayer, DecoderLayer
import copy
import sys

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, opt, channel, model_channel, N, heads):
        super().__init__()
        self.N = N
        self.model_channel = model_channel
        self.layers = get_clones(EncoderLayer(model_channel, heads), N)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, model_channel, 3, 2, 1, bias=False),
            nn.GroupNorm(1, model_channel),
            nn.SiLU(inplace=True),

        )

    def forward(self, x):
        x = self.conv(x.view(-1, *x.shape[2:])).view(x.size(0), x.size(1), self.model_channel, 16, 16)
        
        for i in range(self.N):
            x = self.layers[i](x)
        return x


class Decoder(nn.Module):
    def __init__(self, opt, model_channel, channel, N, heads):
        super().__init__()
        self.N = N
        self.model_channel = model_channel
        self.channel = channel
        self.layers = get_clones(DecoderLayer(model_channel, heads), N)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(model_channel, model_channel, 3, 2, 1, output_padding=1, bias=False),
            nn.GroupNorm(1, model_channel),
            nn.SiLU(inplace=True),

            nn.ConvTranspose2d(model_channel, channel, 1, 1, 0, bias=False),
        )

    def forward(self, e_outputs, x):
        outputs = []
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
            x1 = self.conv(x.view(-1, *x.shape[2:])).view(x.size(0), x.size(1), self.channel, 32, 32)
            outputs.append(x1)
        output = torch.cat(outputs, dim=2)

        return torch.sigmoid(output)


class Transformer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = Encoder(opt, opt.patch_size * opt.patch_size, opt.d_model, opt.n_encoder_layers, opt.heads)
        self.decoder = Decoder(opt, opt.d_model, opt.patch_size * opt.patch_size, opt.n_decoder_layers, opt.heads)

        self.conv = nn.Sequential(
            nn.Conv2d(opt.patch_size * opt.patch_size,  opt.d_model, 3, 2, 1, bias=False),
            nn.GroupNorm(1,  opt.d_model),
            nn.SiLU(inplace=True),

        )

        self.d_model = opt.d_model

    def forward(self, src):
        
        de_input = self.conv(src.view(-1, *src.shape[2:])).view(src.size(0), src.size(1), self.d_model, 16, 16)
        e_outputs = self.encoder(src)
        d_output = self.decoder(e_outputs, de_input)
        return d_output

def get_model(opt):
    model = Transformer(opt)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
