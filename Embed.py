import torch
import torch.nn as nn
import math


class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3),
                      stride=stride, padding=1),
            nn.GroupNorm(1, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if (stride == 2):
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3),
                               stride=stride, padding=1, output_padding=output_padding),
            nn.GroupNorm(1, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class En_Embedder(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.output_channel = output_channel
        self.input_channel = input_channel
        self.c1 = dcgan_conv(input_channel, output_channel//4, stride=2)
        self.c2 = dcgan_conv(output_channel//4, output_channel//8, stride=2)
        self.c3 = dcgan_conv(output_channel//8, output_channel, stride=2)

    def forward(self, input):
        b, s, c, h, w = input.size()
        output = self.c3(self.c2(self.c1(input.view(-1, c, h, w)))).view(b, s, self.output_channel, 8, 8)
        # output = self.c2(self.c1(input.view(-1, c, h, w))).view(b, s, self.output_channel, 16, 16)

        return output

class De_Embedder(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.output_channel = output_channel
        self.c1 = dcgan_upconv(input_channel, input_channel // 8, stride=2)
        self.c2 = dcgan_upconv(input_channel//8, input_channel // 4, stride=2)
        self.c3 = nn.ConvTranspose2d(input_channel // 4, output_channel, kernel_size=(3, 3), stride=2, padding=1,output_padding=1)
        # self.c3 = nn.ConvTranspose2d(input_channel // 4, output_channel, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, input):
        b, s, c, h, w = input.size()
        output = self.c3(self.c2(self.c1(input.view(-1, c, h, w)))).view(b, s, self.output_channel, 64, 64)

        return torch.sigmoid(output)


class PositionalEncoder(nn.Module):
    def __init__(self, opt, d_model, max_seq_len=20, h_w=8):
        super().__init__()
        self.d_model = d_model
        self.batch = opt.batch_size
        self.h_w = h_w
        self.s = 10
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, h_w, h_w)
        for pos in range(max_seq_len):
            for i in range(0, h_w, 2):
                for j in range(0, h_w, 2):
                    pe[pos, i, j] = math.sin(pos / (10000 ** ((2 * i) / d_model))) + \
                                    math.sin(pos / (10000 ** ((2 * j) / d_model)))
                    pe[pos, i, j + 1] = math.sin(pos / (10000 ** ((2 * i) / d_model))) + \
                                        math.cos(pos / (10000 ** ((2 * j + 1) / d_model)))
                    pe[pos, i + 1, j] = math.sin(pos / (10000 ** ((2 * i + 1) / d_model))) + \
                                        math.cos(pos / (10000 ** ((2 * j) / d_model)))
                    pe[pos, i + 1, j + 1] = math.cos(pos / (10000 ** ((2 * i + 1) / d_model))) + \
                                            math.cos(pos / (10000 ** ((2 * j + 1) / d_model)))

        pe = pe.unsqueeze(0)
        pe = pe.unsqueeze(2)
        self.register_buffer('pe', pe)

    def forward(self, x = None):
        # make embeddings relatively larger
        if x is not None:
            # x = x * math.sqrt(self.d_model*self.h_w*self.h_w)
            # add constant to embedding
            pe = self.pe[:, :10]  # 1, 10, 16, 1024
            x = pe.expand(x.size())
        else:
            pe = self.pe[:, 10:]  # 1, 10, 16, 1024
            x = pe.expand([self.batch, self.s, self.d_model, self.h_w, self.h_w])
        return x.cuda() if x.is_cuda else x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=32, num_frames=10, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.frames = num_frames
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, shape):
        b, s, h, w, c = shape
        # mask = torch.ones((b, s, h, w), device=x.device)
        mask = torch.ones((b, s, h, w))

        z_embed = mask.cumsum(1, dtype=torch.float32)
        y_embed = mask.cumsum(2, dtype=torch.float32)
        x_embed = mask.cumsum(3, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos = torch.cat((pos_z, pos_y, pos_x), dim=4)

        return pos