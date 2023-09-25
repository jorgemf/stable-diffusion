import torch
import torch.nn as nn


class Upsample(nn.Module):

    def __init__(self, channels, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.Conv2d(2, self.channels, self.out_channels, 3, padding=1, dtype=self.dtype)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x



class Downsample(nn.Module):

    def __init__(self, channels,  out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=1, dtype=self.dtype)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            activation,
            out_channels=None,
            use_scale_shift_norm=False,
            up=False,
            down=False,
            efficient_activation=False,
            scale_skip_connection=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.efficient_activation = efficient_activation
        self.scale_skip_connection = scale_skip_connection

        self.in_layers = nn.Sequential(
            normalization(channels, dtype=self.dtype),
            get_activation(activation),
            conv_nd(dims, channels, self.out_channels, 3, padding=1, dtype=self.dtype),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.Identity() if self.efficient_activation else get_activation(activation),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                dtype=self.dtype
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, dtype=self.dtype),
            get_activation(activation),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, dtype=self.dtype)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1, dtype=self.dtype)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1, dtype=self.dtype)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        res = self.skip_connection(x) + h
        if self.scale_skip_connection:
            res *= 0.7071  # 1 / sqrt(2), https://arxiv.org/pdf/2104.07636.pdf
        return res