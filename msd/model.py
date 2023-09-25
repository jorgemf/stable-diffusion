import math

import torch
from torch import nn


class UNet(nn.Module):

    def __init__(self, image_size: int,
                 time_embeddings: int = 64,
                 time_emb_channels: int = 512,
                 text_emb_channels: int = 512,
                 num_layers: int = 2,
                 attention_heads: int = 8,
                 dtype: torch.dtype = torch.bfloat16):
        super(UNet, self).__init__()
        assert math.log2(image_size).is_integer(), "Image size must be a power of 2"
        assert image_size >= 64, "Image size must be at least 64"
        assert image_size < 2048, "Image size must be at most 2048"
        self.num_blocks = int(math.log2(image_size)) - 5
        self.num_layers = num_layers
        # calculate size of channels
        self.channels = [512, 512]
        emb_channels = time_emb_channels + text_emb_channels
        while len(self.channels) < self.num_blocks + 1:
            self.channels = self.channels + [self.channels[-1] // 2] * 2
        self.channels = self.channels[:self.num_blocks + 1]

        # transformation of the embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_embeddings, dtype=dtype),
            nn.Linear(time_embeddings, time_emb_channels, dtype=dtype),
            nn.GELU(),
            nn.Linear(time_emb_channels, time_emb_channels, dtype=dtype),
            nn.GELU()
        )

        self.down_blocks = []
        for i in range(self.num_blocks):
            use_attention = i != 0
            in_channels = self.channels[-i - 1]
            out_channels = self.channels[-i - 2]
            step_down = []
            for j in range(num_layers):
                if i == 0 and j == 0:
                    step_down.append(ResidualBlock(3, in_channels, emb_channels, dtype=dtype))
                else:
                    step_down.append(
                        ResidualBlock(in_channels, in_channels, emb_channels, dtype=dtype))
                if use_attention:
                    step_down.append(AttentionBlock(in_channels, attention_heads, dtype=dtype))
            step_down.append(DownSample(out_channels, dtype=dtype))
            self.down_blocks.append(nn.ModuleList(step_down))
        self.down_blocks = nn.ModuleList(self.down_blocks)
        mid_channels = self.channels[0]
        self.mid_blocks = nn.ModuleList([
            ResidualBlock(mid_channels, mid_channels, emb_channels, dtype=dtype),
            AttentionBlock(mid_channels, attention_heads, dtype=dtype),
            ResidualBlock(mid_channels, mid_channels, emb_channels, dtype=dtype)
        ])
        self.up_blocks = []
        for i in range(self.num_blocks):
            use_attention = i != self.num_blocks - 1
            last_block = i == self.num_blocks - 1
            in_channels = self.channels[i]
            out_channels = self.channels[i + 1]
            if last_block:
                out_channels = 3
            step_up = []
            step_up.append(UpSample(in_channels, dtype=dtype))
            for j in range(num_layers):
                if use_attention:
                    step_up.append(AttentionBlock(in_channels, dtype=dtype))
                if j != num_layers - 1:
                    step_up.append(
                        ResidualBlock(in_channels * 2, in_channels, emb_channels, dtype=dtype))
                else:
                    step_up.append(
                        ResidualBlock(in_channels * 2, out_channels, emb_channels, dtype=dtype))
            self.up_blocks.append(nn.ModuleList(step_up))
        self.up_blocks = nn.ModuleList(self.up_blocks)

    def forward(self, x, time, text_embedding):
        # calculate embeddings (positional+text)
        time_embedding = self.time_mlp(time)
        embedding = torch.cat((time_embedding, text_embedding), dim=1)

        residuals = []
        # down
        for i, blocks in enumerate(self.down_blocks):
            use_attention = i != 0
            j = 0
            for _ in range(self.num_layers):
                x = blocks[j](x, embedding)
                residuals.append(x)
                j += 1
                if use_attention:
                    x = blocks[j](x)
                    j += 1
            x = blocks[j](x)  # downsample
        # mid
        x = self.mid_blocks[0](x, embedding)
        x = self.mid_blocks[1](x)
        x = self.mid_blocks[2](x, embedding)
        # up
        for i, blocks in enumerate(self.up_blocks):
            use_attention = i != self.num_blocks - 1
            j = 0
            x = blocks[j](x)  # upsample
            j += 1
            for k in range(self.num_layers):
                if use_attention:
                    x = blocks[j](x)
                    j += 1
                residual = residuals.pop()
                x = torch.cat((x, residual), dim=1)
                x = blocks[j](x, embedding)
                j += 1
        return x


class SinusoidalPositionalEmbedding(nn.Module):

    def __init__(self, dim: int, dtype: torch.dtype = None):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.dim_half = torch.arange(dim // 2)
        emb = math.log(10000) / (dim // 2 - 1)
        emb = torch.tensor(emb)
        self.emb = torch.exp(self.dim_half * -emb).to(dtype=dtype)

    def forward(self, x):
        emb = self.emb.to(x.device)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 emb_channels: int,
                 num_groups: int = 32,
                 dtype: torch.dtype = None):
        super(ResidualBlock, self).__init__()
        num_groups_1 = num_groups if in_channels >= num_groups else in_channels
        self.block_1 = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups_1, num_channels=in_channels, affine=True,
                         dtype=dtype),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, dtype=dtype),
            nn.SiLU()
        )
        num_groups_2 = num_groups if out_channels >= num_groups else out_channels
        self.block_2 = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups_2, num_channels=out_channels, affine=True,
                         dtype=dtype),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False,
                      dtype=dtype),
            nn.SiLU()
        )
        self.output_scale_factor = math.sqrt(2)

        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1, dtype=dtype)

        self.emb_layer = nn.Linear(emb_channels, out_channels*2, dtype=dtype)

    def forward(self, x, embedding):
        emb = self.emb_layer(embedding)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)
        residual = x
        x = self.block_1(x) * scale + shift
        x = self.block_2(x)
        x = (x + self.skip_connection(residual)) / self.output_scale_factor
        return x


class AttentionBlock(nn.Module):

    def __init__(self, channels, num_head_channels=None, dtype: torch.dtype = None):
        super(AttentionBlock, self).__init__()
        self.channels = channels
        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.num_head_size = num_head_channels

        self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=32, affine=True,
                                       dtype=dtype)
        self.query = nn.Linear(channels, channels, dtype=dtype)
        self.key = nn.Linear(channels, channels, dtype=dtype)
        self.value = nn.Linear(channels, channels, dtype=dtype)

        self.proj_attn = nn.Linear(channels, channels, 1, dtype=dtype)
        self.rescale_output_factor = math.sqrt(2)

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.num_heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len,
                                                    dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.num_heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len,
                                                    dim * head_size)
        return tensor

    def forward(self, x):
        residual = x
        batch, channel, height, width = x.shape

        # norm
        x = self.group_norm(x)

        x = x.view(batch, channel, height * width).transpose(1, 2)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query_proj = self.reshape_heads_to_batch_dim(query)
        key_proj = self.reshape_heads_to_batch_dim(key)
        value_proj = self.reshape_heads_to_batch_dim(value)

        scale = 1 / math.sqrt(self.channels / self.num_heads)

        attention_scores = torch.baddbmm(
            torch.empty(
                query_proj.shape[0],
                query_proj.shape[1],
                key_proj.shape[1],
                dtype=query_proj.dtype,
                device=query_proj.device,
            ),
            query_proj,
            key_proj.transpose(-1, -2),
            beta=0,
            alpha=scale,
        )
        attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(
            attention_scores.dtype)
        x = torch.bmm(attention_probs, value_proj)

        # reshape hidden_states
        x = self.reshape_batch_dim_to_heads(x)

        # compute next hidden_states
        x = self.proj_attn(x)

        x = x.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        x = (x + residual) / self.rescale_output_factor
        return x


class DownSample(nn.Module):

    def __init__(self, channels, dtype: torch.dtype = None):
        super(DownSample, self).__init__()
        self.downsample = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1,
                                    dtype=dtype)

    def forward(self, x):
        return self.downsample(x)


class UpSample(nn.Module):

    def __init__(self, channels, dtype: torch.dtype = None):
        super(UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dtype=dtype)

    def forward(self, x):
        # TODO this is fixed in 2.1 nightly build of pytorch
        if x.dtype == torch.bfloat16:
            x = self.upsample(x.float()).type(torch.bfloat16)
        else:
            x = self.upsample(x)
        x = self.upsample_conv(x)
        return x
