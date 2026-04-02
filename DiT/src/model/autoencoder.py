import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # asymmetric padding to match LDM: pad right & bottom by 1, then stride-2 conv with no padding
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)

        self.nin_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        k = k.reshape(b, c, h * w)
        # attn: (b, hw, hw) — query positions attend to key positions
        attn = torch.einsum("bci,bcj->bij", q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=2)

        v = v.reshape(b, c, h * w)
        h_ = torch.einsum("bij,bcj->bci", attn, v)
        h_ = h_.reshape(b, c, h, w)
        return x + self.proj_out(h_)


class Encoder(nn.Module):
    def __init__(self, *, ch, ch_mult, num_res_blocks, in_channels,
                 z_channels, double_z=True, attn_resolutions=(), dropout=0.0,
                 resolution=256, **kwargs):
        super().__init__()
        self.num_resolutions = len(ch_mult)

        # input conv
        self.conv_in = nn.Conv2d(in_channels, ch, 3, stride=1, padding=1)

        # downsampling
        in_ch = ch
        curr_res = resolution
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            out_ch = ch * ch_mult[i_level]
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(in_ch, out_ch, dropout=dropout))
                in_ch = out_ch
            level = nn.Module()
            level.block = block
            if i_level != self.num_resolutions - 1:
                level.downsample = Downsample(in_ch)
                curr_res //= 2
            self.down.append(level)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_ch, in_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(in_ch)
        self.mid.block_2 = ResnetBlock(in_ch, in_ch, dropout=dropout)

        # output
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6, affine=True)
        out_z = 2 * z_channels if double_z else z_channels
        self.conv_out = nn.Conv2d(in_ch, out_z, 3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)

        # down
        for i_level in range(self.num_resolutions):
            for block in self.down[i_level].block:
                h = block(h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)

        # mid
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # out
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, ch_mult, num_res_blocks, out_ch,
                 z_channels, attn_resolutions=(), dropout=0.0,
                 resolution=256, **kwargs):
        super().__init__()
        self.num_resolutions = len(ch_mult)

        # compute channel sizes at each level (bottom=highest mult → top=lowest)
        block_in = ch * ch_mult[-1]
        curr_res = resolution // (2 ** (self.num_resolutions - 1))

        # input conv (from z_channels)
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout=dropout)

        # upsampling — level 0 is lowest resolution, level num_res-1 is highest
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block_out = ch * ch_mult[i_level]
            block_list = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                block_list.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
            level = nn.Module()
            level.block = block_list
            if i_level != 0:
                level.upsample = Upsample(block_in)
                curr_res *= 2
            self.up.insert(0, level)  # insert at front so index 0 = lowest res

        # output
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)

        # mid
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # up (traverse from highest level down to 0)
        for i_level in reversed(range(self.num_resolutions)):
            for block in self.up[i_level].block:
                h = block(h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        # out
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h

class DiagonalGaussianDistribution:
    def __init__(self, moments):
        self.mean, self.logvar = torch.chunk(moments, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self):
        return self.mean

    def kl(self):
        return 0.5 * (self.mean.pow(2) + self.std.pow(2) - self.logvar - 1).sum(dim=[1, 2, 3])


class AutoencoderKL(nn.Module):
    def __init__(self, ddconfig, embed_dim, ckpt_path=None):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        # filter out loss / discriminator keys
        sd = {k: v for k, v in sd.items() if not k.startswith("loss.")}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored AutoencoderKL from {path}")
        if missing:
            print(f"  missing keys: {missing}")
        if unexpected:
            print(f"  unexpected keys: {unexpected}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)
        return dec, posterior
