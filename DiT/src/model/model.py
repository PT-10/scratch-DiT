import torch
import torch.nn as nn
from model.position_embedding import positional_embedding_1d, positional_embedding_2d, timestep_embedding

class PatchEmbed(nn.Module):
    """
    The input to DiT is a spatial representation z
    (for 256 x 256 x 3 images, z has shape 32 x 32 x 4).
    """
    def __init__(self, input_dim: tuple, patch_size: int, embedding_dim: int, pos_embed: str = "2d"):
        super().__init__()
        C, H, W = input_dim

        self.p = patch_size
        self.d = embedding_dim

        assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch size"

        self.H_p = H // patch_size
        self.W_p = W // patch_size
        self.num_patches = (self.H_p) * (self.W_p)

        # B, C, H, W -> B, embedding_dim, H/p, W/p
        self.conv = nn.Conv2d(in_channels = C,
                                out_channels = self.d,
                                kernel_size = self.p,
                                stride = self.p,
                                padding = 0)

        if pos_embed == "2d":
            pe = positional_embedding_2d(self.H_p, self.W_p, self.d)
        elif pos_embed == "1d":
            pe = positional_embedding_1d(self.num_patches, self.d)
        else:
            raise ValueError(f"pos_embed must be '1d' or '2d', got '{pos_embed}'")
        self.register_buffer("pos_embed", pe.unsqueeze(0))  # (1,T,d)

    def forward(self, x):
        x = self.conv(x).flatten(2).transpose(1, 2) + self.pos_embed
        return x


class TimeEmbed(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.time_dim = 256

        self.mlp = nn.Sequential(
            nn.Linear(self.time_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, t):
        t_emb = timestep_embedding(t, self.time_dim)
        return self.mlp(t_emb)
    

class DiT_Block(nn.Module):
    """
    DiT block using AdaLN conditioning
    """

    def __init__(self, hidden_size, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6)
        )

        #zero init (from DiT paper)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, cond):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)

        # Attention block
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(h, h, h)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP block
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class DiT(nn.Module):
    def __init__(
        self,
        input_dim=(4, 32, 32),
        patch_size=2,
        hidden_size=512,
        depth=8,
        num_heads=8,
        num_classes=10,
        class_dropout_prob=0.1,
        pos_embed="2d",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob

        self.patch_embed = PatchEmbed(
            input_dim,
            patch_size,
            hidden_size,
            pos_embed=pos_embed,
        )

        self.time_embed = TimeEmbed(hidden_size)

        # +1 for CFG null class
        self.class_embed = nn.Embedding(
            num_classes + 1,
            hidden_size
        )

        self.blocks = nn.ModuleList([
            DiT_Block(hidden_size, num_heads)
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2)
        )
        nn.init.zeros_(self.final_adaLN_modulation[1].weight)
        nn.init.zeros_(self.final_adaLN_modulation[1].bias)

        # 2C output: predict noise and diagonal covariance
        out_channels = input_dim[0] * 2
        patch_dim = patch_size * patch_size * out_channels

        self.final_proj = nn.Linear(
            hidden_size,
            patch_dim
        )

        self.patch_size = patch_size
        self.input_dim = input_dim
        self.out_channels = out_channels

    def unpatchify(self, x):
        B, T, D = x.shape
        _, H, W = self.input_dim
        p = self.patch_size
        C = self.out_channels

        H_p = H // p
        W_p = W // p

        x = x.reshape(B, H_p, W_p, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, H, W)
        return x

    def forward(self, x, t, y):
        # classifier free guidance label dropout
        if self.training:
            drop_mask = torch.rand(y.shape, device=y.device) < self.class_dropout_prob
            y = y.clone()
            y[drop_mask] = self.num_classes

        x = self.patch_embed(x)
        t = self.time_embed(t)
        y = self.class_embed(y)
        cond = t + y

        for block in self.blocks:
            x = block(x, cond)

        shift, scale = self.final_adaLN_modulation(cond).chunk(2, dim=1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_proj(x)
        x = self.unpatchify(x)
        return x

def _dit(depth, patch_size, num_heads, **kwargs):
    return DiT(
        input_dim=(4, 32, 32),
        patch_size=patch_size,
        hidden_size=384,
        depth=depth,
        num_heads=num_heads,
        **kwargs,
    )

#_dit(depth, patch_size, num_heads)
def DiT_S_2(**kwargs):
    """DiT-Small/2 — official config from Peebles & Xie 2023."""
    return _dit(12, 2, 6, **kwargs)

#Depth Ablation (heads=6, patch=2)
def DiT_d4_h6_p2(**kwargs):
    return _dit(4, 2, 6, **kwargs)

def DiT_d6_h6_p2(**kwargs):
    return _dit(6, 2, 6, **kwargs)

def DiT_d8_h6_p2(**kwargs):
    return _dit(8, 2, 6, **kwargs)


#Head Ablation (depth=6, patch=2)
def DiT_d6_h1_p2(**kwargs):
    return _dit(6, 2, 1, **kwargs)

def DiT_d8_h1_p2(**kwargs):
    return _dit(8, 2, 1, **kwargs)

def DiT_d6_h2_p2(**kwargs):
    return _dit(6, 2, 2, **kwargs)

def DiT_d6_h4_p2(**kwargs):
    return _dit(6, 2, 4, **kwargs)


#Patch Ablation (depth=6, heads=6)
def DiT_d6_h6_p4(**kwargs):
    return _dit(6, 4, 6, **kwargs)

def DiT_d6_h6_p8(**kwargs):
    return _dit(6, 8, 6, **kwargs)


DiT_models = {
    # Baseline
    "DiT-d6-h6-p2": DiT_d6_h6_p2,
    # Depth
    "DiT-d4-h6-p2": DiT_d4_h6_p2,
    "DiT-d8-h6-p2": DiT_d8_h6_p2,
    # Heads
    "DiT-d6-h1-p2": DiT_d6_h1_p2,
    "DiT-d8-h1-p2": DiT_d8_h1_p2,
    "DiT-d6-h2-p2": DiT_d6_h2_p2,
    "DiT-d6-h4-p2": DiT_d6_h4_p2,
    # Patch
    "DiT-d6-h6-p4": DiT_d6_h6_p4,
    "DiT-d6-h6-p8": DiT_d6_h6_p8,
}