import math
import torch

def positional_embedding_1d(T, d):
    """
        PE(pos,2i) = sin(pos/10000^(2i/dmodel))
        PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))
    """
    if d % 2 != 0:
        raise ValueError("Embedding dimension must be divisible by 2")

    pe = torch.zeros(T, d)

    position = torch.arange(0, T).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d, 2) * -(math.log(10000.0) / d)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def positional_embedding_2d(H_p, W_p, d):
    #4 because, divisibility by 2 so that we can get 2i and 2i + 1
    #further divisibility by 2 to divide in height and width
    if d % 4 != 0:
        raise ValueError("Embedding dimension must be divisible by 4 for 2D pos embed")

    half = d // 2

    # (H_p, half) and (W_p, half)
    emb_h = positional_embedding_1d(H_p, half)
    emb_w = positional_embedding_1d(W_p, half)

    # broadcast over the grid: shape (H_p, W_p, half)
    emb_h = emb_h.unsqueeze(1).expand(-1, W_p, -1)  # (H_p, W_p, half)
    emb_w = emb_w.unsqueeze(0).expand(H_p, -1, -1)  # (H_p, W_p, half)

    # cat along last dim → (H_p, W_p, d), then flatten to (H_p*W_p, d)
    pe = torch.cat([emb_h, emb_w], dim=-1).reshape(H_p * W_p, d)

    return pe


def timestep_embedding(t, dim, max_period=10000):
    """
    t: (B,) list of time steps
    returns: (B, dim)
    """
    half = dim // 2

    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]

    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

    return emb