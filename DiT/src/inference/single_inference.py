import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.model import DiT
from training.training import VAE_SCALE, build_vae
from diffusers import DDIMScheduler


CKPT        = "training-checkpoints/d8-h1-p2/dit_step0060000.pt"
VAE_CKPT    = "pretrained-models/kl-f8-model.ckpt"
CLASS_LABEL = 4          # change to whichever class index you want
CFG_SCALE   = 4.0
NUM_STEPS   = 100
SEED        = 42
DEVICE      = "cuda"


def _load_config_csv():
    import csv
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training-checkpoints/model_configs.csv"))
    table = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            table[row["checkpoint_dir"].rstrip("/")] = row
    return table


def infer_config(sd, raw, ckpt_path):
    """Read model hyperparameters from state dict tensor shapes."""
    hidden_size = sd["time_embed.mlp.0.weight"].shape[0]
    patch_size  = sd["patch_embed.conv.weight"].shape[2]
    num_classes = sd["class_embed.weight"].shape[0] - 1
    depth       = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".adaLN_modulation.1.bias"))

    if "heads" in raw:
        num_heads = raw["heads"]
    else:
        table = _load_config_csv()
        ckpt_abs = os.path.abspath(ckpt_path)
        match = next(
            (row for dir_key, row in table.items()
             if ckpt_abs.startswith(os.path.abspath(dir_key))),
            None,
        )
        if match is None:
            raise RuntimeError(
                f"No entry in model_configs.csv covers '{ckpt_path}'. "
                "Add a row for this checkpoint directory."
            )
        num_heads = int(match["num_heads"])
        print(f"num_heads={num_heads} (from model_configs.csv — '{match['checkpoint_dir']}')")

    return dict(
        input_dim=(4, 32, 32),
        patch_size=patch_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        num_classes=num_classes,
        class_dropout_prob=0.1,
    )


def load_mydit(ckpt_path, device):
    raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd  = raw["ema"] if "ema" in raw else (raw["model"] if "model" in raw else raw)

    cfg = infer_config(sd, raw, ckpt_path)
    print(f"Inferred config: {cfg}")

    model = DiT(**cfg).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded  ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)"
          f"  epoch={raw.get('epoch', '?')}  step={raw.get('step', '?')}")
    return model


@torch.no_grad()
def sample_with_cfg(model, scheduler, class_label, cfg_scale, device, seed):
    torch.manual_seed(seed)
    C, H, W = model.input_dim
    latents     = torch.randn(1, C, H, W, device=device)
    label_cond  = torch.tensor([class_label],       dtype=torch.long, device=device)
    label_uncond = torch.tensor([model.num_classes], dtype=torch.long, device=device)

    scheduler.set_timesteps(NUM_STEPS)

    for t in scheduler.timesteps:
        t_in = t.reshape(1).to(device)

        noise_cond   = model(latents, t_in, label_cond)[:, :C]
        noise_uncond = model(latents, t_in, label_uncond)[:, :C]
        guided = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

        latents = scheduler.step(guided, t, latents).prev_sample

    return latents


def main():
    torch.manual_seed(SEED)
    device = torch.device(DEVICE)

    model = load_mydit(CKPT, device)
    vae   = build_vae(VAE_CKPT, device)

    scheduler = DDIMScheduler(
        num_train_timesteps = 1000,
        beta_start          = 0.0001,
        beta_end            = 0.02,
        beta_schedule       = "linear",
        clip_sample         = False,
        set_alpha_to_one    = False,
    )

    for CLASS_LABEL in [0,1,2,3,4,5,6,7,8]:
        print(f"Sampling class {CLASS_LABEL}, CFG={CFG_SCALE}, {NUM_STEPS} steps...")
        latents = sample_with_cfg(model, scheduler, CLASS_LABEL, CFG_SCALE, device, SEED)

        with torch.no_grad():
            pixels = vae.decode(latents / VAE_SCALE)
        pixels = (pixels.clamp(-1, 1) + 1) / 2
        OUT = f"src/outputs/myweights_test_{CLASS_LABEL}_local.png"
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        plt.imsave(OUT, pixels[0].permute(1, 2, 0).cpu().numpy())
        print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
