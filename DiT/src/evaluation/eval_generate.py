import argparse
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.model import DiT
from training.diffusion import Diffusion
from training.training import VAE_SCALE, build_vae


# ---------------------------------------------------------------------------
# Config inference (matches test_myweights_grid.py)
# ---------------------------------------------------------------------------

def _load_config_csv():
    import csv
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training-checkpoints/model_configs.csv"))
    table = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            table[row["checkpoint_dir"].rstrip("/")] = row
    return table


def infer_config(sd, raw, ckpt_path):
    hidden_size = sd["time_embed.mlp.0.weight"].shape[0]
    patch_size  = sd["patch_embed.conv.weight"].shape[2]
    num_classes = sd["class_embed.weight"].shape[0] - 1
    depth       = sum(
        1 for k in sd
        if k.startswith("blocks.") and k.endswith(".adaLN_modulation.1.bias")
    )
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
        print(f"  num_heads={num_heads} (from model_configs.csv — '{match['checkpoint_dir']}')")
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
    print(f"  config: {cfg}")
    model = DiT(**cfg).to(device)
    model.load_state_dict(sd)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  loaded {params:.1f}M params  step={raw.get('step', '?')}")
    return model, cfg


# ---------------------------------------------------------------------------
# Decode a batch of latents → uint8 PIL images
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_latents(vae, latents):
    """latents: (B, 4, 32, 32) scaled — returns list of PIL Images."""
    imgs = vae.decode(latents / VAE_SCALE)          # (B, 3, H, W) in [-1, 1]
    imgs = (imgs.clamp(-1, 1) + 1) / 2             # → [0, 1]
    imgs = (imgs * 255).byte().permute(0, 2, 3, 1) # → (B, H, W, 3) uint8
    return [Image.fromarray(img.cpu().numpy()) for img in imgs]


# ---------------------------------------------------------------------------
# Generate images for one checkpoint
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_for_checkpoint(
    model, diffusion, vae, device,
    num_images, num_classes, cfg_scale, num_steps, batch_size,
    out_dir,
):
    os.makedirs(out_dir, exist_ok=True)

    # Skip if already fully generated
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    if existing >= num_images:
        print(f"  already have {existing} images in {out_dir}, skipping")
        return

    C, H, W = model.input_dim

    generated = 0
    pbar = tqdm(total=num_images, desc="  generating", unit="img", leave=False)

    while generated < num_images:
        this_batch = min(batch_size, num_images - generated)

        # Fixed seeds per image for reproducibility
        seeds   = list(range(generated, generated + this_batch))
        labels  = torch.tensor(
            [i % num_classes for i in seeds],
            dtype=torch.long, device=device,
        )

        # Sample with a single seed covering the batch start
        torch.manual_seed(seeds[0])
        x = torch.randn(this_batch, C, H, W, device=device)

        # DDIM (eta=0) denoising
        timesteps = diffusion._strided_timesteps(num_steps)
        pairs = list(zip(timesteps, timesteps[1:] + [-1]))

        for t_curr, t_prev in pairs:
            x = diffusion._ddpm_step(
                model, x, t_curr, t_prev, labels, cfg_scale, eta=0.0
            )

        # Decode and save
        pil_imgs = decode_latents(vae, x)
        for j, img in enumerate(pil_imgs):
            idx = generated + j
            save_path = os.path.join(out_dir, f"img_{idx:05d}.jpg")
            img.save(save_path, quality=95)

        generated += this_batch
        pbar.update(this_batch)

    pbar.close()
    print(f"  saved {generated} images to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate FID evaluation samples")
    p.add_argument("--checkpoints_dir", default="training-checkpoints/checkpoints_sweep",
                   help="root dir containing per-config checkpoint dirs (e.g. d6-h6-p2/)")
    p.add_argument("--out_dir",     default="samples/",
                   help="output root dir; images saved to out_dir/{config}/{step}/")
    p.add_argument("--num_images",  type=int,   default=5000)
    p.add_argument("--cfg_scale",   type=float, default=3.0)
    p.add_argument("--num_steps",   type=int,   default=250,
                   help="number of DDIM denoising steps")
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--steps",        default=None,
                   help="comma-separated training step checkpoints to eval, e.g. '10000,30000'")
    p.add_argument("--step_divisor", type=int, default=1,
                   help="divide checkpoint step by this when naming output dirs (e.g. 2 if local batch was half of remote)")
    p.add_argument("--vae_ckpt",    required=True,
                   help="path to KL-f8 VAE checkpoint (.ckpt)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Parse requested steps (None means all found)
    wanted_steps = None
    if args.steps:
        wanted_steps = {int(s.strip()) for s in args.steps.split(",")}

    # Build VAE once
    print("Loading VAE …")
    vae = build_vae(args.vae_ckpt, device)

    # Build diffusion helper - must match training schedule (linear)
    diffusion = Diffusion(schedule="linear", T=1000, device=str(device))

    checkpoints_dir = args.checkpoints_dir

    # Find all config sub-dirs
    config_dirs = sorted([
        d for d in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, d))
    ])
    if not config_dirs:
        print(f"No config dirs found under {checkpoints_dir}")
        return

    print(f"Found configs: {config_dirs}")

    for config_name in config_dirs:
        config_path = os.path.join(checkpoints_dir, config_name)

        # Find checkpoint files for requested steps
        ckpt_files = sorted([
            f for f in os.listdir(config_path)
            if f.startswith("dit_step") and f.endswith(".pt")
        ])

        if not ckpt_files:
            print(f"[{config_name}] no step checkpoints found, skipping")
            continue

        for ckpt_fname in ckpt_files:
            # Parse step number from filename: dit_step0029000.pt
            try:
                step = int(ckpt_fname.replace("dit_step", "").replace(".pt", ""))
            except ValueError:
                continue

            if wanted_steps is not None and step not in wanted_steps:
                continue

            ckpt_path = os.path.join(config_path, ckpt_fname)
            out_dir   = os.path.join(args.out_dir, config_name, str(step // args.step_divisor))

            print(f"\n[{config_name}] step {step:,}")

            model, cfg = load_mydit(ckpt_path, device)
            num_classes = cfg["num_classes"]

            generate_for_checkpoint(
                model=model,
                diffusion=diffusion,
                vae=vae,
                device=device,
                num_images=args.num_images,
                num_classes=num_classes,
                cfg_scale=args.cfg_scale,
                num_steps=args.num_steps,
                batch_size=args.batch_size,
                out_dir=out_dir,
            )

            # Free GPU memory before loading next checkpoint
            del model
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
