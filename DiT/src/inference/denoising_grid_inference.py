import argparse, os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from training.diffusion import Diffusion
from training.training import build_vae, VAE_SCALE
from diffusers import DDPMScheduler, DDIMScheduler


#defaults
MYDIT_CKPT   = "training-checkpoints/d8-h1-p2/dit_step0060000.pt"
OFFICIAL_CKPT = "src/pretrained-models/DiT-XL-2-256x256.pt"
VAE_CKPT     = "pretrained-models/kl-f8-model.ckpt"
CLASS_LABEL  = 7
CFG_SCALE    = 3.0
NUM_STEPS    = 100
SEED         = 42
DEVICE       = "cuda"
OUT          = "src/outputs/grid.png"
GRID_ROWS    = 10
GRID_COLS    = 10
NUM_IMAGES   = GRID_ROWS * GRID_COLS



def load_official(ckpt, device):
    from model.dit_facebook import DiT_XL_2
    model = DiT_XL_2(input_size=32, num_classes=1000).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    print(f"Loaded official DiT-XL/2  ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")
    return model


def _load_config_csv():
    import csv
    csv_path = os.path.join(_HERE, "../../training-checkpoints/model_configs.csv")
    table = {}
    with open(os.path.abspath(csv_path)) as f:
        for row in csv.DictReader(f):
            table[row["checkpoint_dir"].rstrip("/")] = row
    return table


def infer_config(sd, raw, ckpt_path):
    hidden_size = sd["time_embed.mlp.0.weight"].shape[0]
    patch_size  = sd["patch_embed.conv.weight"].shape[2]
    num_classes = sd["class_embed.weight"].shape[0] - 1
    depth       = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".adaLN_modulation.1.bias"))

    # Prefer heads saved in checkpoint, then CSV lookup, then error
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

    return dict(input_dim=(4, 32, 32), patch_size=patch_size, hidden_size=hidden_size,
                depth=depth, num_heads=num_heads, num_classes=num_classes, class_dropout_prob=0.1)


def load_mydit(ckpt, device):
    from model.model import DiT
    raw = torch.load(ckpt, map_location=device, weights_only=True)
    sd  = raw["ema"] if "ema" in raw else (raw["model"] if "model" in raw else raw)
    cfg = infer_config(sd, raw, ckpt)
    print(f"Inferred config: {cfg}")
    model = DiT(**cfg).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded  ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)"
          f"  epoch={raw.get('epoch', '?')}  step={raw.get('step', '?')}")
    return model


@torch.no_grad()
def sample_evolution_custom(model, diffusion, class_label, cfg_scale, device,
                             seed, num_steps, num_images, eta):
    torch.manual_seed(seed)
    C, H, W     = model.input_dim
    x           = torch.randn(1, C, H, W, device=device)
    labels      = torch.tensor([class_label],      dtype=torch.long, device=device)
    null_labels = torch.full_like(labels, model.num_classes)

    timesteps = diffusion._strided_timesteps(num_steps)
    pairs     = list(zip(timesteps, timesteps[1:] + [-1]))
    x0_preds  = []

    for t_curr, t_prev in pairs:
        t_t        = torch.full((1,), t_curr, device=device, dtype=torch.long)
        eps_cond   = model(x, t_t, labels)[:, :C]
        eps_uncond = model(x, t_t, null_labels)[:, :C]
        eps        = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        ah_t    = diffusion.sched.alphas_cumprod[t_curr].to(device)
        x0_pred = (x - (1 - ah_t).sqrt() * eps) / ah_t.sqrt()
        x0_preds.append(x0_pred.clone())

        x = diffusion._ddpm_step(model, x, t_curr, t_prev, labels, cfg_scale, eta=eta)

    indices = np.linspace(0, len(x0_preds) - 1, num_images).astype(int)
    return [x0_preds[i] for i in indices]


@torch.no_grad()
def sample_evolution_diffusers(model, scheduler, class_label, cfg_scale, device,
                                seed, num_steps, num_images, is_official=False):
    torch.manual_seed(seed)
    C = 4  # both official and mydit latent channels
    x = torch.randn(1, C, 32, 32, device=device)

    if is_official:
        labels = torch.tensor([class_label, 1000], dtype=torch.long, device=device)
    else:
        label_c = torch.tensor([class_label],      dtype=torch.long, device=device)
        label_u = torch.tensor([model.num_classes], dtype=torch.long, device=device)

    scheduler.set_timesteps(num_steps)
    x0_preds = []

    for t in scheduler.timesteps:
        t_in = t.reshape(1).to(device)

        if is_official:
            out          = model(torch.cat([x, x]), t_in.expand(2), labels)[:, :C]
            eps_c, eps_u = out.chunk(2)
        else:
            eps_c = model(x, t_in, label_c)[:, :C]
            eps_u = model(x, t_in, label_u)[:, :C]

        guided  = eps_u + cfg_scale * (eps_c - eps_u)
        alpha_t = scheduler.alphas_cumprod[t].to(device)
        x0_preds.append(((x - guided * (1 - alpha_t).sqrt()) / alpha_t.sqrt()).clone())
        x = scheduler.step(guided, t, x).prev_sample

    indices = np.linspace(0, len(x0_preds) - 1, num_images).astype(int)
    return [x0_preds[i] for i in indices]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       choices=["official", "mydit"], default="mydit")
    p.add_argument("--sampler",     choices=["ddim", "ddpm", "d_ddim", "d_ddpm"], default="ddim")
    p.add_argument("--ckpt",        default=None)
    p.add_argument("--class_label", type=int,   default=CLASS_LABEL)
    p.add_argument("--cfg_scale",   type=float, default=CFG_SCALE)
    p.add_argument("--num_steps",   type=int,   default=NUM_STEPS)
    p.add_argument("--seed",        type=int,   default=SEED)
    p.add_argument("--out",         default=OUT)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(DEVICE)

    if args.model == "official" and args.sampler in ("ddim", "ddpm"):
        raise ValueError("--sampler ddim/ddpm require --model mydit (custom Diffusion class)")

    ckpt = args.ckpt or (OFFICIAL_CKPT if args.model == "official" else MYDIT_CKPT)
    model = load_official(ckpt, device) if args.model == "official" else load_mydit(ckpt, device)
    vae   = build_vae(VAE_CKPT, device)

    print(f"model={args.model}  sampler={args.sampler}  class={args.class_label}"
          f"  CFG={args.cfg_scale}  steps={args.num_steps}")

    if args.sampler in ("ddim", "ddpm"):
        eta       = 0.0 if args.sampler == "ddim" else 1.0
        diffusion = Diffusion(schedule="linear", T=1000, device=str(device))
        x0_preds  = sample_evolution_custom(
            model, diffusion, args.class_label, args.cfg_scale,
            device, args.seed, args.num_steps, NUM_IMAGES, eta,
        )
    else:
        sched_cls = DDIMScheduler if args.sampler == "d_ddim" else DDPMScheduler
        scheduler = sched_cls(
            num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
            beta_schedule="linear", clip_sample=False,
            **({} if args.sampler == "d_ddpm" else {"set_alpha_to_one": False}),
        )
        x0_preds = sample_evolution_diffusers(
            model, scheduler, args.class_label, args.cfg_scale,
            device, args.seed, args.num_steps, NUM_IMAGES,
            is_official=(args.model == "official"),
        )

    with torch.no_grad():
        decoded = [(vae.decode(x0 / VAE_SCALE).clamp(-1, 1) + 1) / 2
                   for x0 in x0_preds]
    decoded = [d[0].permute(1, 2, 0).cpu().numpy() for d in decoded]

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(GRID_COLS * 2, GRID_ROWS * 2))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    for ax, img in zip(axes.flat, decoded):
        ax.imshow(img)
        ax.axis("off")

    out = args.out.replace(".png", f"_{args.model}_{args.sampler}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
