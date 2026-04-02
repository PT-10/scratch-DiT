import os
import copy
import wandb
from dotenv import load_dotenv
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model.model import DiT_d8_h1_p2
from model.autoencoder import AutoencoderKL
from training.diffusion import Diffusion
from vae_utils.encode_latent import load_latent_cache


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAE_SCALE   = 0.18215
_VAE_CONFIG = dict(
    ch=128, ch_mult=[1, 2, 4, 4], num_res_blocks=2,
    in_channels=3, out_ch=3, z_channels=4, double_z=True, resolution=256,
)


def build_vae(ckpt_path, device=device):
    vae = AutoencoderKL(_VAE_CONFIG, embed_dim=4, ckpt_path=ckpt_path).to(device)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    return vae


class Config:
    data_root       = "data/data_labeled_v2"
    data_source     = "clip"
    unlabeled_roots = []
    batch_size      = 128
    num_workers     = 4
    augment         = True
    latents_cache   = "training-checkpoints/latents_dit_s2.pt"
    vae_ckpt        = "pretrained-models/kl-f8-model.ckpt"
    schedule        = "linear"
    timesteps       = 1000
    max_steps       = 100_000
    lr              = 1e-4
    ckpt_dir        = "training-checkpoints"
    save_every      = 5_000
    log_every       = 100


def _save_loss_plot(step_losses, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(step_losses); ax.set_title("Loss/step"); ax.set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"))
    plt.close(fig)


def _train_step(model, ema, optimizer, diffusion, vae, images, labels):
    images, labels = images.to(device), labels.to(device)
    z = vae.encode(images).sample() * VAE_SCALE if vae is not None else images
    t = diffusion.sample_timesteps(z.shape[0])
    z_noisy, noise = diffusion.noise_images(z, t)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = F.mse_loss(model(z_noisy, t, labels)[:, :4], noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        for ema_p, p in zip(ema.parameters(), model.parameters()):
            ema_p.mul_(ema.decay).add_(p.data, alpha=1 - ema.decay)
    return loss.item()


def train(model, ema, optimizer, diffusion, loader, vae, cfg, start_step=0, on_save=None):
    load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    wandb.init(project="DiT", config=cfg.__dict__)
    step_losses, global_step = [], start_step

    pbar = tqdm(total=cfg.max_steps, initial=start_step, desc="training")
    while global_step < cfg.max_steps:
        model.train()
        for images, labels in loader:
            if global_step >= cfg.max_steps:
                break
            loss_val = _train_step(model, ema, optimizer, diffusion, vae, images, labels)
            step_losses.append(loss_val)
            wandb.log({"loss/step": loss_val}, step=global_step)
            global_step += 1; pbar.update(1)
            if global_step % cfg.log_every == 0:
                print(f"step {global_step}/{cfg.max_steps}  loss={loss_val:.4f}")
            if global_step % cfg.save_every == 0:
                path = os.path.join(cfg.ckpt_dir, f"dit_step{global_step:07d}.pt")
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                            "step": global_step}, path)
                print(f"saved {path}")
                _save_loss_plot(step_losses, cfg.ckpt_dir)
                if on_save: on_save()
    pbar.close()
    path = os.path.join(cfg.ckpt_dir, "dit_final.pt")
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "step": global_step, "opt": optimizer.state_dict()}, path)
    print(f"saved {path}")
    _save_loss_plot(step_losses, cfg.ckpt_dir)
    if on_save: on_save()


if __name__ == "__main__":
    RESUME = "training-checkpoints/d8-h1-p2/dit_step0055000.pt"

    cfg = Config()
    cfg.max_steps = 60_000
    cfg.ckpt_dir  = "training-checkpoints/d8-h1-p2"

    all_z, all_labels, num_classes = load_latent_cache(cfg.latents_cache)
    loader = DataLoader(TensorDataset(all_z, all_labels),
                        batch_size=cfg.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    model = DiT_d8_h1_p2(num_classes=num_classes, class_dropout_prob=0.15).to(device)

    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    ema.decay = 0.9999

    start_step = 0
    if RESUME and os.path.exists(RESUME):
        ckpt = torch.load(RESUME, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        start_step = ckpt.get("step", 0)
        print(f"resumed from {RESUME} at step {start_step}")

    print(f"DiT {sum(p.numel() for p in model.parameters())/1e6:.1f}M params | "
          f"{len(loader.dataset)} samples | {num_classes} classes")

    diffusion = Diffusion(schedule=cfg.schedule, T=cfg.timesteps, device=str(device))
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)

    train(model, ema, optimizer, diffusion, loader, None, cfg, start_step=start_step)
