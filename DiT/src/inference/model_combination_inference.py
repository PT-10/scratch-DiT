import os, sys, argparse
import torch
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                      # utils/dit_facebook
sys.path.insert(0, os.path.join(_HERE, ".."))  # model, training

CKPT     = "src/pretrained-models/DiT-XL-2-256x256.pt"
VAE_CKPT = "src/pretrained-models/kl-f8-model.ckpt"


def remap_state_dict(sd):
    out = {}
    for k, v in sd.items():
        if   k == "pos_embed":                              out["patch_embed.pos_embed"] = v
        elif k.startswith("x_embedder.proj."):              out[k.replace("x_embedder.proj.", "patch_embed.conv.")] = v
        elif k.startswith("t_embedder.mlp."):               out[k.replace("t_embedder.mlp.", "time_embed.mlp.")] = v
        elif k == "y_embedder.embedding_table.weight":      out["class_embed.weight"] = v
        elif ".attn.qkv.weight" in k:                      out[k.replace(".attn.qkv.weight", ".attn.in_proj_weight")] = v
        elif ".attn.qkv.bias"   in k:                      out[k.replace(".attn.qkv.bias",   ".attn.in_proj_bias")]   = v
        elif ".attn.proj."      in k:                      out[k.replace(".attn.proj.",       ".attn.out_proj.")]       = v
        elif ".mlp.fc1."        in k:                      out[k.replace(".mlp.fc1.", ".mlp.0.")] = v
        elif ".mlp.fc2."        in k:                      out[k.replace(".mlp.fc2.", ".mlp.2.")] = v
        elif k.startswith("final_layer.adaLN_modulation."): out[k.replace("final_layer.adaLN_modulation.", "final_adaLN_modulation.")] = v
        elif k.startswith("final_layer.linear."):           out[k.replace("final_layer.linear.", "final_proj.")] = v
        elif ".adaLN_modulation." in k:                    out[k] = v
        # norm layers have no params (elementwise_affine=False) → skip
    return out



def load_official(ckpt, device):
    from model.dit_facebook import DiT_XL_2
    model = DiT_XL_2(input_size=32, num_classes=1000).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    print(f"Loaded official DiT-XL/2  ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")
    return model


def load_mydit(ckpt, device):
    from model.model import DiT
    model = DiT(
        input_dim=(4, 32, 32), patch_size=2, hidden_size=1152,
        depth=28, num_heads=16, num_classes=1000, class_dropout_prob=0.1,
    ).to(device)
    raw = torch.load(ckpt, map_location=device, weights_only=True)
    remapped = remap_state_dict(raw["model"] if "model" in raw else raw)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:    print(f"  Missing keys    ({len(missing)}): {missing[:5]}")
    if unexpected: print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    model.eval()
    print(f"Loaded my DiT-XL/2  ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")
    return model


def load_vae(args, device):
    if args.vae == "diffusers":
        from diffusers import AutoencoderKL
        print("Loading VAE from stabilityai/sd-vae-ft-ema ...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
        vae.eval()
        return vae, 0.18215, True
    else:
        from training.training import VAE_SCALE, build_vae
        return build_vae(args.vae_ckpt, device), VAE_SCALE, False


def decode(vae, latents, scale, is_diffusers):
    with torch.no_grad():
        out = vae.decode(latents / scale)
        pixels = out.sample if is_diffusers else out
    return (pixels.clamp(-1, 1) + 1) / 2


def make_alphas_cumprod(T=1000, beta_start=1e-4, beta_end=0.02):
    return torch.cumprod(1.0 - torch.linspace(beta_start, beta_end, T), dim=0)


def ddim_timesteps(num_steps, T=1000):
    step = T // num_steps
    return list(range(T - 1, -1, -step))[:num_steps]


@torch.no_grad()
def sample_diffusers(model, args, device):
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="linear", clip_sample=False, set_alpha_to_one=False,
    )
    torch.manual_seed(args.seed)
    latents = torch.randn(1, 4, 32, 32, device=device)
    labels  = torch.tensor([args.class_label, 1000], dtype=torch.long, device=device)
    scheduler.set_timesteps(args.steps)

    for t in scheduler.timesteps:
        x_in = torch.cat([latents, latents])
        noise_pred = model(x_in, t.expand(2).to(device), labels)[:, :4]
        eps_c, eps_u = noise_pred.chunk(2)
        latents = scheduler.step(eps_u + args.cfg * (eps_c - eps_u), t, latents).prev_sample

    return latents


@torch.no_grad()
def sample_ddim(model, args, device):
    alphas = make_alphas_cumprod().to(device)
    torch.manual_seed(args.seed)
    timesteps = ddim_timesteps(args.steps)

    is_mydit = hasattr(model, "input_dim")
    C = model.input_dim[0] if is_mydit else 4
    x = torch.randn(1, C, 32, 32, device=device)

    for i, t in enumerate(timesteps):
        a_t    = alphas[t]
        a_prev = alphas[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)

        if is_mydit:
            lc = torch.tensor([args.class_label],   dtype=torch.long, device=device)
            lu = torch.tensor([model.num_classes],   dtype=torch.long, device=device)
            tt = torch.tensor([t],                   dtype=torch.long, device=device)
            eps = model(x, tt, lu)[:, :C] + args.cfg * (model(x, tt, lc)[:, :C] - model(x, tt, lu)[:, :C])
        else:
            labels = torch.tensor([args.class_label, 1000], dtype=torch.long, device=device)
            tt     = torch.tensor([t, t],                   dtype=torch.long, device=device)
            out    = model(torch.cat([x, x]), tt, labels)[:, :4]
            eps_c, eps_u = out.chunk(2)
            eps = eps_u + args.cfg * (eps_c - eps_u)

        x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        x  = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps

    return x


def sample_ddpm(model, args, device):
    from training.diffusion import Diffusion
    diffusion = Diffusion(schedule="linear", device=str(device))
    torch.manual_seed(args.seed)
    labels = torch.tensor([args.class_label], dtype=torch.long, device=device)
    return diffusion.sample(model, n=1, labels=labels,
                            cfg_scale=args.cfg, inference_steps=args.steps, eta=1.0)

def parse_args():
    p = argparse.ArgumentParser(description="DiT inference")
    p.add_argument("--model",       choices=["official", "mydit"], default="mydit")
    p.add_argument("--vae",         choices=["diffusers", "local"], default="local")
    p.add_argument("--sampler",     choices=["diffusers", "ddim", "ddpm"], default="ddim")
    p.add_argument("--ckpt",        default=CKPT)
    p.add_argument("--vae-ckpt",    default=VAE_CKPT, dest="vae_ckpt")
    p.add_argument("--class-label", type=int,   default=207,  dest="class_label")
    p.add_argument("--cfg",         type=float, default=4.0)
    p.add_argument("--steps",       type=int,   default=100)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--out",         default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    if args.sampler == "ddpm" and args.model == "official":
        raise ValueError("--sampler ddpm requires --model mydit (Diffusion.sample uses mydit interface)")

    if args.out is None:
        args.out = f"src/outputs/{args.model}_{args.vae}vae_{args.sampler}.png"

    model = load_official(args.ckpt, device) if args.model == "official" else load_mydit(args.ckpt, device)
    vae, scale, is_diffusers_vae = load_vae(args, device)

    print(f"Sampling class {args.class_label}, CFG={args.cfg}, {args.steps} steps ({args.sampler})...")
    if   args.sampler == "diffusers": latents = sample_diffusers(model, args, device)
    elif args.sampler == "ddim":      latents = sample_ddim(model, args, device)
    else:                             latents = sample_ddpm(model, args, device)

    pixels = decode(vae, latents, scale, is_diffusers_vae)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.imsave(args.out, pixels[0].permute(1, 2, 0).cpu().numpy())
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
