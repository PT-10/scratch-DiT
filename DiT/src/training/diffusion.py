import tqdm
import math
import torch

def q_sample(x0, t, schedule):
    device = x0.device
    sqrt_alpha = schedule.sqrt_alphas_cumprod.to(device)[t]
    sqrt_one_minus = schedule.sqrt_one_minus_alphas_cumprod.to(device)[t]

    sqrt_alpha = sqrt_alpha[:, None, None, None]
    sqrt_one_minus = sqrt_one_minus[:, None, None, None]

    noise = torch.randn_like(x0)
    x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
    return x_t, noise

class BetaSchedule:
    def __init__(self, schedule='linear', T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, T)
        elif schedule == 'cosine':
            # Cosine schedule (Nichol & Dhariwal 2021)
            s = 0.008
            t = torch.linspace(0, T, T + 1) / T
            alphas_cumprod = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = betas.clamp(max=0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)


class Diffusion:
    def __init__(self, schedule='linear', T=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.T = T
        self.device = device
        self.sched = BetaSchedule(schedule, T, beta_start, beta_end)

    def noise_images(self, x0, t):
        return q_sample(x0, t, self.sched)

    def sample_timesteps(self, n):
        return torch.randint(0, self.T, (n,), device=self.device)

    def _strided_timesteps(self, inference_steps):
        step = max(self.T // inference_steps, 1)
        return list(range(self.T - 1, -1, -step))[:inference_steps]

    def _ddpm_step(self, model, x, t_curr, t_prev, labels, cfg_scale, eta=1.0):
        C = x.shape[1]
        n = x.shape[0]
        null_labels = torch.full_like(labels, model.num_classes)
        t_tensor = torch.full((n,), t_curr, device=self.device, dtype=torch.long)

        out_cond   = model(x, t_tensor, labels)
        out_uncond = model(x, t_tensor, null_labels)
        eps = out_uncond[:, :C] + cfg_scale * (out_cond[:, :C] - out_uncond[:, :C])
    
        ah_t    = self.sched.alphas_cumprod[t_curr].to(self.device)
        ah_prev = self.sched.alphas_cumprod[t_prev].to(self.device) if t_prev >= 0 else torch.tensor(1.0)

        # predict x0
        x0_pred = (x - (1 - ah_t).sqrt() * eps) / ah_t.sqrt()

        if t_prev < 0:
            return x0_pred

        # sigma splits sqrt(1-a_prev) into stochastic + deterministic parts
        sigma     = eta * ((1 - ah_prev) / (1 - ah_t) * (1 - ah_t / ah_prev)).sqrt()
        dir_coeff = (1 - ah_prev - sigma ** 2).clamp(min=0).sqrt()

        x = ah_prev.sqrt() * x0_pred + dir_coeff * eps
        if eta > 0:
            x = x + sigma * torch.randn_like(x)
        return x

    @torch.no_grad()
    def sample(self, model, n, labels, cfg_scale=3.0, inference_steps=None, eta=1.0):
        model.eval()
        C, H, W = model.input_dim
        x = torch.randn(n, C, H, W, device=self.device)

        timesteps = self._strided_timesteps(inference_steps or self.T)
        pairs = list(zip(timesteps, timesteps[1:] + [-1]))  # (t_curr, t_prev)

        for t_curr, t_prev in tqdm(pairs, position=0, desc="sampling"):
            x = self._ddpm_step(model, x, t_curr, t_prev, labels, cfg_scale, eta=eta)

        model.train()
        return x

    @torch.no_grad()
    def sample_progressive(self, model, n, labels, cfg_scale=3.0, inference_steps=None, eta=1.0):
        model.eval()
        C, H, W = model.input_dim
        x = torch.randn(n, C, H, W, device=self.device)

        timesteps = self._strided_timesteps(inference_steps or self.T)
        pairs = list(zip(timesteps, timesteps[1:] + [-1]))

        frames = [x.clone().cpu()]   # first frame: pure noise

        for t_curr, t_prev in tqdm(pairs, position=0, desc="sampling"):
            x = self._ddpm_step(model, x, t_curr, t_prev, labels, cfg_scale, eta=eta)
            frames.append(x.clone().cpu())

        model.train()
        return x, frames
