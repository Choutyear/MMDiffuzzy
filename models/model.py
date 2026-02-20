import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionScheduler(nn.Module):
    def __init__(self, T: int, beta_start: float, beta_end: float, device: str):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.T = T

    def sample_t(self, B: int, device: str):
        return torch.randint(low=0, high=self.T, size=(B,), device=device)

    def q_sample(self, x0, t, noise):
        a_bar = self.alpha_bar[t].view(-1, 1, 1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

    def x0_from_eps(self, xt, t, eps):
        a_bar = self.alpha_bar[t].view(-1, 1, 1, 1)
        return (xt - torch.sqrt(1.0 - a_bar) * eps) / (torch.sqrt(a_bar) + 1e-12)

class LatentDiffusion(nn.Module):
    def __init__(self, unet, scheduler: DiffusionScheduler):
        super().__init__()
        self.unet = unet
        self.sched = scheduler

    def forward(self, x0, t, Ft):
        noise = torch.randn_like(x0)
        xt = self.sched.q_sample(x0, t, noise)
        pred = self.unet(xt, t, Ft)
        loss = F.mse_loss(pred, noise)
        x0_hat = self.sched.x0_from_eps(xt, t, pred)
        return loss, xt, noise, pred, x0_hat