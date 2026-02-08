"""
DDPM：时间嵌入、ResBlock、UNet、扩散过程与训练。
"""

import math
import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from utils import get_dataloader, ensure_dir, setup_train_logging, get_train_logger


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device).float() / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch))
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """64x64 紧凑 UNet，预测噪声 epsilon。"""

    def __init__(self, in_ch=3, base=64, t_dim=256):
        super().__init__()
        self.t_emb = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.c0 = nn.Conv2d(in_ch, base, 3, padding=1)
        self.d1 = ResBlock(base, base, t_dim)
        self.p1 = nn.Conv2d(base, base, 4, 2, 1)
        self.d2 = ResBlock(base, base * 2, t_dim)
        self.p2 = nn.Conv2d(base * 2, base * 2, 4, 2, 1)
        self.d3 = ResBlock(base * 2, base * 4, t_dim)
        self.p3 = nn.Conv2d(base * 4, base * 4, 4, 2, 1)
        self.d4 = ResBlock(base * 4, base * 8, t_dim)
        self.p4 = nn.Conv2d(base * 8, base * 8, 4, 2, 1)
        self.m1 = ResBlock(base * 8, base * 8, t_dim)
        self.m2 = ResBlock(base * 8, base * 8, t_dim)
        self.u4 = nn.ConvTranspose2d(base * 8, base * 8, 4, 2, 1)
        self.up4 = ResBlock(base * 8 + base * 8, base * 4, t_dim)
        self.u3 = nn.ConvTranspose2d(base * 4, base * 4, 4, 2, 1)
        self.up3 = ResBlock(base * 4 + base * 4, base * 2, t_dim)
        self.u2 = nn.ConvTranspose2d(base * 2, base * 2, 4, 2, 1)
        self.up2 = ResBlock(base * 2 + base * 2, base, t_dim)
        self.u1 = nn.ConvTranspose2d(base, base, 4, 2, 1)
        self.up1 = ResBlock(base + base, base, t_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.t_emb(t)
        x0 = self.c0(x)
        x1 = self.d1(x0, te)
        x1p = self.p1(x1)
        x2 = self.d2(x1p, te)
        x2p = self.p2(x2)
        x3 = self.d3(x2p, te)
        x3p = self.p3(x3)
        x4 = self.d4(x3p, te)
        x4p = self.p4(x4)
        m = self.m2(self.m1(x4p, te), te)
        u4 = self.u4(m)
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.up4(u4, te)
        u3 = self.u3(u4)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.up3(u3, te)
        u2 = self.u2(u3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up2(u2, te)
        u1 = self.u1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.up1(u1, te)
        return self.out(u1)


@dataclass
class DDPMConfig:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class DDPM(nn.Module):
    def __init__(self, eps_model: nn.Module, cfg: DDPMConfig):
        super().__init__()
        self.eps_model = eps_model
        self.cfg = cfg
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T)
        alphas = 1.0 - betas
        a_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("a_bar", a_bar)
        self.register_buffer("sqrt_a_bar", torch.sqrt(a_bar))
        self.register_buffer("sqrt_one_minus_a_bar", torch.sqrt(1.0 - a_bar))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        s1 = self.sqrt_a_bar[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_one_minus_a_bar[t].view(-1, 1, 1, 1)
        return s1 * x0 + s2 * noise

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        device = x0.device
        bs = x0.size(0)
        t = torch.randint(0, self.cfg.T, (bs,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps_pred = self.eps_model(xt, t)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def p_sample(self, xt: torch.Tensor, t: int) -> torch.Tensor:
        device = xt.device
        t_batch = torch.full((xt.size(0),), t, device=device, dtype=torch.long)
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        a_bar_t = self.a_bar[t]
        eps = self.eps_model(xt, t_batch)
        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1 - a_bar_t)
        mean = coef1 * (xt - coef2 * eps)
        if t == 0:
            return mean
        return mean + torch.sqrt(beta_t) * torch.randn_like(xt)

    @torch.no_grad()
    def sample(
        self, n: int, shape: Tuple[int, int, int], device: torch.device, batch: int = 32
    ) -> torch.Tensor:
        C, H, W = shape
        outs = []
        for i in range(0, n, batch):
            bs = min(batch, n - i)
            x = torch.randn(bs, C, H, W, device=device)
            for t in range(self.cfg.T - 1, -1, -1):
                x = self.p_sample(x, t)
            outs.append(x.cpu())
        return torch.cat(outs, dim=0)


def train_ddpm(args):
    log_dir = getattr(args, "log_dir", "./logs")
    setup_train_logging(log_dir, "train_ddpm")
    log = get_train_logger()
    log.info("=== DDPM 训练开始 ===")
    log.info("split=%s data_fraction=%s epochs=%d batch_size=%d lr=%s",
             args.split, args.data_fraction, args.epochs, args.batch_size, args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    _, dl = get_dataloader(
        args.data_root,
        args.batch_size,
        args.num_workers,
        split=args.split,
        data_fraction=args.data_fraction,
        seed=args.seed,
    )

    unet = UNet(in_ch=3, base=args.unet_base, t_dim=args.t_dim).to(device)
    ddpm = DDPM(
        unet,
        DDPMConfig(T=args.ddpm_T, beta_start=args.beta_start, beta_end=args.beta_end),
    ).to(device)
    opt = torch.optim.AdamW(ddpm.parameters(), lr=args.lr, weight_decay=1e-4)

    ensure_dir(args.ckpt_dir)
    ensure_dir(args.sample_dir)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        ddpm.train()
        pbar = tqdm(dl, desc=f"[DDPM] Epoch {epoch}/{args.epochs}", leave=False)
        sum_loss = 0.0
        n_batches = 0
        for x0, _ in pbar:
            x0 = x0.to(device, non_blocking=True)
            loss = ddpm.loss(x0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            opt.step()
            sum_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=float(loss.item()))

        avg_loss = sum_loss / n_batches
        log.info("Epoch %d/%d  loss=%.4f", epoch, args.epochs, avg_loss)

        if epoch % args.sample_every == 0 or epoch == 1:
            ddpm.eval()
            with torch.no_grad():
                samp = ddpm.sample(n=64, shape=(3, 64, 64), device=device, batch=16)
                grid = make_grid((samp + 1) / 2, nrow=8)
                save_image(grid, os.path.join(args.sample_dir, f"ddpm_epoch_{epoch:04d}.png"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            path_best = os.path.join(args.ckpt_dir, "ddpm_best.pt")
            torch.save(
                {"ddpm": ddpm.state_dict(), "epoch": epoch, "args": vars(args), "loss": avg_loss},
                path_best,
            )
            log.info("已更新最佳 checkpoint (loss=%.4f): %s", avg_loss, path_best)

    log.info("=== DDPM 训练结束 ===")