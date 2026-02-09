import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from utils import get_dataloader, ensure_dir, setup_train_logging, get_train_logger
from fid import compute_fid_in_memory


class DCGANGenerator(nn.Module):
    def __init__(self, z_dim=128, base=64, img_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, base * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(True),
            nn.ConvTranspose2d(base, img_ch, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    def __init__(self, base=64, img_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_ch, base, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def gan_weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_gan(args):
    log_dir = getattr(args, "log_dir", "./logs")
    setup_train_logging(log_dir, "train_gan")
    log = get_train_logger()
    log.info("=== GAN 训练开始 ===")
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

    G = DCGANGenerator(z_dim=args.z_dim, base=args.g_base).to(device)
    D = DCGANDiscriminator(base=args.d_base).to(device)
    G.apply(gan_weights_init)
    D.apply(gan_weights_init)

    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    ensure_dir(args.ckpt_dir)
    ensure_dir(args.sample_dir)
    fixed_z = torch.randn(64, args.z_dim, 1, 1, device=device)
    best_fid = float("inf")
    fid_n_eval = getattr(args, "fid_n_best", 1000)

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        pbar = tqdm(dl, desc=f"[GAN] Epoch {epoch}/{args.epochs}", leave=False)
        sum_lossD = 0.0
        sum_lossG = 0.0
        n_batches = 0

        for real, _ in pbar:
            real = real.to(device, non_blocking=True)
            bs = real.size(0)

            z = torch.randn(bs, args.z_dim, 1, 1, device=device)
            fake = G(z).detach()
            logits_real = D(real)
            logits_fake = D(fake)
            y_real = torch.ones(bs, device=device)
            y_fake = torch.zeros(bs, device=device)
            lossD = bce(logits_real, y_real) + bce(logits_fake, y_fake)
            optD.zero_grad(set_to_none=True)
            lossD.backward()
            optD.step()

            z = torch.randn(bs, args.z_dim, 1, 1, device=device)
            fake = G(z)
            logits_fake = D(fake)
            lossG = bce(logits_fake, y_real)
            optG.zero_grad(set_to_none=True)
            lossG.backward()
            optG.step()

            sum_lossD += lossD.item()
            sum_lossG += lossG.item()
            n_batches += 1
            pbar.set_postfix(lossD=float(lossD.item()), lossG=float(lossG.item()))

        avg_lossD = sum_lossD / n_batches
        avg_lossG = sum_lossG / n_batches
        log.info("Epoch %d/%d  lossD=%.4f  lossG=%.4f", epoch, args.epochs, avg_lossD, avg_lossG)

        if epoch % args.sample_every == 0 or epoch == 1:
            with torch.no_grad():
                samp = G(fixed_z)
                grid = make_grid((samp + 1) / 2, nrow=8)
                save_image(grid, os.path.join(args.sample_dir, f"gan_epoch_{epoch:04d}.png"))

        if epoch % 5 == 0 or epoch == 1:
            fid_val = compute_fid_in_memory(G, "gan", args, device, fid_n=fid_n_eval)
            log.info("Epoch %d  FID(%d)=%.4f", epoch, fid_n_eval, fid_val)
            if fid_val < best_fid:
                best_fid = fid_val
                path_best = os.path.join(args.ckpt_dir, "gan_best.pt")
                torch.save(
                    {"G": G.state_dict(), "D": D.state_dict(), "epoch": epoch, "args": vars(args), "fid": fid_val},
                    path_best,
                )
                log.info("已更新最佳 checkpoint (FID=%.4f): %s", fid_val, path_best)

    log.info("=== GAN 训练结束 ===")


@torch.no_grad()
def sample_gan(G: nn.Module, n: int, z_dim: int, device: torch.device, batch: int = 64) -> torch.Tensor:
    G.eval()
    outs = []
    for i in range(0, n, batch):
        bs = min(batch, n - i)
        z = torch.randn(bs, z_dim, 1, 1, device=device)
        x = G(z)
        outs.append(x.cpu())
    return torch.cat(outs, dim=0)
