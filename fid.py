"""
FID（Fréchet Inception Distance）计算：真实图与生成图分布比较。
"""

import torch
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

from utils import get_dataloader, images_to_uint8_0_255
from models.gan import DCGANGenerator, sample_gan
from models.ddpm import UNet, DDPM, DDPMConfig


@torch.no_grad()
def compute_fid_for_model(args, model_type: str) -> float:
    """
    model_type: "gan" 或 "ddpm"
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    _, dl = get_dataloader(
        args.data_root,
        args.batch_size,
        args.num_workers,
        split=args.fid_split,
        data_fraction=args.data_fraction,
        seed=args.seed,
    )

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    real_count = 0
    for x, _ in tqdm(dl, desc="[FID] Real", leave=False):
        x = x.to(device, non_blocking=True)
        u8 = images_to_uint8_0_255(x)
        fid.update(u8, real=True)
        real_count += u8.size(0)
        if real_count >= args.fid_n:
            break

    if model_type == "gan":
        ckpt = torch.load(args.gan_ckpt, map_location="cpu")
        G = DCGANGenerator(z_dim=args.z_dim, base=args.g_base).to(device)
        G.load_state_dict(ckpt["G"])
        G.eval()
        fake = sample_gan(G, n=args.fid_n, z_dim=args.z_dim, device=device, batch=args.fid_batch)
        for i in tqdm(range(0, fake.size(0), args.fid_batch), desc="[FID] Fake(GAN)", leave=False):
            chunk = fake[i : i + args.fid_batch].to(device)
            fid.update(images_to_uint8_0_255(chunk), real=False)

    elif model_type == "ddpm":
        ckpt = torch.load(args.ddpm_ckpt, map_location="cpu")
        unet = UNet(in_ch=3, base=args.unet_base, t_dim=args.t_dim).to(device)
        ddpm = DDPM(
            unet,
            DDPMConfig(T=args.ddpm_T, beta_start=args.beta_start, beta_end=args.beta_end),
        ).to(device)
        ddpm.load_state_dict(ckpt["ddpm"])
        ddpm.eval()
        fake = ddpm.sample(n=args.fid_n, shape=(3, 64, 64), device=device, batch=args.fid_batch)
        for i in tqdm(range(0, fake.size(0), args.fid_batch), desc="[FID] Fake(DDPM)", leave=False):
            chunk = fake[i : i + args.fid_batch].to(device)
            fid.update(images_to_uint8_0_255(chunk), real=False)
    else:
        raise ValueError("model_type 必须是 'gan' 或 'ddpm'")

    return float(fid.compute().item())
