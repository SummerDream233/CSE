import argparse
from utils import set_seed
from models import train_gan, train_ddpm
from fid import compute_fid_for_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train_gan", "train_ddpm", "fid_gan", "fid_ddpm"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--data_fraction", type=float, default=0.01, help="使用训练数据的比例，如 0.1 表示十分之一。")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="训练用 train，FID 真实分布用 test(val)。")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--sample_dir", type=str, default="./samples")
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--g_base", type=int, default=64)
    parser.add_argument("--d_base", type=int, default=64)
    parser.add_argument("--gan_ckpt", type=str, default="./checkpoints/gan_best.pt")
    parser.add_argument("--unet_base", type=int, default=64)
    parser.add_argument("--t_dim", type=int, default=256)
    parser.add_argument("--ddpm_T", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)
    parser.add_argument("--ddpm_ckpt", type=str, default="./checkpoints/ddpm_best.pt")
    parser.add_argument("--fid_n", type=int, default=500)
    parser.add_argument("--fid_batch", type=int, default=64)
    parser.add_argument("--fid_split", type=str, default="test", choices=["train", "test"], help="FID 真实分布所用划分，test 即 val 集。")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.mode == "train_gan":
        train_gan(args)
    elif args.mode == "train_ddpm":
        train_ddpm(args)
    elif args.mode == "fid_gan":
        fid = compute_fid_for_model(args, model_type="gan")
        print(f"FID (GAN): {fid:.4f}")
    elif args.mode == "fid_ddpm":
        fid = compute_fid_for_model(args, model_type="ddpm")
        print(f"FID (DDPM): {fid:.4f}")


if __name__ == "__main__":
    main()
