"""
从 train_gan.log 解析 lossD / lossG 并画在一张图上。
用法: python plot_gan_loss.py [--log logs/train_gan.log] [--out gan_loss.png]
"""

import argparse
import re
import matplotlib.pyplot as plt

# 指定为新罗马字体（Times New Roman）
plt.rcParams["font.family"] = "Times New Roman"


def parse_gan_log(log_path: str):
    """解析 GAN 训练 log，返回 (epochs, lossD_list, lossG_list)。"""
    pattern = re.compile(r"Epoch (\d+)/\d+\s+lossD=([\d.]+)\s+lossG=([\d.]+)")
    epochs, lossD_list, lossG_list = [], [], []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                lossD_list.append(float(m.group(2)))
                lossG_list.append(float(m.group(3)))
    return epochs, lossD_list, lossG_list


def main():
    parser = argparse.ArgumentParser(description="绘制 GAN 训练 loss 曲线")
    parser.add_argument("--log", type=str, default="logs/train_gan.log", help="train_gan.log 路径")
    parser.add_argument("--out", type=str, default="gan_loss.png", help="输出图片路径")
    parser.add_argument("--dpi", type=int, default=150, help="图片 DPI")
    args = parser.parse_args()

    epochs, lossD_list, lossG_list = parse_gan_log(args.log)
    if not epochs:
        print(f"未在 {args.log} 中找到 loss 行（格式: Epoch N/M  lossD=...  lossG=...）")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lossD_list, label="lossD", color="C0", alpha=0.9)
    plt.plot(epochs, lossG_list, label="lossG", color="C1", alpha=0.9)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi)
    print(f"已保存: {args.out}（共 {len(epochs)} 个 epoch）")
    plt.close()


if __name__ == "__main__":
    main()
