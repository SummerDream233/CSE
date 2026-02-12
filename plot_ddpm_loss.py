"""
从 train_ddpm.log 解析 loss 并绘图。
用法: python plot_ddpm_loss.py [--log logs/train_ddpm.log] [--out ddpm_loss.png]
"""

import argparse
import re
import matplotlib.pyplot as plt

# 指定为新罗马字体（Times New Roman）
plt.rcParams["font.family"] = "Times New Roman"


def parse_ddpm_log(log_path: str):
    """解析 DDPM 训练 log，返回 (epochs, loss_list)。"""
    pattern = re.compile(r"Epoch (\d+)/\d+\s+loss=([\d.]+)")
    epochs, loss_list = [], []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                loss_list.append(float(m.group(2)))
    return epochs, loss_list


def main():
    parser = argparse.ArgumentParser(description="绘制 DDPM 训练 loss 曲线")
    parser.add_argument("--log", type=str, default="logs/train_ddpm.log", help="train_ddpm.log 路径")
    parser.add_argument("--out", type=str, default="ddpm_loss.png", help="输出图片路径")
    parser.add_argument("--dpi", type=int, default=150, help="图片 DPI")
    args = parser.parse_args()

    epochs, loss_list = parse_ddpm_log(args.log)
    if not epochs:
        print(f"未在 {args.log} 中找到 loss 行（格式: Epoch N/M  loss=...）")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_list, color="C0", alpha=0.9)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi)
    print(f"已保存: {args.out}（共 {len(epochs)} 个 epoch）")
    plt.close()


if __name__ == "__main__":
    main()
