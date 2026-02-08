import os
import logging
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from PIL import Image

IMG_SIZE = 64

_train_logger = None


def setup_train_logging(log_dir: str, run_name: str):
    global _train_logger
    ensure_dir(log_dir)
    log_file = os.path.join(log_dir, f"{run_name}.log")

    _train_logger = logging.getLogger("train")
    _train_logger.setLevel(logging.DEBUG)
    _train_logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    _train_logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    _train_logger.addHandler(ch)
    return _train_logger


def get_train_logger():
    global _train_logger
    if _train_logger is None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        return logging.getLogger("train")
    return _train_logger


class TinyImageNet200(Dataset):
    def __init__(self, root: str, split: str, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = []  # list of (path, label)

        if split == "train":
            train_dir = os.path.join(root, "train")
            if not os.path.isdir(train_dir):
                raise FileNotFoundError(f"Tiny-ImageNet train 目录不存在: {train_dir}")
            class_dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
            self.wnid_to_idx = {wnid: i for i, wnid in enumerate(class_dirs)}
            for wnid in class_dirs:
                img_dir = os.path.join(train_dir, wnid, "images")
                if not os.path.isdir(img_dir):
                    img_dir = os.path.join(train_dir, wnid)
                for f in os.listdir(img_dir):
                    if f.lower().endswith((".jpeg", ".jpg", ".png")):
                        self.samples.append((os.path.join(img_dir, f), self.wnid_to_idx[wnid]))
        else:
            val_img_dir = os.path.join(root, "val", "images")
            if not os.path.isdir(val_img_dir):
                val_img_dir = os.path.join(root, "val")
            if not os.path.isdir(val_img_dir):
                raise FileNotFoundError(f"Tiny-ImageNet val 目录不存在: {os.path.join(root, 'val')}")
            for f in sorted(os.listdir(val_img_dir)):
                if f.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((os.path.join(val_img_dir, f), 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def images_to_uint8_0_255(x: torch.Tensor) -> torch.Tensor:
    x = (x.clamp(-1, 1) + 1) / 2.0
    x = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
    return x


def _normalize_96_rgb():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def get_dataloader(
    data_root: str,
    batch_size: int,
    num_workers: int,
    split: str = "train",
    data_fraction: float = 1.0,
    seed: int = 42,
):
    if split == "unlabeled":
        split = "train"
    for sub in ("archive/tiny-imagenet-200", "tiny-imagenet-200"):
        root = os.path.join(data_root, sub)
        if os.path.isdir(root):
            break
    else:
        raise FileNotFoundError(
            f"在 {data_root} 下未找到 Tiny-ImageNet-200（请放于 archive/tiny-imagenet-200 或 tiny-imagenet-200）"
        )
    tiny_split = "train" if split == "train" else "val"
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        _normalize_96_rgb(),
    ])
    ds = TinyImageNet200(root=root, split=tiny_split, transform=tfm)

    if data_fraction < 1.0:
        n_total = len(ds)
        n_use = max(1, int(n_total * data_fraction))
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_total, generator=g)[:n_use].tolist()
        ds = Subset(ds, indices)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return ds, dl
