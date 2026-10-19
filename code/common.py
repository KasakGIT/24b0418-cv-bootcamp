"""Shared utilities: seeding, data loading (GPU-resident), augmentation, evaluation."""
import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SEED = 0


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True is faster but not bit-for-bit deterministic; fine for this task.
    torch.backends.cudnn.benchmark = True


def load_data(data_dir, device="cuda"):
    """Returns a dict of GPU tensors. Images are uint8 NCHW; labels are int64."""
    X = np.load(os.path.join(data_dir, "train_images.npy"))   # (50000,32,32,3) uint8
    Xte = np.load(os.path.join(data_dir, "test_images.npy"))  # (10000,32,32,3) uint8
    df = pd.read_csv(os.path.join(data_dir, "labels.csv")).sort_values("index")
    assert (df["index"].values == np.arange(len(df))).all(), "labels.csv index mismatch"
    tr = np.loadtxt(os.path.join(data_dir, "train_idx.txt"), dtype=int)
    va = np.loadtxt(os.path.join(data_dir, "val_idx.txt"), dtype=int)
    assert len(set(tr) & set(va)) == 0, "train/val overlap"

    def to_t(a):  # NHWC uint8 -> NCHW uint8 tensor on device
        return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous().to(device)

    def lab(col, idx):
        return torch.from_numpy(df[col].values[idx].astype(np.int64)).to(device)

    d = dict(
        x_train=to_t(X[tr]), x_val=to_t(X[va]), x_test=to_t(Xte),
        train_idx=tr, val_idx=va,                      # original CIFAR indices (numpy)
        y_noisy_train=lab("noisy_label", tr), y_noisy_val=lab("noisy_label", va),
        y_synth_train=lab("synthetic_label", tr), y_synth_val=lab("synthetic_label", va),
        # CLEAN labels: reference model (Task 1) and *measuring* only. Never for Tasks 2-5 training/selection.
        y_clean_train=lab("clean_label", tr), y_clean_val=lab("clean_label", va),
    )
    # Normalisation stats come from training IMAGES only (no labels involved).
    f = d["x_train"].float() / 255.0
    d["mean"] = f.mean(dim=(0, 2, 3)).view(1, 3, 1, 1)
    d["std"] = f.std(dim=(0, 2, 3)).view(1, 3, 1, 1)
    return d


def augment(x):
    """x: float (B,3,32,32) in [0,1]. Random horizontal flip + random crop (pad 4)."""
    B, dev = x.size(0), x.device
    flip = torch.rand(B, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    xp = F.pad(x, (4, 4, 4, 4))                      # zero padding, like standard CIFAR recipes
    i = torch.randint(0, 9, (B,), device=dev)
    j = torch.randint(0, 9, (B,), device=dev)
    ar = torch.arange(32, device=dev)
    rows, cols = i[:, None] + ar, j[:, None] + ar    # (B,32)
    b = torch.arange(B, device=dev)[:, None, None, None]
    c = torch.arange(3, device=dev)[None, :, None, None]
    return xp[b, c, rows[:, None, :, None], cols[:, None, None, :]]


def prep(xb_uint8, d, train):
    x = xb_uint8.float() / 255.0
    if train:
        x = augment(x)
    return (x - d["mean"]) / d["std"]


@torch.no_grad()
def predict_logits(net, x_uint8, d, bs=1000):
    """Eval-mode logits for a uint8 image tensor (no augmentation)."""
    net.eval()
    out = [net(prep(x_uint8[i:i + bs], d, train=False)) for i in range(0, len(x_uint8), bs)]
    return torch.cat(out)


@torch.no_grad()
def accuracy(net, x_uint8, y, d):
    return (predict_logits(net, x_uint8, d).argmax(1) == y).float().mean().item()
