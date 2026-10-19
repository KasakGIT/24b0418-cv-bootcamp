"""One training function reused by every experiment (identical recipe across methods)."""
import math, time
import torch
import torch.nn.functional as F
import pandas as pd
from common import prep, set_seed, SEED
from model import ResNet18

LR, WD, BS, MOMENTUM = 0.1, 5e-4, 128, 0.9


@torch.no_grad()
def get_logits(net, x_uint8, d, bs=1000):
    """Eval-mode logits (no augmentation), fp16 autocast on GPU."""
    net.eval()
    dev = x_uint8.device
    outs = []
    for i in range(0, len(x_uint8), bs):
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=(dev.type == "cuda")):
            outs.append(net(prep(x_uint8[i:i + bs], d, train=False)).float())
    return torch.cat(outs)


def train(d, labels, epochs=60, subset=None, loss_fn=None, val_sets=None,
          track_train=False, seed=SEED, verbose=True):
    """
    d          : dict from common.load_data
    labels     : LongTensor (45000,) -- the labels to TRAIN on (noisy / clean / synthetic)
    subset     : optional LongTensor of positions into the 45k train set (for filtering)
    loss_fn    : f(logits, targets) -> scalar. Default: cross-entropy
    val_sets   : dict name -> LongTensor (5000,) of val labels to score against each epoch
    track_train: if True, log train acc on correctly-labeled / mislabeled subsets
                 (uses clean labels for MEASUREMENT only -- Task 2 curves)
    Returns dict(net, log, test_preds): log is a DataFrame, test_preds is (epochs, 10000).
    """
    set_seed(seed)
    dev = d["x_train"].device
    use_amp = dev.type == "cuda"
    loss_fn = loss_fn or F.cross_entropy
    val_sets = val_sets or {}

    x = d["x_train"]
    idx = torch.arange(len(x), device=dev) if subset is None else subset.to(dev)
    n = len(idx)
    steps_per_epoch = math.ceil(n / BS)

    net = ResNet18().to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=WD, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler(dev.type, enabled=use_amp)

    if track_train:
        m_correct = d["y_noisy_train"] == d["y_clean_train"]   # measurement only
        m_wrong = ~m_correct

    log, test_preds = [], []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        net.train()
        perm = idx[torch.randperm(n, device=dev)]
        run_loss, seen = 0.0, 0
        for i in range(0, n, BS):
            b = perm[i:i + BS]
            if len(b) < 2:           # BatchNorm needs >1 sample
                continue
            xb, yb = prep(x[b], d, train=True), labels[b]
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                out = net(xb)
            loss = loss_fn(out.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(b)
            seen += len(b)

        row = {"epoch": ep, "train_loss": run_loss / seen, "lr": opt.param_groups[0]["lr"]}
        # (compute val logits once, reuse for every label set)
        vlog = get_logits(net, d["x_val"], d).argmax(1)
        for name, yv in val_sets.items():
            row[f"val_acc_{name}"] = (vlog == yv).float().mean().item()
        if track_train:
            tp = get_logits(net, x, d).argmax(1)
            row["train_acc_correct"] = (tp[m_correct] == labels[m_correct]).float().mean().item()
            row["train_acc_mislabeled"] = (tp[m_wrong] == labels[m_wrong]).float().mean().item()
        test_preds.append(get_logits(net, d["x_test"], d).argmax(1).cpu())
        row["sec"] = time.time() - t0
        log.append(row)
        if verbose:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    return dict(net=net, log=pd.DataFrame(log), test_preds=torch.stack(test_preds))


def write_pred_csv(preds, path):
    """preds: 1-D tensor of 10000 class ids -> id,pred csv (the format check_format.py wants)."""
    pd.DataFrame({"id": range(len(preds)), "pred": preds.numpy().astype(int)}).to_csv(path, index=False)


if __name__ == "__main__":
    # Smoke test: tiny slice, 2 epochs, runs on a laptop CPU
    import sys
    from common import load_data
    d = load_data("data", device="cpu")
    for k in ["x_train", "y_noisy_train", "y_clean_train", "y_synth_train"]:
        d[k] = d[k][:1000]
    for k in ["x_val", "y_noisy_val", "y_clean_val", "y_synth_val"]:
        d[k] = d[k][:500]
    d["x_test"] = d["x_test"][:500]
    r = train(d, d["y_noisy_train"], epochs=2, track_train=True,
              val_sets={"noisy": d["y_noisy_val"], "clean": d["y_clean_val"]})
    print(r["log"].T)
    print("test_preds:", r["test_preds"].shape)
