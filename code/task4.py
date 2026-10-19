"""Task 4: robust training methods (same recipe as Task 1). Resumable."""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_data
from train import train, write_pred_csv

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default=".")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--methods", default="ls,sce,filter")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke:
    args.out, args.epochs = "/tmp/cv_smoke", 2
dev = "cuda" if torch.cuda.is_available() else "cpu"
pred_dir, out_dir = os.path.join(args.out, "predictions"), os.path.join(args.out, "outputs")
os.makedirs(pred_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)

d = load_data(args.data, device=dev)
if args.smoke:
    for k in ["x_train", "y_noisy_train", "y_clean_train", "y_synth_train"]: d[k] = d[k][:500]
    for k in ["x_val", "y_noisy_val", "y_clean_val", "y_synth_val"]: d[k] = d[k][:200]
    d["x_test"] = d["x_test"][:200]
yn = d["y_noisy_train"]
vs = {"noisy": d["y_noisy_val"]}

def ls_loss(out, y):
    return F.cross_entropy(out, y, label_smoothing=0.2)

def sce_loss(out, y, alpha=0.5, beta=1.0, A=-4.0):
    ce = F.cross_entropy(out, y)
    p = F.softmax(out, 1).gather(1, y[:, None]).squeeze(1)
    rce = -A * (1 - p).mean()          # reverse CE with log(0) clipped to A
    return alpha * ce + beta * rce

def filter_subset(drop=0.35):
    s = pd.read_csv(os.path.join(pred_dir if not args.smoke else "predictions", "t3_noise_scores.csv"))["score"].values
    s = s[:len(yn)]
    keep = np.where(s <= np.quantile(s, 1 - drop))[0]
    return torch.as_tensor(keep, dtype=torch.long)

cfg = {"ls": dict(loss_fn=ls_loss), "sce": dict(loss_fn=sce_loss),
       "filter": dict(subset="FILTER")}
summary = {}
for m in args.methods.split(","):
    lp, pp = os.path.join(out_dir, f"t4_{m}_log.csv"), os.path.join(out_dir, f"t4_{m}_test_preds.pt")
    if os.path.exists(lp) and os.path.exists(pp):
        log, preds = pd.read_csv(lp), torch.load(pp); print(f"[skip] {m}")
    else:
        print(f"\n=== {m} ===", flush=True)
        kw = dict(cfg[m])
        if kw.get("subset") == "FILTER": kw["subset"] = filter_subset()
        r = train(d, yn, epochs=args.epochs, val_sets=vs, verbose=False, **kw)
        log, preds = r["log"], r["test_preds"]
        log.to_csv(lp, index=False); torch.save(preds, pp)
    write_pred_csv(preds[-1], os.path.join(pred_dir, f"t4_{m}.csv"))
    summary[m] = dict(final_val_acc_noisy=float(log.val_acc_noisy.iloc[-1]),
                      best_val_acc_noisy=float(log.val_acc_noisy.max()))
    print(m, summary[m], flush=True)
json.dump(summary, open(os.path.join(out_dir, "task4_summary.json"), "w"), indent=2)
