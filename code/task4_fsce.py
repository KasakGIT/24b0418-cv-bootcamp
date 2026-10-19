"""Task 4 candidate: filter (Task 3 scores, drop 35%) + symmetric CE loss. Same recipe otherwise."""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_data
from train import train, write_pred_csv

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default=".")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--drop", type=float, default=0.35)
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
yn = d["y_noisy_train"]; N = len(yn)

def sce_loss(out, y, alpha=0.5, beta=1.0, A=-4.0):
    ce = F.cross_entropy(out, y)
    p = F.softmax(out, 1).gather(1, y[:, None]).squeeze(1)
    return alpha * ce + beta * (-A) * (1 - p).mean()

# Task 3 scores (same fold split / signal as task3.py)
logits = torch.zeros(N, 10)
if args.smoke:
    logits = torch.randn(N, 10)          # smoke test only
else:
    folds = torch.randperm(N, generator=torch.Generator().manual_seed(0)).chunk(5)
    for k in range(5):
        logits[folds[k]] = torch.load(os.path.join(out_dir, f"t3_fold{k}_logits.pt"))
lp = torch.log_softmax(logits, 1); ync = yn.cpu()
score = (lp.argmax(1) != ync).float() + 1e-3 * (-lp[torch.arange(N), ync]).clamp(max=100)
keep = torch.as_tensor(np.where(score.numpy() <= np.quantile(score.numpy(), 1 - args.drop))[0], dtype=torch.long)

r = train(d, yn, epochs=args.epochs, subset=keep, loss_fn=sce_loss,
          val_sets={"noisy": d["y_noisy_val"], "clean": d["y_clean_val"]}, verbose=False)   # clean val = measurement only
log, preds = r["log"], r["test_preds"]
log.to_csv(os.path.join(out_dir, "t4_fsce_log.csv"), index=False)
torch.save(preds, os.path.join(out_dir, "t4_fsce_test_preds.pt"))
write_pred_csv(preds[-1], os.path.join(pred_dir, "t4_fsce.csv"))
s = dict(final_val_acc_noisy=float(log.val_acc_noisy.iloc[-1]), best_val_acc_noisy=float(log.val_acc_noisy.max()),
         final_val_acc_clean=float(log.val_acc_clean.iloc[-1]))
json.dump(s, open(os.path.join(out_dir, "task4_fsce_summary.json"), "w"), indent=2)
print(s)
