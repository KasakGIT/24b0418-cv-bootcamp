"""Extra Task 4 candidate: filter (top 20%) + relabel (next 15%) with CV predictions, then retrain. Same recipe."""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_data
from train import train, write_pred_csv

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default=".")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--drop", type=float, default=0.20)
ap.add_argument("--relabel", type=float, default=0.15)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke:
    args.out, args.epochs = "/tmp/cv_smoke", 2
dev = "cuda" if torch.cuda.is_available() else "cpu"
pred_dir, out_dir = os.path.join(args.out, "predictions"), os.path.join(args.out, "outputs")

d = load_data(args.data, device=dev)
if args.smoke:
    for k in ["x_train", "y_noisy_train", "y_clean_train", "y_synth_train"]: d[k] = d[k][:500]
    for k in ["x_val", "y_noisy_val", "y_clean_val", "y_synth_val"]: d[k] = d[k][:200]
    d["x_test"] = d["x_test"][:200]
yn = d["y_noisy_train"].cpu(); N = len(yn)

# rebuild the Task 3 cross-validated logits (same fold split as task3.py)
folds = torch.randperm(N, generator=torch.Generator().manual_seed(0)).chunk(5)
logits = torch.zeros(N, 10)
for k in range(5):
    logits[folds[k]] = torch.load(os.path.join(out_dir, f"t3_fold{k}_logits.pt"))
lp = torch.log_softmax(logits, 1); pred = lp.argmax(1)
score = (pred != yn).float() + 1e-3 * (-lp[torch.arange(N), yn]).clamp(max=100)   # same signal as Task 3

order = torch.argsort(score, descending=True)
n_drop, n_rel = int(args.drop * N), int(args.relabel * N)
drop_idx, rel_idx = order[:n_drop], order[n_drop:n_drop + n_rel]
keep = torch.ones(N, dtype=torch.bool); keep[drop_idx] = False
new_labels = yn.clone(); new_labels[rel_idx] = pred[rel_idx]

# measurement only (clean labels never used for training/selection)
yc = d["y_clean_train"].cpu()
print(f"dropped {n_drop}: {(yc[drop_idx] != yn[drop_idx]).float().mean():.3f} truly wrong")
print(f"relabeled {n_rel}: old label correct {(yc[rel_idx] == yn[rel_idx]).float().mean():.3f} -> new label correct {(yc[rel_idx] == new_labels[rel_idx]).float().mean():.3f}", flush=True)

r = train(d, new_labels.to(dev), epochs=args.epochs, subset=torch.where(keep)[0],
          val_sets={"noisy": d["y_noisy_val"], "clean": d["y_clean_val"]}, verbose=False)
log, preds = r["log"], r["test_preds"]
os.makedirs(out_dir, exist_ok=True); os.makedirs(pred_dir, exist_ok=True)
log.to_csv(os.path.join(out_dir, "t4_relabel_log.csv"), index=False)
torch.save(preds, os.path.join(out_dir, "t4_relabel_test_preds.pt"))
write_pred_csv(preds[-1], os.path.join(pred_dir, "t4_relabel.csv"))
s = dict(final_val_acc_noisy=float(log.val_acc_noisy.iloc[-1]), best_val_acc_noisy=float(log.val_acc_noisy.max()),
         final_val_acc_clean=float(log.val_acc_clean.iloc[-1]))
json.dump(s, open(os.path.join(out_dir, "task4_relabel_summary.json"), "w"), indent=2)
print(s)
