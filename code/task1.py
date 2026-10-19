"""Task 1 (+ Task 2 dynamics). Confusion matrix, clean-ref model, noisy-CE baseline. Resumable."""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_data
from train import train, write_pred_csv

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default=".", help="project root; predictions/ and outputs/ go here")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--smoke", action="store_true", help="tiny CPU/GPU sanity run, writes to /tmp")
args = ap.parse_args()
if args.smoke:
    args.out, args.epochs = "/tmp/cv_smoke", 2

dev = "cuda" if torch.cuda.is_available() else "cpu"
pred_dir, out_dir = os.path.join(args.out, "predictions"), os.path.join(args.out, "outputs")
os.makedirs(pred_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)
print("device:", dev, "| epochs:", args.epochs)

d = load_data(args.data, device=dev)
if args.smoke:
    for k in ["x_train", "y_noisy_train", "y_clean_train", "y_synth_train"]: d[k] = d[k][:500]
    for k in ["x_val", "y_noisy_val", "y_clean_val", "y_synth_val"]: d[k] = d[k][:200]
    d["x_test"] = d["x_test"][:200]

# ---- 1. noise rate + confusion matrix (clean labels used for MEASURING only)
classes = open(os.path.join(args.data, "classes.txt")).read().split()
yc, yn = d["y_clean_train"].cpu().numpy(), d["y_noisy_train"].cpu().numpy()
noise_rate = float((yc != yn).mean())
cm = np.zeros((10, 10), dtype=int); np.add.at(cm, (yc, yn), 1)   # rows=clean, cols=noisy
pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(out_dir, "t1_confusion_matrix.csv"))
print(f"noise rate: {noise_rate:.4f}")

# ---- 2. the two runs
vs = {"noisy": d["y_noisy_val"], "clean": d["y_clean_val"]}

def run(name, labels, track):
    lp, pp = os.path.join(out_dir, f"{name}_log.csv"), os.path.join(out_dir, f"{name}_test_preds.pt")
    if os.path.exists(lp) and os.path.exists(pp):
        print(f"[skip] {name} already done"); return pd.read_csv(lp), torch.load(pp)
    print(f"\n=== training {name} ===")
    r = train(d, labels, epochs=args.epochs, val_sets=vs, track_train=track)
    r["log"].to_csv(lp, index=False)
    torch.save(r["test_preds"], pp)                                   # (epochs, 10000), every epoch
    torch.save(r["net"].state_dict(), os.path.join(out_dir, f"{name}_final.pt"))
    return r["log"], r["test_preds"]

clean_log, clean_preds = run("clean_ref", d["y_clean_train"], track=False)
noisy_log, noisy_preds = run("noisy_ce", d["y_noisy_train"], track=True)

# ---- 3. required prediction files (final epoch) + dynamics file
write_pred_csv(clean_preds[-1], os.path.join(pred_dir, "t1_clean_ref.csv"))
write_pred_csv(noisy_preds[-1], os.path.join(pred_dir, "t1_noisy_ce.csv"))
noisy_log[["epoch", "train_acc_correct", "train_acc_mislabeled", "val_acc_noisy", "val_acc_clean"]] \
    .to_csv(os.path.join(pred_dir, "t2_dynamics.csv"), index=False)

summary = dict(noise_rate=noise_rate,
               val_acc_noisy_labels__clean_ref_model=float(clean_log.val_acc_noisy.iloc[-1]),
               val_acc_noisy_labels__noisy_ce_model=float(noisy_log.val_acc_noisy.iloc[-1]))
json.dump(summary, open(os.path.join(out_dir, "task1_summary.json"), "w"), indent=2)
print("\n", json.dumps(summary, indent=2))
print("best noisy-val epoch (noisy run):", int(noisy_log.val_acc_noisy.idxmax()) + 1)
