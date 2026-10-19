"""Task 5: synthetic noise. CE baseline vs filter+retrain. Filter method uses 2 networks:
(1) scoring net, 60-epoch schedule stopped at epoch 30 (pre-memorization), scores its own training images;
(2) retrain on kept images. Model selection on SYNTH val only; clean labels for MEASUREMENT only."""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_data
from train import train, get_logits, write_pred_csv

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default=".")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--score_epoch", type=int, default=30)
ap.add_argument("--drop", type=float, default=0.35)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke:
    args.out, args.epochs, args.score_epoch = "/tmp/cv_smoke", 2, 1
dev = "cuda" if torch.cuda.is_available() else "cpu"
pred_dir, out_dir = os.path.join(args.out, "predictions"), os.path.join(args.out, "outputs")
os.makedirs(pred_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)

d = load_data(args.data, device=dev)
if args.smoke:
    for k in ["x_train", "y_noisy_train", "y_clean_train", "y_synth_train"]: d[k] = d[k][:500]
    for k in ["x_val", "y_noisy_val", "y_clean_val", "y_synth_val"]: d[k] = d[k][:200]
    d["x_test"] = d["x_test"][:200]
ys = d["y_synth_train"]; N = len(ys); vs = {"synth": d["y_synth_val"]}
print("synthetic noise rate:", float((ys != d["y_clean_train"]).float().mean()))

def run(name, **kw):
    lp, pp = os.path.join(out_dir, f"{name}_log.csv"), os.path.join(out_dir, f"{name}_test_preds.pt")
    if os.path.exists(lp) and os.path.exists(pp):
        print(f"[skip] {name}"); return pd.read_csv(lp), torch.load(pp)
    print(f"\n=== {name} ===", flush=True)
    r = train(d, ys, val_sets=vs, verbose=False, **kw)
    r["log"].to_csv(lp, index=False); torch.save(r["test_preds"], pp)
    run.net = r["net"]
    return r["log"], r["test_preds"]

# 1. CE baseline
b_log, b_pred = run("t5_baseline", epochs=args.epochs)
write_pred_csv(b_pred[-1], os.path.join(pred_dir, "t5_baseline_synth.csv"))
print("baseline final synth-val:", float(b_log.val_acc_synth.iloc[-1]), flush=True)

# 2. network 1: scoring net (stopped before memorization), scores its own training images
sp = os.path.join(out_dir, "t5_score_logits.pt")
if os.path.exists(sp):
    logits = torch.load(sp); print("[skip] scoring net")
else:
    print("\n=== scoring net ===", flush=True)
    r = train(d, ys, epochs=args.epochs, stop_epoch=args.score_epoch, val_sets=vs, verbose=False)
    logits = get_logits(r["net"], d["x_train"], d).cpu(); torch.save(logits, sp)
lp_ = torch.log_softmax(logits, 1); yc = ys.cpu()
loss = -lp_[torch.arange(N), yc]
score = (lp_.argmax(1) != yc).float() + 1e-3 * loss.clamp(max=100)
wrong = (ys != d["y_clean_train"]).cpu().numpy()                      # measurement only
from sklearn.metrics import roc_auc_score
print(f"score AUROC (measurement): {roc_auc_score(wrong, score.numpy()):.4f}")
keep_np = np.where(score.numpy() <= np.quantile(score.numpy(), 1 - args.drop))[0]
keep = torch.as_tensor(keep_np, dtype=torch.long)
dropped = np.setdiff1d(np.arange(N), keep_np)
print(f"keeps {len(keep)}; fraction of dropped truly wrong: {wrong[dropped].mean():.3f}")

# 3. network 2: retrain on kept images
f_log, f_pred = run("t5_filter", epochs=args.epochs, subset=keep)
write_pred_csv(f_pred[-1], os.path.join(pred_dir, "t5_best_synth.csv"))
summary = dict(baseline_final_val_acc_synth=float(b_log.val_acc_synth.iloc[-1]),
               filter_final_val_acc_synth=float(f_log.val_acc_synth.iloc[-1]),
               networks_for_filter_method=2)
json.dump(summary, open(os.path.join(out_dir, "task5_summary.json"), "w"), indent=2)
print(summary)
