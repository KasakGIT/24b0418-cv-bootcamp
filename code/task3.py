"""Task 3: 5-fold cross-validated noise scores. Resumable. Clean labels used only to MEASURE AUROC."""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_data
from train import train, get_logits

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default=".")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke:
    args.out, args.epochs = "/tmp/cv_smoke", 2
dev = "cuda" if torch.cuda.is_available() else "cpu"
pred_dir, out_dir = os.path.join(args.out, "predictions"), os.path.join(args.out, "outputs")
os.makedirs(pred_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)

d = load_data(args.data, device=dev)
tidx = np.loadtxt(os.path.join(args.data, "train_idx.txt"), dtype=int)
if args.smoke:
    for k in ["x_train", "y_noisy_train", "y_clean_train", "y_synth_train"]: d[k] = d[k][:500]
    for k in ["x_val", "y_noisy_val", "y_clean_val", "y_synth_val"]: d[k] = d[k][:200]
    d["x_test"] = d["x_test"][:200]; tidx = tidx[:500]
N = len(d["x_train"]); assert len(tidx) == N
yn = d["y_noisy_train"]

g = torch.Generator().manual_seed(0)
folds = torch.randperm(N, generator=g).chunk(5)
logits = torch.zeros(N, 10)

for k in range(5):
    fp = os.path.join(out_dir, f"t3_fold{k}_logits.pt")
    held = folds[k]
    if os.path.exists(fp):
        print(f"[skip] fold {k}"); logits[held] = torch.load(fp); continue
    train_pos = torch.cat([folds[j] for j in range(5) if j != k])
    print(f"=== fold {k}: train on {len(train_pos)}, score {len(held)} ===", flush=True)
    r = train(d, yn, epochs=args.epochs, subset=train_pos, val_sets={"noisy": d["y_noisy_val"]}, verbose=False)
    print(r["log"].iloc[-1][["epoch", "train_loss", "val_acc_noisy", "sec"]].to_dict(), flush=True)
    lg = get_logits(r["net"], d["x_train"][held.to(dev)], d).cpu()
    torch.save(lg, fp); logits[held] = lg

# ---- signals (higher = more likely mislabeled)
lp = torch.log_softmax(logits, 1); p = lp.exp(); ync = yn.cpu()
p_noisy = p[torch.arange(N), ync]
signals = {
    "loss": -lp[torch.arange(N), ync],
    "margin": p.max(1).values - p_noisy,
    "disagree": (p.argmax(1) != ync).float() + 1e-3 * (-lp[torch.arange(N), ync]).clamp(max=100),
}
# ---- measurement with clean labels (allowed: own Task 3 check)
from sklearn.metrics import roc_auc_score
wrong = (d["y_clean_train"] != yn).cpu().numpy(); kk = int(wrong.sum()); res = {}
for name, s in signals.items():
    s = s.numpy(); top = np.argsort(-s)[:kk]
    res[name] = dict(auroc=float(roc_auc_score(wrong, s)), precision_at_k=float(wrong[top].mean()))
    print(name, res[name])
best = max(res, key=lambda n: res[n]["auroc"])
print("best signal:", best)
pd.DataFrame({"index": tidx, "score": signals[best].numpy()}).to_csv(os.path.join(pred_dir, "t3_noise_scores.csv"), index=False)
json.dump(dict(method=f"5-fold CV, {args.epochs} epochs, signal={best}", results=res),
          open(os.path.join(out_dir, "task3_summary.json"), "w"), indent=2)
