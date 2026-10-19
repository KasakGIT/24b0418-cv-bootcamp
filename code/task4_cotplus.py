"""Co-teaching variants: --forget rate, optional --plus (disagreement-based, Co-teaching+). Same recipe as Task 1."""
import os, sys, json, math, time, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_data, prep, set_seed, SEED
from model import ResNet18
from train import get_logits, write_pred_csv, LR, WD, BS, MOMENTUM

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data"); ap.add_argument("--out", default=".")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--forget", type=float, default=0.35); ap.add_argument("--ramp", type=int, default=10)
ap.add_argument("--plus", action="store_true"); ap.add_argument("--init", type=int, default=20)
ap.add_argument("--labels", default="noisy"); ap.add_argument("--tag", default="cotplus")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke:
    args.out, args.epochs, args.ramp, args.init = "/tmp/cv_smoke", 2, 1, 1
dev = "cuda" if torch.cuda.is_available() else "cpu"; use_amp = dev == "cuda"
pred_dir, out_dir = os.path.join(args.out, "predictions"), os.path.join(args.out, "outputs")
os.makedirs(pred_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)

d = load_data(args.data, device=dev)
if args.smoke:
    for k in ["x_train", "y_noisy_train", "y_clean_train", "y_synth_train"]: d[k] = d[k][:500]
    for k in ["x_val", "y_noisy_val", "y_clean_val", "y_synth_val"]: d[k] = d[k][:200]
    d["x_test"] = d["x_test"][:200]
synth = args.labels == "synth"
x = d["x_train"]; yn = d["y_synth_train"] if synth else d["y_noisy_train"]
yval = d["y_synth_val"] if synth else d["y_noisy_val"]
n = len(x); spe = math.ceil(n / BS)

set_seed(SEED)
nets = [ResNet18().to(dev) for _ in range(2)]
opts = [torch.optim.SGD(m.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD, nesterov=True) for m in nets]
scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.epochs * spe) for o in opts]
scalers = [torch.amp.GradScaler(dev, enabled=use_amp) for _ in nets]
probs = lambda split: [F.softmax(get_logits(m, d[split], d), 1) for m in nets]

log, test_preds = [], []
for ep in range(1, args.epochs + 1):
    t0 = time.time()
    forget = args.forget * min(1.0, (ep - 1) / args.ramp)
    perm = torch.randperm(n, device=dev); run_loss = 0.0; seen = 0
    for i in range(0, n, BS):
        b = perm[i:i + BS]
        if len(b) < 2: continue
        xb, yb = prep(x[b], d, train=True), yn[b]
        for m in nets: m.train()
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
            outs = [m(xb) for m in nets]
        losses = [F.cross_entropy(o.float(), yb, reduction="none") for o in outs]
        pool = torch.arange(len(b), device=dev)
        if args.plus and ep > args.init:
            dis = (outs[0].argmax(1) != outs[1].argmax(1)).nonzero().squeeze(1)
            if len(dis) >= 16: pool = dis                      # disagreement set (else fall back to whole batch)
        k = max(1, int(round((1 - forget) * len(pool))))
        sel = [pool[torch.argsort(l.detach()[pool])[:k]] for l in losses]
        for j in range(2):                                     # net j learns from the OTHER net's picks
            loss = losses[j][sel[1 - j]].mean()
            opts[j].zero_grad(set_to_none=True)
            scalers[j].scale(loss).backward()
            scalers[j].step(opts[j]); scalers[j].update(); scheds[j].step()
        run_loss += loss.item() * len(b); seen += len(b)
    pv = probs("x_val"); ens = (pv[0] + pv[1]).argmax(1)
    row = {"epoch": ep, "train_loss": run_loss / seen, "forget": forget,
           "val_acc_noisy": (ens == yval).float().mean().item(),
           "val_acc_clean": (ens == d["y_clean_val"]).float().mean().item(),   # measurement only
           "sec": time.time() - t0}
    pt = probs("x_test"); test_preds.append((pt[0] + pt[1]).argmax(1).cpu())
    log.append(row)
    if ep % 10 == 0 or ep == args.epochs:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)

pre = "t5_" if synth else "t4_"
log = pd.DataFrame(log); preds = torch.stack(test_preds)
log.to_csv(os.path.join(out_dir, f"{pre}{args.tag}_log.csv"), index=False)
torch.save(preds, os.path.join(out_dir, f"{pre}{args.tag}_test_preds.pt"))
write_pred_csv(preds[-1], os.path.join(pred_dir, f"t5_best_synth_{args.tag}.csv" if synth else f"t4_{args.tag}.csv"))
s = dict(final_val_acc_noisy=float(log.val_acc_noisy.iloc[-1]), best_val_acc_noisy=float(log.val_acc_noisy.max()),
         final_val_acc_clean=float(log.val_acc_clean.iloc[-1]))
json.dump(s, open(os.path.join(out_dir, f"{pre}{args.tag}_summary.json"), "w"), indent=2)
print(s)
