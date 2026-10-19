"""Task 2: pick early-stop epoch by NOISY val accuracy only, export predictions."""
import os, json, torch, pandas as pd, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import write_pred_csv

log = pd.read_csv("outputs/noisy_ce_log.csv")
preds = torch.load("outputs/noisy_ce_test_preds.pt")      # (60, 10000)
ep = int(log.val_acc_noisy.idxmax()) + 1                  # noisy val only
write_pred_csv(preds[ep - 1], "predictions/t2_early_stop.csv")

# memorization start: first epoch where mislabeled train acc exceeds its early baseline by 3 pts
base = log.train_acc_mislabeled.iloc[9:30].mean()   # epochs 10-30
late = log[log.epoch > 30]
mem = int(late.epoch[late.train_acc_mislabeled > base + 0.03].iloc[0])
out = dict(memorization_start_epoch=mem, early_stop_epoch=ep,
           val_acc_noisy_labels_at_early_stop=float(log.val_acc_noisy.iloc[ep - 1]))
json.dump(out, open("outputs/task2_summary.json", "w"), indent=2)
print(out)
