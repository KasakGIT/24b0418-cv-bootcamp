import sys, torch
sys.path.insert(0, "code")
from common import *
from model import ResNet18

set_seed(0)
d = load_data("data", device="cpu")
print("train", d["x_train"].shape, "val", d["x_val"].shape, "test", d["x_test"].shape)
print("mean", d["mean"].flatten().tolist(), "std", d["std"].flatten().tolist())
print("noise rate (train):", (d["y_noisy_train"] != d["y_clean_train"]).float().mean().item())

xb = prep(d["x_train"][:8], d, train=True)
print("augmented batch:", xb.shape, "mean %.3f std %.3f" % (xb.mean(), xb.std()))
net = ResNet18()
print("logits:", net(xb).shape)
print("untrained val acc (should be ~0.10):", accuracy(net, d["x_val"][:500], d["y_noisy_val"][:500], d))
