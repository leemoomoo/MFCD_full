import math
# coding: utf-8
import sys, os, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models", "ncdm"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from NCDM_Fluency import NCDMFluency
from assist09_dataset import build_assist09_splits_from_folder

DATA_DIR = r"D:\12487\NCDM PLUS2\data\a0910"
OUT_DIR  = r"D:\12487\NCDM PLUS2\outputs\loss_curves"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

EPOCHS = 30
BATCH = 512

print("Loading data...")
train_ds, valid_ds, test_ds, meta = build_assist09_splits_from_folder(DATA_DIR, use_fluency=True)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_ds, batch_size=BATCH, shuffle=False, num_workers=0)

print(f"  train={meta['n_train']}, valid={meta['n_valid']}, test={meta['n_test']}")
print(f"  student_n={meta['student_n']}, exer_n={meta['exer_n']}, knowledge_n={meta['knowledge_n']}")

print("Building NCDMFluency model...")
model = NCDMFluency(
    knowledge_n=meta["knowledge_n"],
    exer_n=meta["exer_n"],
    student_n=meta["student_n"],
    lr=0.002, fluency_lambda=0.5, dropout=0.5,
)
# Unified weight design: all aux losses share the same weight
model.net.to(DEVICE)
optimizer = optim.Adam(model.net.parameters(), lr=0.002)

# Training records: 5 loss curves
history = {
    "epoch": [], "L_total": [], "L_y": [],
    "L_opp": [], "L_acc": [], "L_sta": [],
    "valid_auc": [], "valid_acc": [],
}

print(f"\nTraining for {EPOCHS} epochs...")

for epoch_i in range(EPOCHS):
    model.net.train()
    losses_total, losses_y, losses_opp, losses_acc, losses_sta = [], [], [], [], []

    for batch in train_loader:
        uid, iid, kq, y, opp, acc, sta = [x.to(DEVICE) for x in batch]

        # Forward with aux outputs
        out = model.net(uid, iid, kq, opp, acc, sta, return_aux=True)

        # Formula (18): BCE for y
        ly = model.loss_y(out["p"], y)

        # Formula (19): MSE for opp (exposure intensity)
        lopp = model.loss_opp(out["opp_hat"], out["opp_tgt"])

        # Formula (20): SmoothL1 for acc (empirical accuracy)
        lacc = model.loss_acc(out["acc_hat"], out["acc_tgt"])

        # Formula (21): SmoothL1 for sta (response consistency)
        lsta = model.loss_sta(out["sta_hat"], out["sta_tgt"])

        # Formula (17): Total loss (unified weight lambda)
        loss = (1 - model.fluency_lambda) * ly + model.fluency_lambda * (lopp + lacc + lsta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_total.append(loss.item())
        losses_y.append(ly.item())
        losses_opp.append(lopp.item())
        losses_acc.append(lacc.item())
        losses_sta.append(lsta.item())

    # Record epoch means
    history["epoch"].append(epoch_i)
    history["L_total"].append(float(np.mean(losses_total)))
    history["L_y"].append(float(np.mean(losses_y)))
    history["L_opp"].append(float(np.mean(losses_opp)))
    history["L_acc"].append(float(np.mean(losses_acc)))
    history["L_sta"].append(float(np.mean(losses_sta)))

    # Evaluate on validation set
    model.net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in valid_loader:
            uid, iid, kq, y, opp, acc, sta = [x.to(DEVICE) for x in batch]
            pred = model.net(uid, iid, kq, opp, acc, sta, return_aux=False)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())

    from sklearn.metrics import roc_auc_score, accuracy_score
    auc = roc_auc_score(y_true, y_pred)
    acc_val = accuracy_score(y_true, np.array(y_pred) >= 0.5)
    history["valid_auc"].append(auc)
    history["valid_acc"].append(acc_val)

    print(f"Epoch {epoch_i:02d}:  L_total={history['L_total'][-1]:.4f}  "
          f"L_y={history['L_y'][-1]:.4f}  L_opp={history['L_opp'][-1]:.4f}  "
          f"L_acc={history['L_acc'][-1]:.4f}  L_sta={history['L_sta'][-1]:.4f}  "
          f"valid_auc={auc:.4f}")

# ===== Save raw data =====
df = pd.DataFrame(history)
csv_path = os.path.join(OUT_DIR, "loss_curves.csv")
df.to_csv(csv_path, index=False, float_format="%.6f")
print(f"\nSaved: {csv_path}")

# ===== Figure 1a: Main Losses =====
fig1, ax1 = plt.subplots(figsize=(5.5, 3.2))

ax1.plot(history["epoch"], history["L_total"], "k-", lw=2.5, label=r"$L_{\mathrm{Total}}$")
ax1.plot(history["epoch"], history["L_y"], "r-", lw=1.8, label=r"$L_y$")
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Loss", fontsize=11)
ax1.legend(fontsize=9.5, loc="upper right", framealpha=0.9)
ax1.set_xlim(left=-0.3, right=EPOCHS - 1 + 0.3)
ax1.grid(alpha=0.15, linestyle="--")
ax1.tick_params(labelsize=9)

fig1.tight_layout(pad=0.8)
path_main = os.path.join(OUT_DIR, "main_losses.png")
fig1.savefig(path_main, dpi=200, bbox_inches="tight")
fig1.savefig(os.path.join(OUT_DIR, "main_losses.pdf"), bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {path_main}")

# ===== Figure 1b: Auxiliary Losses =====
fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))

ax2b = ax2.twinx()
ax2.plot(history["epoch"], history["L_acc"], "g-", lw=1.8, label=r"$L_{\mathrm{ea}}$")
ax2.plot(history["epoch"], history["L_sta"], "m-", lw=1.8, label=r"$L_{\mathrm{rc}}$")
ax2b.plot(history["epoch"], history["L_opp"], "b-", lw=1.8, label=r"$L_{\mathrm{ei}}$")
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9.5, framealpha=0.9)
ax2b.set_ylabel(r"$L_{\mathrm{ei}}$", fontsize=11, color="b")
ax2b.tick_params(axis="y", labelcolor="b", labelsize=9)
ax2.set_xlim(left=-0.3, right=EPOCHS - 1 + 0.3)
ax2.grid(alpha=0.15, linestyle="--")
ax2.tick_params(labelsize=9)

fig2.tight_layout(pad=0.8)
path_aux = os.path.join(OUT_DIR, "aux_losses.png")
fig2.savefig(path_aux, dpi=200, bbox_inches="tight")
fig2.savefig(os.path.join(OUT_DIR, "aux_losses.pdf"), bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {path_aux}")

# ===== Figure 2: Validation Performance =====
fig, ax1 = plt.subplots(figsize=(6, 3.2))

ax1.plot(history["epoch"], history["valid_auc"], "b-o", lw=2.2, ms=6, label="Valid AUC")
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("AUC", fontsize=11, color="b")
ax1.tick_params(axis="y", labelsize=9, labelcolor="b")
ax1.set_xlim(left=-0.3, right=EPOCHS - 1 + 0.3)
ax1.grid(alpha=0.15, linestyle="--")

ax1b = ax1.twinx()
ax1b.plot(history["epoch"], history["valid_acc"], "g--s", lw=2.2, ms=6, label="Valid ACC")
ax1b.set_ylabel("ACC", fontsize=11, color="g")
ax1b.tick_params(axis="y", labelsize=9, labelcolor="g")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax1b.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=10, framealpha=0.9)

fig.tight_layout(pad=0.8)
path_val = os.path.join(OUT_DIR, "valid_curves.png")
fig.savefig(path_val, dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "valid_curves.pdf"), bbox_inches="tight")
plt.close()
print(f"Saved: {path_val}")
print("Done.")
