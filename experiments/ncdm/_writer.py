# coding: utf-8
import sys, os, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models", "ncdm"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(ROOT, "models", "ncdm"))
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

        # Formula (17): Total loss
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

# ===== Plot =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: 5 loss curves (L_total, L_y, L_opp, L_acc, L_sta)
ax = axes[0]
ax.plot(history["epoch"], history["L_total"], "k-", lw=2.5, label="$L_{\\mathrm{Total}}$ (Eq.17)")
ax.plot(history["epoch"], history["L_y"], "r-", lw=2, label="$L_y$ (Eq.18)")
ax.plot(history["epoch"], history["L_opp"], "b--", lw=2, label="$L_{\\mathrm{ei}}$ (Eq.19)")
ax.plot(history["epoch"], history["L_acc"], "g--", lw=2, label="$L_{\\mathrm{ea}}$ (Eq.20)")
ax.plot(history["epoch"], history["L_sta"], "m--", lw=2, label="$L_{\\mathrm{rc}}$ (Eq.21)")
ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Loss", fontsize=13)
ax.set_title("Training Loss Curves (Formulas 17-21)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(alpha=0.3, linestyle="--")
ax.set_ylim(bottom=0)

# Right: Validation AUC & ACC
ax2 = axes[1]
ax3 = ax2.twinx()
ax2.plot(history["epoch"], history["valid_auc"], "b-o", lw=2.5, ms=6, label="Valid AUC")
ax3.plot(history["epoch"], history["valid_acc"], "g--s", lw=2.5, ms=6, label="Valid ACC")
ax2.set_xlabel("Epoch", fontsize=13)
ax2.set_ylabel("AUC", fontsize=13, color="b")
ax3.set_ylabel("ACC", fontsize=13, color="g")
ax2.set_title("Validation Performance", fontsize=14, fontweight="bold")
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=11, framealpha=0.9)
ax2.grid(alpha=0.3, linestyle="--")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "loss_curves.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "loss_curves.pdf"), bbox_inches="tight")
plt.close()
print(f"Saved: {os.path.join(OUT_DIR, 'loss_curves.png')}")
print("Done.")


