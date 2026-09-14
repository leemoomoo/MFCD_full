# coding: utf-8
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = r"D:\12487\NCDM PLUS2\outputs\loss_curves\loss_curves.csv"
OUT_DIR = r"D:\12487\NCDM PLUS2\outputs\loss_curves"

df = pd.read_csv(CSV_PATH)
epochs = df["epoch"]

# ===== Figure 1: Main Losses (L_Total, L_y) =====
fig1, ax1 = plt.subplots(figsize=(5.5, 3.2))

ax1.plot(epochs, df["L_total"], "k-", lw=2.5, label=r"$L_{\mathrm{Total}}$")
ax1.plot(epochs, df["L_y"], "r-", lw=1.8, label=r"$L_y$")
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Loss", fontsize=11)
ax1.legend(fontsize=9.5, loc="upper right", framealpha=0.9)
ax1.set_xlim(left=-0.3, right=epochs.max() + 0.3)
ax1.grid(alpha=0.15, linestyle="--")
ax1.tick_params(labelsize=9)

fig1.tight_layout(pad=0.8)
fig1.savefig(os.path.join(OUT_DIR, "main_losses.png"), dpi=200, bbox_inches="tight")
fig1.savefig(os.path.join(OUT_DIR, "main_losses.pdf"), bbox_inches="tight")
plt.close(fig1)
print("Saved: main_losses.png/pdf")

# ===== Figure 2: Auxiliary Losses (L_ea, L_rc with twin y-axis for L_ei) =====
fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))

ax2b = ax2.twinx()
ax2.plot(epochs, df["L_acc"], "g-", lw=1.8, label=r"$L_{\mathrm{ea}}$")
ax2.plot(epochs, df["L_sta"], "m-", lw=1.8, label=r"$L_{\mathrm{rc}}$")
ax2b.plot(epochs, df["L_opp"], "b-", lw=1.8, label=r"$L_{\mathrm{ei}}$")
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel(r"$L_{\mathrm{ea}}$ / $L_{\mathrm{rc}}$", fontsize=11)
ax2b.set_ylabel(r"$L_{\mathrm{ei}}$", fontsize=11, color="b")
ax2b.tick_params(axis="y", labelcolor="b", labelsize=9)

h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9.5, framealpha=0.9)
ax2.set_xlim(left=-0.3, right=epochs.max() + 0.3)
ax2.grid(alpha=0.15, linestyle="--")
ax2.tick_params(labelsize=9)

fig2.tight_layout(pad=0.8)
fig2.savefig(os.path.join(OUT_DIR, "aux_losses.png"), dpi=200, bbox_inches="tight")
fig2.savefig(os.path.join(OUT_DIR, "aux_losses.pdf"), bbox_inches="tight")
plt.close(fig2)
print("Saved: aux_losses.png/pdf")