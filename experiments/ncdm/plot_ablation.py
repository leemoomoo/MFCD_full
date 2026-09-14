# coding: utf-8
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = r"D:\12487\NCDM PLUS2\outputs\results\hp_analysis_a0910"

# ====================================================
# Helper: set consistent style
# ====================================================
C_BLUE = "#2E86AB"
C_GREEN = "#A23B72"
C_RED = "#F18F01"

def style_ax(ax):
    ax.grid(alpha=0.15, linestyle="--", zorder=0)
    ax.tick_params(labelsize=9)

# ====================================================
# Figure 2: fluency_lambda sensitivity
df = pd.read_csv(os.path.join(DATA_DIR, "02_lambda_sensitivity.csv"))
df = df[df["fluency_lambda"] < 0.95]

fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(df["fluency_lambda"], df["auc"], "-o", c="#2E86AB", lw=2.5, ms=7, zorder=4, label="AUC")
ax1.plot(df["fluency_lambda"], df["acc"], "--s", c="#A23B72", lw=2.5, ms=7, zorder=4, label="ACC")
ax1.plot(df["fluency_lambda"], df["f1"], ":d", c="#F18F01", lw=2.5, ms=7, zorder=4, label="F1")
ax1.set_xticks([0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
ax1.set_xlabel("λ", fontsize=12)
ax1.set_ylim(0.70, 0.80)
ax1.set_ylabel("AUC / ACC / F1", fontsize=12)
ax1.tick_params(labelsize=10)
ax1.grid(alpha=0.15, linestyle="--", zorder=0)

ax2 = ax1.twinx()
ax2.plot(df["fluency_lambda"], df["rmse"], "-.^", c="#446455", lw=2.5, ms=7, zorder=4, label="RMSE")
ax2.set_ylabel("RMSE", color="#446455", fontsize=12)
ax2.tick_params(axis="y", labelcolor="#446455", labelsize=10)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(1.15, 0), framealpha=0.9, fontsize=10)
fig.tight_layout(rect=[0, 0, 0.85, 1])
fig.savefig(os.path.join(DATA_DIR, "fig2_lambda.png"), dpi=200, bbox_inches="tight")
plt.close()
print("fig2_lambda.png saved")
# Figure 3: latent_dim sensitivity
df = pd.read_csv(os.path.join(DATA_DIR, "03_latent_dim_sensitivity.csv"))

fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(df["flu_latent_dim"], df["auc"], "-o", c="#2E86AB", lw=2.5, ms=7, zorder=4, label="AUC")
ax1.plot(df["flu_latent_dim"], df["acc"], "--s", c="#A23B72", lw=2.5, ms=7, zorder=4, label="ACC")
ax1.plot(df["flu_latent_dim"], df["f1"], ":d", c="#F18F01", lw=2.5, ms=7, zorder=4, label="F1")
ax1.set_xticks([4, 8, 16, 32, 64])
ax1.set_xlabel("D", fontsize=12)
ax1.set_ylim(0.70, 0.82)
ax1.set_ylabel("AUC / ACC / F1", fontsize=12)
ax1.tick_params(labelsize=10)
ax1.grid(alpha=0.15, linestyle="--", zorder=0)

ax2 = ax1.twinx()
ax2.plot(df["flu_latent_dim"], df["rmse"], "-.^", c="#446455", lw=2.5, ms=7, zorder=4, label="RMSE")
ax2.set_ylabel("RMSE", color="#446455", fontsize=12)
ax2.tick_params(axis="y", labelcolor="#446455", labelsize=10)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(1.15, 0), framealpha=0.9, fontsize=10)
fig.tight_layout(rect=[0, 0, 0.85, 1])
fig.savefig(os.path.join(DATA_DIR, "fig3_latent_dim.png"), dpi=200, bbox_inches="tight")
plt.close()
print("fig3_latent_dim.png saved")
# Figure 6: Ablation study - grouped bar chart
# ====================================================
df = pd.read_csv(os.path.join(DATA_DIR, "06_ablation.csv"))
model_short = ["Base", "L1", "L2", "Ours"]
metrics = [("auc", "AUC"), ("acc", "ACC"), ("f1", "F1"), ("rmse", "RMSE")]

fig, ax1 = plt.subplots(figsize=(11, 2.3))
x = np.arange(len(model_short))
width = 0.18

for mi, (col, mname) in enumerate([("auc", "AUC"), ("acc", "ACC"), ("f1", "F1")]):
    offset = (mi - 1.5) * width
    vals = df[col].values
    ax1.bar(x + offset, vals, width, label=mname, color=["#2E86AB", "#A23B72", "#F18F01"][mi], edgecolor="white", linewidth=0.5, zorder=3)

ax1.set_xticks(x)
ax1.set_xticklabels(model_short, fontsize=13)
ax1.set_ylabel("AUC / ACC / F1", fontsize=14)
ax1.tick_params(labelsize=12)
ax1.set_ylim(0.70, 0.84)
ax1.grid(axis="y", alpha=0.2, linestyle="--", zorder=0)

ax2 = ax1.twinx()
vals_rmse = df["rmse"].values
ax2.bar(x + 0.27, vals_rmse, width, label="RMSE", color="#446455", edgecolor="white", linewidth=0.5, zorder=3)
ax2.set_ylabel("RMSE", color="#446455", fontsize=14)
ax2.tick_params(axis="y", labelcolor="#446455", labelsize=12)
ax2.set_ylim(0.40, 0.46)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=12, loc="lower left", bbox_to_anchor=(1.15, 0), framealpha=0.9)
fig.tight_layout(pad=0.5, rect=[0, 0, 0.88, 1])
fig.savefig(os.path.join(DATA_DIR, "fig6_ablation.png"), dpi=200, bbox_inches="tight")
plt.close()
print("fig6_ablation.png saved")
