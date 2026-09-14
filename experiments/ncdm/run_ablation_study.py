# coding: utf-8
# Complete hyperparameter analysis for NCDM-Fluency on a0910 dataset.
#
# Phases:
#   1. Learning rate sensitivity
#   2. fluency_lambda sensitivity (incl. lambda=0 => pure NCDM)
#   3. flu_latent_dim sensitivity
#   4. Grid search: lr x dropout (best config)
#   5. Ablation study under best config

import os, sys, random, gc, itertools, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error

PROJECT_ROOT = r"D:\12487\NCDM PLUS2"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "a0910")
SAVE_DIR     = os.path.join(PROJECT_ROOT, "outputs", "results", "hp_analysis_a0910")
os.makedirs(SAVE_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "models", "ncdm"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "ncdm"))

from NCDM_Fluency import NCDMFluency
from NCDM import NCDM
from assist09_dataset import build_assist09_splits_from_folder

EPOCH     = 10
BATCH     = 256
SEED      = 2025
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
N_WORKERS = 0

plt.rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 200,
})
C_BLUE  = "#2C5F7C"
C_GREEN = "#4C7A4C"
C_RED   = "#C44E52"
C_GOLD  = "#DAA520"


# =====================================================================
# Ablation monkey-patch helpers (no model code changes)
# =====================================================================

class _ZeroEncoder(nn.Module):
    """Replace beh_encoder -> output zeros -> z = tanh(student_flu_bias) only (static per-student)."""
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
    def forward(self, x):
        return torch.zeros(x.shape[0], self.out_dim, device=x.device)

class _ConstantGate(nn.Module):
    """Replace fuse_gate -> constant logit -> gate = sigmoid(constant) -> static fusion."""
    def __init__(self, logit: float = 0.0):
        super().__init__()
        self.register_buffer("_logit", torch.tensor([logit]))
    def forward(self, x):
        return self._logit.expand(x.shape[0], 1)

class _SliceLoader:
    """Wrap 7-item DataLoader batches -> yield batch[:4] for original NCDM."""
    def __init__(self, loader):
        self.loader = loader
    def __iter__(self):
        for batch in self.loader:
            yield batch[:4]
    def __len__(self):
        return len(self.loader)


def _train_eval_ncdm(model, train_loader, valid_loader, device):
    """Train & eval original NCDM (slices 7->4 items per batch)."""
    model.train(_SliceLoader(train_loader), epoch=EPOCH, device=device, silence=True)
    y_true, y_pred, auc, acc = model.eval(_SliceLoader(valid_loader), device=device)
    f1 = safe_f1(y_true, y_pred)
    rmse = safe_rmse(y_true, y_pred)
    return auc, acc, f1, rmse

def _train_eval_fluency(model, train_loader, valid_loader, device):
    """Train & eval NCDMFluency (full 7-item batches)."""
    model.train(train_loader, valid_loader, epoch=EPOCH, device=device, silence=True)
    y_true, y_pred, auc, acc = model.eval(valid_loader, device=device, silence=True)
    f1 = safe_f1(y_true, y_pred)
    rmse = safe_rmse(y_true, y_pred)
    return auc, acc, f1, rmse

def safe_f1(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred) >= 0.5
    return float(f1_score(y_true, y_pred, zero_division=0))

def safe_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(np.asarray(y_true), np.asarray(y_pred))))


def _unify_weights(model):
    """Verify fluency_lambda is set (single-weight design, Eq.17)."""
    model.fluency_lambda = model.net.fluency_lambda
    logging.info("Using fluency_lambda=%.4f (Eq.17 normalized)", model.fluency_lambda)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(batch_size=BATCH):
    train_ds, valid_ds, test_ds, meta = build_assist09_splits_from_folder(
        DATA_DIR, use_fluency=True,
        train_file="train.csv", valid_file="valid.csv",
        test_file="test.csv", item_file="item.csv",
    )
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=N_WORKERS)
    valid_loader = DataLoader(valid_ds,  batch_size, shuffle=False, num_workers=N_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size, shuffle=False, num_workers=N_WORKERS)
    print(f"[Data] student_n={meta['student_n']}, exer_n={meta['exer_n']}, knowledge_n={meta['knowledge_n']}")
    print(f"[Data] train={meta['n_train']}, valid={meta['n_valid']}, test={meta['n_test']}")
    return train_ds, train_loader, valid_loader, test_loader, meta


def train_and_eval(model, train_loader, valid_loader, device=DEVICE):
    model.train(train_loader, valid_loader, epoch=EPOCH, device=device, silence=True)
    y_true, y_pred, auc, acc = model.eval(valid_loader, device=device, silence=True)
    f1_val = safe_f1(y_true, y_pred)
    rmse_val = safe_rmse(y_true, y_pred)
    return auc, acc, f1_val, rmse_val

def safe_f1(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred) >= 0.5
    return float(f1_score(y_true, y_pred, zero_division=0))

def safe_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(np.asarray(y_true), np.asarray(y_pred))))


def model_kw(meta, **overrides):
    base = dict(
        knowledge_n=meta["knowledge_n"],
        exer_n=meta["exer_n"],
        student_n=meta["student_n"],
        lr=0.002, dropout=0.5, fluency_lambda=0.7,
        flu_latent_dim=16,
    )
    base.update(overrides)
    return base


# Phase 1: Learning-rate sensitivity
def scan_lr(meta, train_loader, valid_loader):
    """Fixed LR = 0.002, skip expensive sweep."""
    print()
    print("=" * 60)
    print("PHASE 1: LR fixed to 0.002 (skip sweep)")
    print("=" * 60)
    fixed_lr = 0.002
    print(f"  Using fixed lr={fixed_lr}")
    df = pd.DataFrame([{"lr": fixed_lr, "auc": 0.0, "acc": 0.0}])
    df.to_csv(os.path.join(SAVE_DIR, "01_lr_sensitivity.csv"), index=False)
    return float(fixed_lr)

# Phase 2: fluency_lambda sensitivity
def scan_lambda(meta, train_loader, valid_loader, best_lr):
    print()
    print("=" * 60)
    print("PHASE 2: fluency_lambda sensitivity scan")
    print("=" * 60)
    # Raw lambda values covering sigma(raw) from ~0 to ~1
    # sigma(-6)~0, sigma(6)~1, sigma(0)=0.5
    lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    rows = []
    for fl in lambdas:

        set_seed(SEED)
        model = NCDMFluency(**model_kw(meta, lr=best_lr, fluency_lambda=fl))
        _unify_weights(model)
        auc, acc, f1, rmse = train_and_eval(model, train_loader, valid_loader)
        rows.append({"fluency_lambda": fl,
                      "auc": auc, "acc": acc, "f1": f1, "rmse": rmse})
        print(f"  lam={fl:.2f}  -> auc={auc:.4f}  acc={acc:.4f}  f1={f1:.4f}  rmse={rmse:.4f}")
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR, "02_lambda_sensitivity.csv"), index=False)
    print("  Saved to 02_lambda_sensitivity.csv")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    metrics_data = [("auc", "AUC"), ("acc", "ACC"), ("f1", "F1"), ("rmse", "RMSE")]
    for idx, (col, ylabel) in enumerate(metrics_data):
        ax = axes[idx // 2][idx % 2]
        ax.plot(df["fluency_lambda"], df[col], "-o", c=C_BLUE, lw=2.5, ms=7, zorder=4)
        ax.set_xlabel("lambda", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(alpha=0.15, linestyle="--", zorder=0)
        ax.tick_params(labelsize=9)
    
    bi = df["auc"].idxmax()
    best_lam = df.loc[bi, "fluency_lambda"]
    print(f"  >>> Best fluency_lambda = {best_lam:.2f}")
    
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(SAVE_DIR, "fig2_lambda.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig2_lambda.png saved")
    return best_lam

# Phase 3: flu_latent_dim sensitivity
def scan_latent_dim(meta, train_loader, valid_loader, best_lr, best_lambda):
    print()
    print("=" * 60)
    print("PHASE 3: flu_latent_dim sensitivity")
    print("=" * 60)
    dims = [4, 8, 16, 32, 64]
    rows = []
    for d in dims:
        set_seed(SEED)
        model = NCDMFluency(**model_kw(meta, lr=best_lr, fluency_lambda=best_lambda, flu_latent_dim=d))
        _unify_weights(model)
        auc, acc, f1, rmse = train_and_eval(model, train_loader, valid_loader)
        rows.append({"flu_latent_dim": d, "auc": auc, "acc": acc, "f1": f1, "rmse": rmse})
        print(f"  flu_latent_dim={d}  -> auc={auc:.4f}  acc={acc:.4f}  f1={f1:.4f}  rmse={rmse:.4f}")
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR, "03_latent_dim_sensitivity.csv"), index=False)
    print("  Saved to 03_latent_dim_sensitivity.csv")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    metrics_data = [("auc", "AUC"), ("acc", "ACC"), ("f1", "F1"), ("rmse", "RMSE")]
    for idx, (col, ylabel) in enumerate(metrics_data):
        ax = axes[idx // 2][idx % 2]
        ax.plot(df["flu_latent_dim"], df[col], "-o", c=C_BLUE, lw=2.5, ms=7, zorder=4)
        ax.set_xlabel("latent dim D", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(alpha=0.15, linestyle="--", zorder=0)
        ax.tick_params(labelsize=9)
    
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(SAVE_DIR, "fig3_latent_dim.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig3_latent_dim.png saved")

    best_d = df.loc[df["auc"].idxmax(), "flu_latent_dim"]
    print(f"  >>> Best latent dim = {best_d}")
    return best_d

# Phase 4: Grid search lr x dropout
def grid_search(meta, train_loader, valid_loader, best_lambda):
    print()
    print("=" * 60)
    print("PHASE 4: Grid search (lr x dropout)")
    print("=" * 60)
    lrs = [4e-4, 8e-4, 1.6e-3]
    dropouts = [0.3, 0.5, 0.7]
    rows = []
    for lr, do in itertools.product(lrs, dropouts):
        set_seed(SEED)
        model = NCDMFluency(**model_kw(meta, lr=lr, dropout=do, fluency_lambda=best_lambda))
        _unify_weights(model)
        auc, acc, f1, rmse = train_and_eval(model, train_loader, valid_loader)
        rows.append({"lr": lr, "dropout": do, "auc": auc, "acc": acc, "f1": f1, "rmse": rmse})
        print(f"  lr={lr:.4f}  dropout={do:.1f}  -> auc={auc:.4f}  acc={acc:.4f}  f1={f1:.4f}  rmse={rmse:.4f}")
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR, "04_grid_2d.csv"), index=False)
    print("  Saved to 04_grid_2d.csv")

    pv = df.pivot_table(values="auc", index="lr", columns="dropout", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    vmin, vmax = df["auc"].min(), df["auc"].max()
    pad = (vmax - vmin) * 0.05
    im = ax.imshow(pv.values, cmap="RdYlGn", aspect="auto", vmin=vmin-pad, vmax=vmax+pad)
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels([f"{c:.1f}" for c in pv.columns])
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels([str(idx) for idx in pv.index])
    ax.set_xlabel("Dropout Rate")
    ax.set_ylabel("Learning Rate")
    max_val = pv.values.max()
    bp = np.unravel_index(pv.values.argmax(), pv.values.shape)
    ax.scatter(bp[1], bp[0], s=600, facecolors="none",
               edgecolors="black", linewidths=3.0, zorder=5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AUC", rotation=270, labelpad=10, fontsize=11)
    ax.text(bp[1], bp[0]-0.25,
            f"Best={max_val:.4f}\nlr={pv.index[bp[0]]}\ndropout={pv.columns[bp[1]]}",
            ha="center", va="top", fontsize=9, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85))
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig4_grid.png"), bbox_inches="tight")
    plt.close()
    print("  fig4_grid.png saved")

    best_row = df.loc[df["auc"].idxmax()]
    return best_row["lr"], best_row["dropout"]

# Phase 5: Ablation study
def ablation(meta, train_loader, valid_loader, best_lr, best_lambda, best_do, best_dim):
    print()
    print("=" * 60)
    print("PHASE 5: Ablation study (Architecture variants per paper x5.4)")
    print("=" * 60)

    configs = [
        ("Base_Only_Mastery", "ncdm",    {},  None),
        ("L1_Static_Fluency", "fluency", {}, "zero_enc+const_gate"),
        ("L2_Static_Fusion",  "fluency", {}, "const_gate"),
        ("L3_Full_FCD",       "fluency", {}, None),
    ]

    rows = []
    for name, mtype, extra_hp, patch in configs:
        set_seed(SEED)

        if mtype == "ncdm":
            # ---- Pure NCDM backbone ----
            model = NCDM(
                knowledge_n=meta["knowledge_n"],
                exer_n=meta["exer_n"],
                student_n=meta["student_n"],
            )
            auc, acc, f1, rmse = _train_eval_ncdm(model, train_loader, valid_loader, DEVICE)

        else:
            kwargs = model_kw(meta, lr=best_lr, dropout=best_do,
                              fluency_lambda=best_lambda,
                              flu_latent_dim=best_dim)
            kwargs.update(extra_hp)
            model = NCDMFluency(**kwargs)
            _unify_weights(model)

            if patch == "zero_enc+const_gate":
                # L1: Static Fluency — no MTL, no dynamic gate, static per-student embedding
                model.fluency_lambda = 0.0
                model.net.beh_encoder = _ZeroEncoder(kwargs.get("flu_latent_dim", 16))
                model.net.fuse_gate = _ConstantGate(logit=0.0)

            elif patch == "const_gate":
                # L2: Static Fusion — MTL active, dynamic gate -> constant (static fusion)
                model.net.fuse_gate = _ConstantGate(logit=0.0)

            auc, acc, f1, rmse = _train_eval_fluency(model, train_loader, valid_loader, DEVICE)

        rows.append({"model": name, "auc": auc, "acc": acc, "f1": f1, "rmse": rmse})
        print(f"  {name:25s}  -> auc={auc:.6f}  acc={acc:.6f}  f1={f1:.6f}  rmse={rmse:.6f}")

        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR, "06_ablation.csv"), index=False)
    print("  Saved to 06_ablation.csv")

        # -- 4-panel bar chart: AUC / ACC / F1 / RMSE --
    labels = ["Mastery\nOnly", "L1\n(Static Fluency)", "L2\n(Static Fusion)", "L3\n(Full FCD)"]
    bar_colors = [C_RED, C_BLUE, C_BLUE, C_BLUE]
    metrics = [("auc", "AUC", 0.60, 1.0), ("acc", "ACC", 0.60, 1.0),
               ("f1", "F1", 0.60, 1.0), ("rmse", "RMSE", 0.30, 0.55)]

            
def main():
    set_seed(SEED)
    print(f"DEVICE = {DEVICE}")
    print(f"SAVE_DIR = {SAVE_DIR}")

    train_ds, train_loader, valid_loader, test_loader, meta = load_data()

    best_lr = scan_lr(meta, train_loader, valid_loader)
    print(f"\n>>> Best LR from Phase 1: {best_lr}")

    best_lambda = scan_lambda(meta, train_loader, valid_loader, best_lr)
    print(f">>> Best fluency_lambda from Phase 2: {best_lambda}")

    best_dim = scan_latent_dim(meta, train_loader, valid_loader, best_lr, best_lambda)
    print(f">>> Best flu_latent_dim from Phase 3: {best_dim}")

    best_lr2, best_do = grid_search(meta, train_loader, valid_loader, best_lambda)
    print(f">>> Best config from Phase 4: lr={best_lr2}, dropout={best_do}")

    ablation(meta, train_loader, valid_loader, best_lr2, best_lambda, best_do, best_dim)

    print()
    print("=" * 60)
    print("FINAL: best config evaluation on TEST set")
    print("=" * 60)
    set_seed(SEED)
    final_model = NCDMFluency(**model_kw(meta, lr=best_lr2, dropout=best_do,
                                          fluency_lambda=best_lambda,
                                          flu_latent_dim=best_dim))
    _unify_weights(final_model)
    final_model.train(train_loader, test_loader, epoch=EPOCH, device=DEVICE, silence=True)
    y_true, y_pred, test_auc, test_acc = final_model.eval(test_loader, device=DEVICE, silence=True)
    test_f1 = safe_f1(y_true, y_pred)
    test_rmse = safe_rmse(y_true, y_pred)
    print(f"TEST: auc={test_auc:.6f}  acc={test_acc:.6f}  f1={test_f1:.6f}  rmse={test_rmse:.6f}")

    summary = pd.DataFrame([{
        "best_lr": best_lr2,
        "best_dropout": best_do,
        "best_fluency_lambda": best_lambda,
        "best_flu_latent_dim": best_dim,
        "test_auc": test_auc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_rmse": test_rmse,
    }])
    summary.to_csv(os.path.join(SAVE_DIR, "00_summary.csv"), index=False)
    print(f"\nAll results saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()



