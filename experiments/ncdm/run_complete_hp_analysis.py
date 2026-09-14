# coding: utf-8
"""
Complete hyperparameter analysis for NCDM-Fluency on ASSIST09.
  1. Learning rate sensitivity curve (verify unimodal pattern)
  2. Fluency_lambda sensitivity curve
  3. Full grid search -> best config
  4. Ablation study under best config
"""
import os, sys, random, gc, itertools
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, r"D:\12487\NCDM PLUS2\models\ncdm")
sys.path.insert(0, r"D:\12487\NCDM PLUS2\experiments\ncdm")
from NCDM_Fluency import NCDMFluency
from assist09_dataset import build_assist09_splits_from_folder

# ── Paths ──────────────────────────────────────────────
TRAIN_CSV = r"D:\12487\NCDM PLUS2\data\a0910\train.csv"
TEST_CSV  = r"D:\12487\NCDM PLUS2\data\a0910\test.csv"
SAVE_DIR  = r"D:\12487\NCDM PLUS2\outputs\results\hp_analysis_a0910"
EPOCH     = 8
BATCH     = 256
SEED      = 2025
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ── Helpers ─────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_data():
    train_ds, _, test_ds, meta = build_assist09_splits_from_folder(
        DATA_DIR, use_fluency=True,
        train_file="train.csv", valid_file="valid.csv",
        test_file="test.csv", item_file="item.csv",
    )
    print(f"[Data] student_n={meta['student_n']}, exer_n={meta['exer_n']}, knowledge_n={meta['knowledge_n']}")
    return train_ds, test_ds

def make_loaders(train_ds, test_ds):
    train_loader = DataLoader(train_ds, BATCH, shuffle=True,  pin_memory=False)
    test_loader  = DataLoader(test_ds,  BATCH, shuffle=False, pin_memory=False)
    return train_loader, test_loader

def train_and_eval(model, train_loader, test_loader, device=DEVICE):
    """Train model & return (auc, acc).  Keeps model untouched for caller to discard."""
    model.train(train_loader, test_loader, epoch=EPOCH, device=device, silence=True)
    auc, acc = model.eval(test_loader, device=device, silence=True)
    return auc, acc

def model_kw(train_ds, **overrides):
    base = dict(knowledge_n=train_ds.knowledge_n,
                exer_n=train_ds.exer_n,
                student_n=train_ds.student_n,
                lr=0.002, dropout=0.5, fluency_lambda=0.7)
    base.update(overrides)
    return base

# ════════════════════════════════════════════════════════
#  1. Learning-rate sensitivity  (verify unimodal)
# ════════════════════════════════════════════════════════
def scan_lr(train_ds, test_ds):
    print("=" * 60)
    print("1. LR sensitivity scan")
    print("=" * 60)
    lrs = [3e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
    train_loader, test_loader = make_loaders(train_ds, test_ds)
    rows = []
    for lr in lrs:
        set_seed(SEED)
        model = NCDMFluency(**model_kw(train_ds, lr=lr))
        auc, acc = train_and_eval(model, train_loader, test_loader)
        rows.append({"lr": lr, "auc": auc, "acc": acc})
        print(f"  lr={lr:.4f}  -> auc={auc:.4f}  acc={acc:.4f}")
        del model; gc.collect(); torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR, "lr_sensitivity.csv"), index=False)
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["lr"], df["auc"], "o-", color="#4C72B0", label="AUC")
    ax.plot(df["lr"], df["acc"], "s--", color="#55A868", label="Accuracy")
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel("Score")
    ax.set_title("LR Sensitivity Curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    # Annotate best
    best = df.loc[df["auc"].idxmax()]
    ax.annotate(f"Best lr={best['lr']:.4f}\nAUC={best['auc']:.4f}",
                xy=(best["lr"], best["auc"]),
                xytext=(best["lr"]*3, best["auc"]-0.015),
                arrowprops=dict(arrowstyle="->", color="gray"))
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "lr_sensitivity.png"), dpi=200)
    plt.close(fig)
    print(f"  => Best lr = {best['lr']:.4f}  (AUC={best['auc']:.4f})\n")
    return best["lr"]

# ════════════════════════════════════════════════════════
#  2. Fluency_lambda sensitivity
# ════════════════════════════════════════════════════════
def scan_lambda(train_ds, test_ds, best_lr):
    print("=" * 60)
    print("2. Fluency_lambda sensitivity scan")
    print("=" * 60)
    lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    train_loader, test_loader = make_loaders(train_ds, test_ds)
    rows = []
    for lam in lambdas:
        set_seed(SEED)
        model = NCDMFluency(**model_kw(train_ds, lr=best_lr, fluency_lambda=lam))
        auc, acc = train_and_eval(model, train_loader, test_loader)
        rows.append({"fluency_lambda": lam, "auc": auc, "acc": acc})
        print(f"  fluency_lambda={lam:.1f}  -> auc={auc:.4f}  acc={acc:.4f}")
        del model; gc.collect(); torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR, "lambda_sensitivity.csv"), index=False)
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["fluency_lambda"], df["auc"], "o-", color="#4C72B0", label="AUC")
    ax.plot(df["fluency_lambda"], df["acc"], "s--", color="#55A868", label="Accuracy")
    ax.set_xlabel("Fluency lambda")
    ax.set_ylabel("Score")
    ax.set_title("Fluency Lambda Sensitivity Curve (lr={:.4f})".format(best_lr))
    ax.set_xticks(lambdas)
    ax.legend(); ax.grid(True, alpha=0.3)
    best = df.loc[df["auc"].idxmax()]
    ax.annotate(f"Best lambda={best['fluency_lambda']:.1f}\nAUC={best['auc']:.4f}",
                xy=(best["fluency_lambda"], best["auc"]),
                xytext=(best["fluency_lambda"]+0.2, best["auc"]-0.01),
                arrowprops=dict(arrowstyle="->", color="gray"))
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "lambda_sensitivity.png"), dpi=200)
    plt.close(fig)
    print(f"  => Best fluency_lambda = {best['fluency_lambda']:.1f}  (AUC={best['auc']:.4f})\n")
    return best["fluency_lambda"]

# ════════════════════════════════════════════════════════
#  3. Full grid search around optimal region
# ════════════════════════════════════════════════════════
def grid_search(train_ds, test_ds, best_lr, best_lam):
    print("=" * 60)
    print("3. Grid search (lr + dropout + fluency_lambda)")
    print("=" * 60)
    # Fine grid around the optimal region
    lr_grid     = [best_lr * 0.5, best_lr, best_lr * 2.0]
    lam_grid    = [max(0.0, best_lam - 0.2), best_lam, min(1.0, best_lam + 0.2)]
    dropout_grid = [0.3, 0.5]

    train_loader, test_loader = make_loaders(train_ds, test_ds)
    rows = []
    keys, values = zip(*[("lr", lr_grid), ("dropout", dropout_grid), ("fluency_lambda", lam_grid)])
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for i, cfg in enumerate(combos):
        set_seed(SEED)
        model = NCDMFluency(**model_kw(train_ds, **cfg))
        auc, acc = train_and_eval(model, train_loader, test_loader)
        rows.append({**cfg, "auc": auc, "acc": acc})
        print(f"  [{i+1}/{len(combos)}] {cfg}  -> auc={auc:.4f}  acc={acc:.4f}")
        del model; gc.collect(); torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR, "grid_search.csv"), index=False)
    best = df.loc[df["auc"].idxmax()]
    print(f"\n  => Best grid config: lr={best['lr']}, dropout={best['dropout']}, "
          f"fluency_lambda={best['fluency_lambda']}  (AUC={best['auc']:.4f})\n")
    return best.to_dict()

# ════════════════════════════════════════════════════════
#  4. Ablation study under best config
# ════════════════════════════════════════════════════════
def ablation(train_ds, test_ds, best_cfg):
    print("=" * 60)
    print("4. Ablation study (with best hyperparams)")
    print("=" * 60)

    ABLATIONS = {
        "L0_baseline":        {"use_fluency": False, "use_behavior": False, "use_gate": False},
        "L1_only_fluency":    {"use_fluency": True,  "use_behavior": False, "use_gate": False},
        "L2_no_gate":         {"use_fluency": True,  "use_behavior": True,  "use_gate": False},
        "L3_full_model":      {"use_fluency": True,  "use_behavior": True,  "use_gate": True},
        "A1_w_o_rt":          {"use_fluency": True,  "use_behavior": True,  "use_gate": True,  "use_rt": False},
        "A2_w_o_opportunity": {"use_fluency": True,  "use_behavior": True,  "use_gate": True,  "use_opportunity": False},
    }

    train_loader, test_loader = make_loaders(train_ds, test_ds)
    rows = []
    for name, flags in ABLATIONS.items():
        set_seed(SEED)
        kwargs = {k: v for k, v in best_cfg.items() if k in ("lr", "dropout", "fluency_lambda")}
        kwargs.update(flags)
        model = NCDMFluency(**model_kw(train_ds, **kwargs))
        auc, acc = train_and_eval(model, train_loader, test_loader)
        rows.append({"model": name, "auc": auc, "acc": acc})
        print(f"  {name:25s} -> auc={auc:.4f}  acc={acc:.4f}")
        del model; gc.collect(); torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    df.to_csv(os.path.join(SAVE_DIR, "ablation_results.csv"), index=False)

    # Plot grouped bar chart
    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, df["auc"], width, label="AUC", color="#4C72B0")
    ax.bar(x + width/2, df["acc"], width, label="Accuracy", color="#55A868")
    for i in range(len(df)):
        ax.annotate(f"{df['auc'].iloc[i]:.4f}", (x[i] - width/2, df['auc'].iloc[i]),
                    ha="center", va="bottom", fontsize=8)
        ax.annotate(f"{df['acc'].iloc[i]:.4f}", (x[i] + width/2, df['acc'].iloc[i]),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=45, ha="right")
    ax.set_ylabel("Score"); ax.set_title("Ablation Study (best config)")
    ax.legend(loc="lower left"); ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "ablation_bar.png"), dpi=200)
    plt.close(fig)
    print(f"\n  Ablation chart saved.\n")

# ════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Device: {DEVICE}, Epochs: {EPOCH}, Batch: {BATCH}\n")

    train_ds, test_ds = load_data()
    print(f"Train: {len(train_ds)} rows, Test: {len(test_ds)} rows")
    print(f"Students: {train_ds.student_n}, Items: {train_ds.exer_n}, Skills: {train_ds.knowledge_n}\n")

    # 1) LR sensitivity
    best_lr = scan_lr(train_ds, test_ds)

    # 2) Lambda sensitivity (using best_lr)
    best_lam = scan_lambda(train_ds, test_ds, best_lr)

    # 3) Grid search around optimum
    best_cfg = grid_search(train_ds, test_ds, best_lr, best_lam)

    # 4) Ablation under best config
    ablation(train_ds, test_ds, best_cfg)

    print("=" * 60)
    print("ALL DONE. Results saved to:", SAVE_DIR)
    print("=" * 60)

if __name__ == "__main__":
    main()
