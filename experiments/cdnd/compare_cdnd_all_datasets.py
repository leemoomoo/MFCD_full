# coding: utf-8
"""
compare_cdnd_all_datasets.py
对比 CDND baseline (ACD) vs CDND+Fluency (ACDWithFluency) on 6 datasets
"""
import os, sys, random, logging, gc, argparse
import numpy as np
import pandas as pd
import torch
import shutil
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# ---------- path setup ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "cdnd"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "ncdm"))

from model import cdnd
from model import cdnd_fluency

from experiments.ncdm.math1_dataset import build_math1_splits_from_folder
from experiments.ncdm.math2_dataset import build_math2_splits_from_folder
from experiments.ncdm.frcsub_dataset import build_frcsub_splits_from_folder
from experiments.ncdm.Assist17_dataset import build_assist17_splits_from_folder
from experiments.ncdm.assist09_dataset import build_assist09_splits_from_folder
from experiments.ncdm.junyi_dataset import build_junyi_splits_from_csv

# ---------- config ----------
SEED = 2025
EPOCHS = 10
LR = 0.002
BATCH = 256
FLU_LATENT_DIM = 16
AUX_LOSS_WEIGHT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
SAVE_ROOT = os.path.join(PROJECT_ROOT, "experiments", "cdnd", "outputs", "results")
os.makedirs(SAVE_ROOT, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class TrimLoader:
    """Reusable wrapper: yields only first 4 elements per batch (baseline models expect 4-tuples)."""
    def __init__(self, loader):
        self.loader = loader
    def __iter__(self):
        for batch in self.loader:
            yield batch[:4]
    def __len__(self):
        return len(self.loader)


def trim_loader(loader):
    """Wrap a DataLoader to yield only first 4 elements (baseline models expect 4-tuples)."""
    return TrimLoader(loader)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ====== Dataset loading (matching IRT comparison) ======
def get_dataset(dataset_name, use_fluency=True, alpha=1.0, beta=1.0):
    name = dataset_name.lower()
    if name == "math1":
        return build_math1_splits_from_folder(
            os.path.join(DATA_ROOT, "math2015", "Math1"), seed=SEED,
            use_fluency=use_fluency, alpha=alpha, beta=beta)
    if name == "math2":
        return build_math2_splits_from_folder(
            os.path.join(DATA_ROOT, "math2015", "Math2"), seed=SEED,
            use_fluency=use_fluency, alpha=alpha, beta=beta)
    if name == "frcsub":
        return build_frcsub_splits_from_folder(
            os.path.join(DATA_ROOT, "math2015", "FrcSub"), seed=SEED,
            use_fluency=use_fluency, alpha=alpha, beta=beta)
    if name == "assist17":
        return build_assist17_splits_from_folder(
            os.path.join(DATA_ROOT, "Assist17"),
            use_fluency=use_fluency, alpha=alpha, beta=beta)
    if name == "assist09":
        assist09_dir = os.path.join(DATA_ROOT, "assist09")
        if os.path.isdir(assist09_dir) and os.path.isfile(os.path.join(assist09_dir, "train.csv")):
            data_dir = assist09_dir
        else:
            a0910_dir = os.path.join(DATA_ROOT, "a0910")
            if os.path.isdir(a0910_dir):
                data_dir = a0910_dir
            else:
                raise FileNotFoundError("assist09/a0910 dataset not found")
        return build_assist09_splits_from_folder(
            data_dir, use_fluency=use_fluency, alpha=alpha, beta=beta)
    if name == "junyi":
        data_dir = os.path.join(DATA_ROOT, "junyi")
        return build_junyi_splits_from_csv(
            pd.read_csv(os.path.join(data_dir, "train.csv")),
            pd.read_csv(os.path.join(data_dir, "valid.csv")),
            pd.read_csv(os.path.join(data_dir, "test.csv")),
            use_fluency=use_fluency, alpha=alpha, beta=beta)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_loaders(train_ds, valid_ds, test_ds, batch_size=BATCH):
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


def print_meta(dataset_name, model_name, meta):
    print(f"  [{dataset_name}] {model_name}: "
          f"stu={meta['student_n']} exer={meta['exer_n']} "
          f"kc={meta['knowledge_n']} train={meta['n_train']} "
          f"valid={meta['n_valid']} test={meta['n_test']}")


# ====== Train & eval ======
def train_baseline(train_loader, valid_loader, meta, save_dir):
    set_seed(SEED)
    model = cdnd.ACD(meta["student_n"], meta["exer_n"], meta["knowledge_n"], meta["knowledge_n"])
    model.train(train_data=trim_loader(train_loader), test_data=trim_loader(valid_loader),
                epoch=EPOCHS, device=DEVICE, lr=LR)
    # Load best checkpoint
    src = os.path.join("params", "arcd.params")
    if os.path.exists(src):
        dst = os.path.join(save_dir, "arcd_best.params")
        shutil.copyfile(src, dst)
        model.load(dst)
    return model


def train_fluency(train_loader, valid_loader, meta, save_dir):
    set_seed(SEED)
    model = cdnd_fluency.ACDWithFluency(
        meta["student_n"], meta["exer_n"], meta["knowledge_n"], meta["knowledge_n"],
        flu_latent_dim=FLU_LATENT_DIM, aux_loss_weight=AUX_LOSS_WEIGHT,
    )
    model.train(train_data=train_loader, test_data=valid_loader,
                epoch=EPOCHS, device=DEVICE, lr=LR)
    src = os.path.join("params", "arcd_fluency.params")
    if os.path.exists(src):
        dst = os.path.join(save_dir, "arcd_fluency_best.params")
        shutil.copyfile(src, dst)
        model.load(dst)
    return model


def evaluate(model, test_loader):
    if isinstance(model, cdnd.ACD):
        auc, acc, rmse, f1 = model.eval(trim_loader(test_loader), device=DEVICE)
    else:
        auc, acc, rmse, f1 = model.eval(test_loader, device=DEVICE)
    return {"auc": auc, "acc": acc, "f1": f1, "rmse": rmse}


# ====== Main ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="all",
                        help="comma-separated or 'all'")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    dataset_list = (
        [x.strip().lower() for x in args.datasets.split(",")]
        if args.datasets != "all"
        else ["math1", "math2", "frcsub", "assist17", "assist09", "junyi"]
    )

    print(f"DEVICE = {DEVICE}, EPOCHS = {args.epochs}")
    print(f"Datasets: {dataset_list}\n")

    all_results = []
    os.makedirs("params", exist_ok=True)

    for dataset_name in dataset_list:
        try:
            train_ds, valid_ds, test_ds, meta = get_dataset(dataset_name, use_fluency=True)
        except Exception as e:
            print(f"\n[SKIP] {dataset_name}: dataset load failed - {e}")
            continue

        train_loader, valid_loader, test_loader = build_loaders(train_ds, valid_ds, test_ds)

        # ---- baseline ----
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")
        save_dir = os.path.join(SAVE_ROOT, dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        print_meta(dataset_name, "CDND_baseline", meta)
        m1 = train_baseline(train_loader, valid_loader, meta, save_dir)
        r1 = evaluate(m1, test_loader)
        all_results.append({"dataset": dataset_name, "model": "CDND_baseline", **r1})
        print(f"  -> AUC={r1['auc']:.4f} ACC={r1['acc']:.4f} F1={r1['f1']:.4f}")
        del m1; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- fluency ----
        print_meta(dataset_name, "CDND_fluency", meta)
        m2 = train_fluency(train_loader, valid_loader, meta, save_dir)
        r2 = evaluate(m2, test_loader)
        all_results.append({"dataset": dataset_name, "model": "CDND_fluency", **r2})
        print(f"  -> AUC={r2['auc']:.4f} ACC={r2['acc']:.4f} F1={r2['f1']:.4f}")
        del m2; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ====== Build comparison table ======
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    grouped = {}
    for r in all_results:
        grouped.setdefault(r["dataset"], {})[r["model"]] = r

    header = f"{'Dataset':<12} {'Metric':<8} {'CDND_baseline':<14} {'CDND_fluency':<14} {'Delta':<10}"
    print(header)
    print("-" * len(header))

    comparison_rows = []
    for dataset in sorted(grouped.keys()):
        dd = grouped[dataset]
        base = dd.get("CDND_baseline", {})
        flu = dd.get("CDND_fluency", {})
        for metric in ["auc", "acc", "f1"]:
            bv = base.get(metric, 0)
            fv = flu.get(metric, 0)
            delta = fv - bv
            print(f"{dataset:<12} {metric:<8} {bv:<14.4f} {fv:<14.4f} {delta:<+10.4f}")
        print()

    # Save full CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(SAVE_ROOT, "compare_cdnd_all_datasets.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nFull results saved to: {csv_path}")

    # Save comparison table CSV
    pivot = df.pivot_table(index="dataset", columns="model",
                           values=["auc", "acc", "f1"]).round(6)
    pivot.columns = [f"{c[1]}_{c[0]}" for c in pivot.columns]
    # Add delta columns
    for metric in ["auc", "acc", "f1"]:
        col_base = f"CDND_baseline_{metric}"
        col_flu  = f"CDND_fluency_{metric}"
        if col_base in pivot.columns and col_flu in pivot.columns:
            pivot[f"delta_{metric}"] = pivot[col_flu] - pivot[col_base]
    summary_path = os.path.join(SAVE_ROOT, "compare_cdnd_all_datasets_summary.csv")
    pivot.to_csv(summary_path, encoding="utf-8-sig")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
