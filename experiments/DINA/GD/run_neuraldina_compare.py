# coding: utf-8
# File: experiments/neuraldina/run_neuraldina_compare.py

import os
import sys
import json
import copy
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# ===== models =====
# 按你的实际文件名改这两行
from models.dina.Neural_DINA import DINA as NeuralDINA
from models.dina.Neural_DINA_Fluency import DINAFluency

# ===== datasets =====
# 默认先复用 ncdm 目录下的数据构造脚本
from experiments.ncdm.math1_dataset import build_math1_splits_from_folder
from experiments.ncdm.math2_dataset import build_math2_splits_from_folder
from experiments.ncdm.frcsub_dataset import build_frcsub_splits_from_folder
from experiments.ncdm.Assist17_dataset import build_assist17_splits_from_folder
from experiments.ncdm.assist09_dataset import build_assist09_splits_from_folder
from experiments.ncdm.junyi_dataset import build_junyi_splits_from_csv


def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_loaders(train_ds, valid_ds, test_ds, batch_size: int, num_workers: int = 0):
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, valid_loader, test_loader


def print_meta(dataset_name: str, model_name: str, meta: dict):
    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name} | Model: {model_name}")
    print("=" * 70)
    for k, v in meta.items():
        print(f"{k}: {v}")


def safe_auc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def safe_acc(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_label = (np.asarray(y_prob) >= threshold).astype(int)
    return float(accuracy_score(y_true, y_label))


def safe_f1(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_label = (np.asarray(y_prob) >= threshold).astype(int)
    return float(f1_score(y_true, y_label, zero_division=0))


def safe_rmse(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    return float(np.sqrt(mean_squared_error(y_true, y_prob)))


def get_score_by_metric(metrics: dict, metric_name: str) -> float:
    """
    Higher is better for auc/acc/f1.
    Lower is better for rmse.
    """
    metric_name = metric_name.lower()
    value = metrics.get(metric_name, float("nan"))

    if np.isnan(value):
        return -float("inf")

    if metric_name == "rmse":
        return -float(value)

    return float(value)


def unpack_batch_for_neuraldina(batch_data):
    """
    Original NeuralDINA supports:
      1) (user_id, item_id, knowledge_emb, y)
      2) (user_id, item_id, knowledge_emb, y, opp, acc_prior, stability)  # ignore extra fluency fields
    """
    n = len(batch_data)
    if n >= 4:
        user_id, item_id, knowledge_emb, y = batch_data[:4]
    else:
        raise ValueError(f"Unexpected batch format for NeuralDINA: len(batch_data)={n}")
    return user_id, item_id, knowledge_emb, y


def unpack_batch_for_fluency(batch_data):
    """
    NeuralDINAFluency expects:
      (user_id, item_id, knowledge_emb, y, opp, acc_prior, stability)
    """
    n = len(batch_data)
    if n == 7:
        user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = batch_data
    else:
        raise ValueError(f"Unexpected batch format for DINAFluency: len(batch_data)={n}")
    return user_id, item_id, knowledge_emb, y, opp, acc_prior, stability


def evaluate_original_neuraldina(model, data_loader, device="cpu"):
    model.dina_net = model.dina_net.to(device)
    model.dina_net.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for batch_data in data_loader:
            user_id, item_id, knowledge_emb, y = unpack_batch_for_neuraldina(batch_data)

            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)

            pred = model.dina_net(user_id, item_id, knowledge_emb)

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())

    model.dina_net.train()

    return {
        "auc": safe_auc(y_true, y_pred),
        "acc": safe_acc(y_true, y_pred),
        "f1": safe_f1(y_true, y_pred),
        "rmse": safe_rmse(y_true, y_pred),
    }


def evaluate_fluency(model, data_loader, device="cpu"):
    model.dina_net = model.dina_net.to(device)
    model.dina_net.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for batch_data in data_loader:
            user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = unpack_batch_for_fluency(batch_data)

            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)
            opp = opp.to(device)
            acc_prior = acc_prior.to(device)
            stability = stability.to(device)

            pred = model.dina_net(
                user_id,
                item_id,
                knowledge_emb,
                opp,
                acc_prior,
                stability,
                return_aux=False,
            )

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())

    model.dina_net.train()

    return {
        "auc": safe_auc(y_true, y_pred),
        "acc": safe_acc(y_true, y_pred),
        "f1": safe_f1(y_true, y_pred),
        "rmse": safe_rmse(y_true, y_pred),
    }


def train_original_neuraldina_with_best_epoch(model, train_loader, valid_loader, args):
    model.dina_net = model.dina_net.to(args.device)
    optimizer = optim.Adam(model.dina_net.parameters(), lr=args.lr)
    loss_function = torch.nn.BCELoss()

    best_epoch = -1
    best_valid_metrics = None
    best_state_dict = None
    best_score = -float("inf")
    history = []

    for epoch_i in range(args.epochs):
        model.dina_net.train()
        losses = []

        for batch_data in train_loader:
            user_id, item_id, knowledge_emb, y = unpack_batch_for_neuraldina(batch_data)

            user_id = user_id.to(args.device)
            item_id = item_id.to(args.device)
            knowledge_emb = knowledge_emb.to(args.device)
            y = y.to(args.device).float()

            pred = model.dina_net(user_id, item_id, knowledge_emb)
            loss = loss_function(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        valid_metrics = evaluate_original_neuraldina(
            model,
            valid_loader,
            device=args.device,
        )

        row = {
            "epoch": epoch_i,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "train_y_loss": float(np.mean(losses)) if losses else float("nan"),
            "train_opp_loss": 0.0,
            "train_acc_loss": 0.0,
            "train_sta_loss": 0.0,
            "valid_auc": valid_metrics["auc"],
            "valid_acc": valid_metrics["acc"],
            "valid_f1": valid_metrics["f1"],
            "valid_rmse": valid_metrics["rmse"],
        }
        history.append(row)

        print(
            f"[NeuralDINA][Epoch {epoch_i}] "
            f"loss={row['train_loss']:.6f} | "
            f"valid_auc={row['valid_auc']:.6f} "
            f"valid_acc={row['valid_acc']:.6f} "
            f"valid_f1={row['valid_f1']:.6f} "
            f"valid_rmse={row['valid_rmse']:.6f}"
        )

        score = get_score_by_metric(valid_metrics, args.select_metric)

        if score > best_score:
            best_score = score
            best_epoch = epoch_i
            best_valid_metrics = valid_metrics
            best_state_dict = copy.deepcopy(model.dina_net.state_dict())

    if best_state_dict is not None:
        model.dina_net.load_state_dict(best_state_dict)

    return {
        "best_epoch": best_epoch,
        "best_valid_metrics": best_valid_metrics,
        "history": history,
        "best_metric_name": args.select_metric,
        "best_metric_score": best_score,
    }


def train_fluency_with_best_epoch(model, train_loader, valid_loader, args):
    model.dina_net = model.dina_net.to(args.device)
    optimizer = optim.Adam(model.dina_net.parameters(), lr=args.lr)

    best_epoch = -1
    best_valid_metrics = None
    best_state_dict = None
    best_score = -float("inf")
    history = []

    for epoch_i in range(args.epochs):
        model.dina_net.train()

        losses = []
        ly_list, lopp_list, lacc_list, lsta_list = [], [], [], []

        for batch_data in train_loader:
            user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = unpack_batch_for_fluency(batch_data)

            user_id = user_id.to(args.device)
            item_id = item_id.to(args.device)
            knowledge_emb = knowledge_emb.to(args.device)
            y = y.to(args.device).float()
            opp = opp.to(args.device)
            acc_prior = acc_prior.to(args.device)
            stability = stability.to(args.device)

            out = model.dina_net(
                user_id,
                item_id,
                knowledge_emb,
                opp,
                acc_prior,
                stability,
                return_aux=True,
            )

            ly = model.loss_y(out["p"], y)
            lopp = model.loss_opp(out["opp_hat"], out["opp_tgt"])
            lacc = model.loss_acc(out["acc_hat"], out["acc_tgt"])
            lsta = model.loss_sta(out["sta_hat"], out["sta_tgt"])

            loss = (
                (1 - model.fluency_lambda) * ly
                + model.fluency_lambda * (lopp + lacc + lsta)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            ly_list.append(ly.item())
            lopp_list.append(lopp.item())
            lacc_list.append(lacc.item())
            lsta_list.append(lsta.item())

        valid_metrics = evaluate_fluency(
            model,
            valid_loader,
            device=args.device,
        )

        row = {
            "epoch": epoch_i,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "train_y_loss": float(np.mean(ly_list)) if ly_list else float("nan"),
            "train_opp_loss": float(np.mean(lopp_list)) if lopp_list else float("nan"),
            "train_acc_loss": float(np.mean(lacc_list)) if lacc_list else float("nan"),
            "train_sta_loss": float(np.mean(lsta_list)) if lsta_list else float("nan"),
            "valid_auc": valid_metrics["auc"],
            "valid_acc": valid_metrics["acc"],
            "valid_f1": valid_metrics["f1"],
            "valid_rmse": valid_metrics["rmse"],
        }
        history.append(row)

        print(
            f"[Fluency][Epoch {epoch_i}] "
            f"loss={row['train_loss']:.6f} "
            f"(y={row['train_y_loss']:.6f}, opp={row['train_opp_loss']:.6f}, "
            f"acc={row['train_acc_loss']:.6f}, sta={row['train_sta_loss']:.6f}) | "
            f"valid_auc={row['valid_auc']:.6f} "
            f"valid_acc={row['valid_acc']:.6f} "
            f"valid_f1={row['valid_f1']:.6f} "
            f"valid_rmse={row['valid_rmse']:.6f}"
        )

        score = get_score_by_metric(valid_metrics, args.select_metric)

        if score > best_score:
            best_score = score
            best_epoch = epoch_i
            best_valid_metrics = valid_metrics
            best_state_dict = copy.deepcopy(model.dina_net.state_dict())

    if best_state_dict is not None:
        model.dina_net.load_state_dict(best_state_dict)

    return {
        "best_epoch": best_epoch,
        "best_valid_metrics": best_valid_metrics,
        "history": history,
        "best_metric_name": args.select_metric,
        "best_metric_score": best_score,
    }


def get_dataset(dataset_name, data_root, seed, use_fluency, alpha, beta):
    name = dataset_name.lower()

    if name == "math1":
        data_dir = os.path.join(data_root, "math2015", "Math1")
        return build_math1_splits_from_folder(
            data_dir,
            seed=seed,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
            data_file="data.txt",
            q_file="q.txt",
            raw_file="rawdata.txt",
        )

    if name == "math2":
        data_dir = os.path.join(data_root, "math2015", "Math2")
        return build_math2_splits_from_folder(
            data_dir,
            seed=seed,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
        )

    if name == "frcsub":
        data_dir = os.path.join(data_root, "math2015", "FrcSub")
        return build_frcsub_splits_from_folder(
            data_dir,
            seed=seed,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
        )

    if name == "assist17":
        data_dir = os.path.join(data_root, "Assist17")
        return build_assist17_splits_from_folder(
            data_dir,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
            seed=seed,
            response_file="response.csv",
            q_file="q_matrix.csv",
        )

    if name == "assist09":
        assist09_dir = os.path.join(data_root, "assist09")
        a0910_dir = os.path.join(data_root, "a0910")

        if os.path.isdir(assist09_dir) and os.path.isfile(os.path.join(assist09_dir, "train.csv")):
            data_dir = assist09_dir
        elif os.path.isdir(a0910_dir) and os.path.isfile(os.path.join(a0910_dir, "train.csv")):
            data_dir = a0910_dir
        else:
            raise FileNotFoundError(
                "assist09 dataset folder not found or missing train.csv. "
                f"Checked:\n  {assist09_dir}\n  {a0910_dir}"
            )

        return build_assist09_splits_from_folder(
            data_dir,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
        )

    if name == "junyi":
        data_dir = os.path.join(data_root, "junyi")
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        valid_df = pd.read_csv(os.path.join(data_dir, "valid.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

        return build_junyi_splits_from_csv(
            train_df,
            valid_df,
            test_df,
            use_fluency=use_fluency,
            alpha=alpha,
            beta=beta,
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_model(model_type, meta, args):
    if model_type == "neuraldina":
        return NeuralDINA(
            user_num=meta["student_n"],
            item_num=meta["exer_n"],
            hidden_dim=meta["knowledge_n"],
            ste=args.ste,
        )

    if model_type == "fluency":
        return DINAFluency(
            user_num=meta["student_n"],
            item_num=meta["exer_n"],
            hidden_dim=meta["knowledge_n"],
            ste=args.ste,
            lr=args.lr,
            max_slip=args.max_slip,
            max_guess=args.max_guess,
            dropout=args.dropout,
            fluency_lambda=args.fluency_lambda,
            flu_latent_dim=args.flu_latent_dim,
            gamma_init=args.gamma_init,
            rt_cap_ms=args.rt_cap_ms,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def run_one_model_one_dataset(args, dataset_name, model_type):
    use_fluency = (model_type == "fluency")

    train_ds, valid_ds, test_ds, meta = get_dataset(
        dataset_name=dataset_name,
        data_root=args.data_root,
        seed=args.seed,
        use_fluency=use_fluency,
        alpha=args.alpha,
        beta=args.beta,
    )

    print_meta(dataset_name, model_type, meta)

    train_loader, valid_loader, test_loader = build_loaders(
        train_ds,
        valid_ds,
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(model_type, meta, args)

    if model_type == "neuraldina":
        train_info = train_original_neuraldina_with_best_epoch(model, train_loader, valid_loader, args)
        test_metrics = evaluate_original_neuraldina(model, test_loader, device=args.device)
    else:
        train_info = train_fluency_with_best_epoch(model, train_loader, valid_loader, args)
        test_metrics = evaluate_fluency(model, test_loader, device=args.device)

    print(f"\n[Best Epoch] dataset={dataset_name}, model={model_type}, epoch={train_info['best_epoch']}")
    print(
        f"[Best Valid by {train_info['best_metric_name']}] "
        f"AUC={train_info['best_valid_metrics']['auc']:.6f}  "
        f"ACC={train_info['best_valid_metrics']['acc']:.6f}  "
        f"F1={train_info['best_valid_metrics']['f1']:.6f}  "
        f"RMSE={train_info['best_valid_metrics']['rmse']:.6f}"
    )
    print(
        f"[Test @ Best Epoch] "
        f"AUC={test_metrics['auc']:.6f}  "
        f"ACC={test_metrics['acc']:.6f}  "
        f"F1={test_metrics['f1']:.6f}  "
        f"RMSE={test_metrics['rmse']:.6f}"
    )

    save_dir = os.path.join(args.save_root, model_type, dataset_name)
    ensure_dir(save_dir)

    model_path = os.path.join(save_dir, "best_model.pt")
    model.save(model_path)

    save_json(os.path.join(save_dir, "history.json"), train_info["history"])

    metrics = {
        "dataset": dataset_name,
        "model": model_type,
        "student_n": int(meta["student_n"]),
        "exer_n": int(meta["exer_n"]),
        "knowledge_n": int(meta["knowledge_n"]),
        "hidden_dim": int(meta["knowledge_n"]),
        "ste": bool(args.ste),
        "n_train": int(meta["n_train"]),
        "n_valid": int(meta["n_valid"]),
        "n_test": int(meta["n_test"]),
        "best_epoch": int(train_info["best_epoch"]),
        "select_metric": train_info["best_metric_name"],
        "best_valid_auc": float(train_info["best_valid_metrics"]["auc"]),
        "best_valid_acc": float(train_info["best_valid_metrics"]["acc"]),
        "best_valid_f1": float(train_info["best_valid_metrics"]["f1"]),
        "best_valid_rmse": float(train_info["best_valid_metrics"]["rmse"]),
        "test_auc": float(test_metrics["auc"]),
        "test_acc": float(test_metrics["acc"]),
        "test_f1": float(test_metrics["f1"]),
        "test_rmse": float(test_metrics["rmse"]),
        "model_path": model_path,
    }

    save_json(os.path.join(save_dir, "metrics.json"), metrics)
    return metrics


def build_comparison_table(rows):
    out = []
    grouped = {}

    for r in rows:
        if "error" in r:
            continue
        grouped.setdefault(r["dataset"], {})
        grouped[r["dataset"]][r["model"]] = r

    for dataset, dd in grouped.items():
        row = {
            "dataset": dataset,

            "neuraldina_best_epoch": dd["neuraldina"]["best_epoch"] if "neuraldina" in dd else None,
            "neuraldina_test_auc": dd["neuraldina"]["test_auc"] if "neuraldina" in dd else None,
            "neuraldina_test_acc": dd["neuraldina"]["test_acc"] if "neuraldina" in dd else None,
            "neuraldina_test_f1": dd["neuraldina"]["test_f1"] if "neuraldina" in dd else None,
            "neuraldina_test_rmse": dd["neuraldina"]["test_rmse"] if "neuraldina" in dd else None,

            "fluency_best_epoch": dd["fluency"]["best_epoch"] if "fluency" in dd else None,
            "fluency_test_auc": dd["fluency"]["test_auc"] if "fluency" in dd else None,
            "fluency_test_acc": dd["fluency"]["test_acc"] if "fluency" in dd else None,
            "fluency_test_f1": dd["fluency"]["test_f1"] if "fluency" in dd else None,
            "fluency_test_rmse": dd["fluency"]["test_rmse"] if "fluency" in dd else None,
        }

        if "neuraldina" in dd and "fluency" in dd:
            row["delta_auc"] = dd["fluency"]["test_auc"] - dd["neuraldina"]["test_auc"]
            row["delta_acc"] = dd["fluency"]["test_acc"] - dd["neuraldina"]["test_acc"]
            row["delta_f1"] = dd["fluency"]["test_f1"] - dd["neuraldina"]["test_f1"]
            row["delta_rmse"] = dd["fluency"]["test_rmse"] - dd["neuraldina"]["test_rmse"]
        else:
            row["delta_auc"] = None
            row["delta_acc"] = None
            row["delta_f1"] = None
            row["delta_rmse"] = None

        out.append(row)

    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Compare original NeuralDINA and DINAFluency")

    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--models", type=str, default="neuraldina,fluency")
    parser.add_argument("--data-root", type=str, default=os.path.join(ROOT, "data"))
    parser.add_argument("--save-root", type=str, default=os.path.join(ROOT, "checkpoints", "neuraldina_compare"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument(
        "--select-metric",
        type=str,
        default="auc",
        choices=["auc", "acc", "f1", "rmse"],
        help="Metric used on validation set to choose the best epoch."
    )

    # ===== NeuralDINA =====
    parser.add_argument("--ste", action="store_true")
    parser.add_argument("--max-slip", type=float, default=0.4)
    parser.add_argument("--max-guess", type=float, default=0.4)

    # ===== Fluency =====
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--fluency-lambda", type=float, default=0.5)
    parser.add_argument("--flu-latent-dim", type=int, default=16)
    parser.add_argument("--rt-cap-ms", type=float, default=120000.0)
    parser.add_argument("--gamma-init", type=float, default=1.0)

    # ===== proxy build =====
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    ensure_dir(args.save_root)

    all_datasets = ["math1", "math2", "frcsub", "assist17", "assist09", "junyi"]
    all_models = ["neuraldina", "fluency"]

    if args.datasets.strip().lower() == "all":
        dataset_list = all_datasets
    else:
        dataset_list = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]

    if args.models.strip().lower() == "all":
        model_list = all_models
    else:
        model_list = [x.strip().lower() for x in args.models.split(",") if x.strip()]

    summary = []

    for dataset_name in dataset_list:
        for model_type in model_list:
            try:
                metrics = run_one_model_one_dataset(args, dataset_name, model_type)
                summary.append(metrics)
            except Exception as e:
                print(f"\n[ERROR] dataset={dataset_name}, model={model_type}")
                print(repr(e))
                summary.append({
                    "dataset": dataset_name,
                    "model": model_type,
                    "error": repr(e),
                })

    save_json(os.path.join(args.save_root, "summary.json"), summary)

    comparison = build_comparison_table(summary)
    save_json(os.path.join(args.save_root, "comparison.json"), comparison)

    if len(comparison) > 0:
        pd.DataFrame(comparison).to_csv(
            os.path.join(args.save_root, "comparison.csv"),
            index=False
        )

    print("\n" + "=" * 90)
    print(f"Comparison Summary (best epoch selected by valid {args.select_metric})")
    print("=" * 90)
    for row in comparison:
        print(
            f'{row["dataset"]}\n'
            f'  NeuralDINA -> epoch={row["neuraldina_best_epoch"]}, '
            f'AUC={row["neuraldina_test_auc"]:.6f}, '
            f'ACC={row["neuraldina_test_acc"]:.6f}, '
            f'F1={row["neuraldina_test_f1"]:.6f}, '
            f'RMSE={row["neuraldina_test_rmse"]:.6f}\n'
            f'  Fluency    -> epoch={row["fluency_best_epoch"]}, '
            f'AUC={row["fluency_test_auc"]:.6f}, '
            f'ACC={row["fluency_test_acc"]:.6f}, '
            f'F1={row["fluency_test_f1"]:.6f}, '
            f'RMSE={row["fluency_test_rmse"]:.6f}\n'
            f'  Delta       -> '
            f'AUC={row["delta_auc"]:.6f}, '
            f'ACC={row["delta_acc"]:.6f}, '
            f'F1={row["delta_f1"]:.6f}, '
            f'RMSE={row["delta_rmse"]:.6f}'
        )


if __name__ == "__main__":
    main()