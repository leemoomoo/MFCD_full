# coding: utf-8
# File: experiments/IRT/run_irt_compare.py

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

# =========================================================
# 路径与环境初始化
# =========================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# ===== models (完全适配你本地的文件命名) =====
from models.irt.IRT import IRT
from models.irt.IRT_fluency import IRTFluency

# ===== datasets (复用 NCDM 的数据集构造逻辑) =====
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def print_meta(dataset_name: str, model_name: str, meta: dict):
    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name} | Model: {model_name}")
    print("=" * 70)
    for k, v in meta.items():
        print(f"{k}: {v}")


# =========================================================
# 评价指标 (Metrics)
# =========================================================
def safe_auc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) < 2: return float("nan")
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
    metric_name = metric_name.lower()
    value = metrics.get(metric_name, float("nan"))
    if np.isnan(value): return -float("inf")
    if metric_name == "rmse": return -float(value)
    return float(value)


# =========================================================
# Batch 解包防报错逻辑 (核心修复)
# =========================================================
def unpack_batch_for_irt(batch_data):
    n = len(batch_data)
    if n == 3:
        user_id, item_id, y = batch_data
    elif n >= 4:
        user_id, item_id, _, y = batch_data[:4]
    else:
        raise ValueError(f"Unexpected batch format for IRT: len(batch_data)={n}")
    return user_id, item_id, y


def unpack_batch_for_fluency(batch_data):
    n = len(batch_data)
    if n == 6:
        user_id, item_id, y, opp, acc_prior, stability = batch_data
    elif n == 7:
        user_id, item_id, _, y, opp, acc_prior, stability = batch_data
    else:
        raise ValueError(f"Unexpected batch format for IRTFluency: len(batch_data)={n}")
    return user_id, item_id, y, opp, acc_prior, stability


# =========================================================
# 模型训练与评估逻辑
# =========================================================
def evaluate_original_irt(model, data_loader, device="cpu"):
    model.irt_net = model.irt_net.to(device)
    model.irt_net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_data in data_loader:
            user_id, item_id, y = unpack_batch_for_irt(batch_data)
            pred = model.irt_net(user_id.to(device), item_id.to(device))
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())
    model.irt_net.train()
    return {"auc": safe_auc(y_true, y_pred), "acc": safe_acc(y_true, y_pred), "f1": safe_f1(y_true, y_pred),
            "rmse": safe_rmse(y_true, y_pred)}


def evaluate_fluency(model, data_loader, device="cpu"):
    model.net = model.net.to(device)
    model.net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_data in data_loader:
            user_id, item_id, y, opp, acc_prior, stability = unpack_batch_for_fluency(batch_data)
            pred = model.net(
                user_id.to(device), item_id.to(device),
                opp.to(device).float(), acc_prior.to(device).float(), stability.to(device).float()
            )
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())
    model.net.train()
    return {"auc": safe_auc(y_true, y_pred), "acc": safe_acc(y_true, y_pred), "f1": safe_f1(y_true, y_pred),
            "rmse": safe_rmse(y_true, y_pred)}


def train_original_irt_with_best_epoch(model, train_loader, valid_loader, args):
    model.irt_net = model.irt_net.to(args.device)
    optimizer = optim.Adam(model.irt_net.parameters(), lr=args.lr)
    loss_function = torch.nn.BCELoss()

    best_epoch, best_valid_metrics, best_state_dict = -1, None, None
    best_score = -float("inf")
    history = []

    for epoch_i in range(args.epochs):
        model.irt_net.train()
        losses = []
        for batch_data in train_loader:
            user_id, item_id, y = unpack_batch_for_irt(batch_data)
            pred = model.irt_net(user_id.to(args.device), item_id.to(args.device))
            loss = loss_function(pred, y.to(args.device).float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        valid_metrics = evaluate_original_irt(model, valid_loader, device=args.device)
        row = {
            "epoch": epoch_i, "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "valid_auc": valid_metrics["auc"], "valid_acc": valid_metrics["acc"],
            "valid_f1": valid_metrics["f1"], "valid_rmse": valid_metrics["rmse"],
        }
        history.append(row)

        print(
            f"[IRT][Epoch {epoch_i}] loss={row['train_loss']:.6f} | valid_auc={row['valid_auc']:.6f} valid_acc={row['valid_acc']:.6f}")

        score = get_score_by_metric(valid_metrics, args.select_metric)
        if score > best_score:
            best_score, best_epoch, best_valid_metrics = score, epoch_i, valid_metrics
            best_state_dict = copy.deepcopy(model.irt_net.state_dict())

    if best_state_dict is not None:
        model.irt_net.load_state_dict(best_state_dict)

    return {"best_epoch": best_epoch, "best_valid_metrics": best_valid_metrics, "history": history,
            "best_metric_name": args.select_metric}


def train_fluency_with_best_epoch(model, train_loader, valid_loader, args):
    model.net = model.net.to(args.device)
    optimizer = optim.Adam(model.net.parameters(), lr=args.lr)
    loss_function = torch.nn.BCELoss()

    best_epoch, best_valid_metrics, best_state_dict = -1, None, None
    best_score = -float("inf")
    history = []

    for epoch_i in range(args.epochs):
        model.net.train()
        losses = []
        for batch_data in train_loader:
            user_id, item_id, y, opp, acc_prior, stability = unpack_batch_for_fluency(batch_data)
            pred = model.net(
                user_id.to(args.device), item_id.to(args.device),
                opp.to(args.device).float(), acc_prior.to(args.device).float(), stability.to(args.device).float()
            )
            loss = loss_function(pred, y.to(args.device).float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        valid_metrics = evaluate_fluency(model, valid_loader, device=args.device)
        row = {
            "epoch": epoch_i, "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "valid_auc": valid_metrics["auc"], "valid_acc": valid_metrics["acc"],
            "valid_f1": valid_metrics["f1"], "valid_rmse": valid_metrics["rmse"],
        }
        history.append(row)

        print(
            f"[IRT Fluency][Epoch {epoch_i}] loss={row['train_loss']:.6f} | valid_auc={row['valid_auc']:.6f} valid_acc={row['valid_acc']:.6f}")

        score = get_score_by_metric(valid_metrics, args.select_metric)
        if score > best_score:
            best_score, best_epoch, best_valid_metrics = score, epoch_i, valid_metrics
            best_state_dict = copy.deepcopy(model.net.state_dict())

    if best_state_dict is not None:
        model.net.load_state_dict(best_state_dict)

    return {"best_epoch": best_epoch, "best_valid_metrics": best_valid_metrics, "history": history,
            "best_metric_name": args.select_metric}


# =========================================================
# 数据集构造层
# =========================================================
def get_dataset(dataset_name, data_root, seed, use_fluency, alpha, beta):
    name = dataset_name.lower()
    if name == "math1":
        return build_math1_splits_from_folder(os.path.join(data_root, "math2015", "Math1"), seed=seed,
                                              use_fluency=use_fluency, alpha=alpha, beta=beta, data_file="data.txt",
                                              q_file="q.txt", raw_file="rawdata.txt")
    if name == "math2":
        return build_math2_splits_from_folder(os.path.join(data_root, "math2015", "Math2"), seed=seed,
                                              use_fluency=use_fluency, alpha=alpha, beta=beta)
    if name == "frcsub":
        return build_frcsub_splits_from_folder(os.path.join(data_root, "math2015", "FrcSub"), seed=seed,
                                               use_fluency=use_fluency, alpha=alpha, beta=beta)
    if name == "assist17":
        return build_assist17_splits_from_folder(os.path.join(data_root, "Assist17"), use_fluency=use_fluency,
                                                 alpha=alpha, beta=beta, seed=seed, response_file="response.csv",
                                                 q_file="q_matrix.csv")
    if name == "assist09":
        assist09_dir = os.path.join(data_root, "assist09")
        a0910_dir = os.path.join(data_root, "a0910")
        if os.path.isdir(assist09_dir) and os.path.isfile(os.path.join(assist09_dir, "train.csv")):
            data_dir = assist09_dir
        elif os.path.isdir(a0910_dir) and os.path.isfile(os.path.join(a0910_dir, "train.csv")):
            data_dir = a0910_dir
        else:
            raise FileNotFoundError("assist09 dataset folder not found or missing train.csv.")
        return build_assist09_splits_from_folder(data_dir, use_fluency=use_fluency, alpha=alpha, beta=beta)
    if name == "junyi":
        data_dir = os.path.join(data_root, "junyi")
        return build_junyi_splits_from_csv(pd.read_csv(os.path.join(data_dir, "train.csv")),
                                           pd.read_csv(os.path.join(data_dir, "valid.csv")),
                                           pd.read_csv(os.path.join(data_dir, "test.csv")), use_fluency=use_fluency,
                                           alpha=alpha, beta=beta)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


# =========================================================
# 执行总管 (Pipeline)
# =========================================================
def build_model(model_type, meta, args):
    if model_type == "irt":
        return IRT(user_num=meta["student_n"], item_num=meta["exer_n"], value_range=args.value_range,
                   a_range=args.a_range)
    if model_type == "fluency":
        return IRTFluency(user_num=meta["student_n"], item_num=meta["exer_n"], value_range=args.value_range,
                          a_range=args.a_range)
    raise ValueError(f"Unsupported model type: {model_type}")


def run_one_model_one_dataset(args, dataset_name, model_type):
    use_fluency = (model_type == "fluency")
    train_ds, valid_ds, test_ds, meta = get_dataset(dataset_name, args.data_root, args.seed, use_fluency, args.alpha,
                                                    args.beta)
    print_meta(dataset_name, model_type, meta)

    train_loader, valid_loader, test_loader = build_loaders(train_ds, valid_ds, test_ds, args.batch_size,
                                                            args.num_workers)
    model = build_model(model_type, meta, args)

    if model_type == "irt":
        train_info = train_original_irt_with_best_epoch(model, train_loader, valid_loader, args)
        test_metrics = evaluate_original_irt(model, test_loader, device=args.device)
    else:
        train_info = train_fluency_with_best_epoch(model, train_loader, valid_loader, args)
        test_metrics = evaluate_fluency(model, test_loader, device=args.device)

    save_dir = os.path.join(args.save_root, model_type, dataset_name)
    ensure_dir(save_dir)
    model.save(os.path.join(save_dir, "best_model.pt"))
    save_json(os.path.join(save_dir, "history.json"), train_info["history"])

    return {
        "dataset": dataset_name, "model": model_type,
        "test_auc": float(test_metrics["auc"]), "test_acc": float(test_metrics["acc"]),
        "test_f1": float(test_metrics["f1"]), "test_rmse": float(test_metrics["rmse"]),
        "best_epoch": int(train_info["best_epoch"])
    }


def build_comparison_table(rows):
    grouped, out = {}, []
    for r in rows:
        if "error" in r: continue
        grouped.setdefault(r["dataset"], {})[r["model"]] = r

    for dataset, dd in grouped.items():
        row = {"dataset": dataset}
        for m in ["irt", "fluency"]:
            for metric in ["best_epoch", "test_auc", "test_acc", "test_f1", "test_rmse"]:
                row[f"{m}_{metric}"] = dd[m][metric] if m in dd else None

        if "irt" in dd and "fluency" in dd:
            for metric in ["auc", "acc", "f1", "rmse"]:
                row[f"delta_{metric}"] = dd["fluency"][f"test_{metric}"] - dd["irt"][f"test_{metric}"]
        out.append(row)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Compare original IRT and IRTFluency")
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--models", type=str, default="irt,fluency")
    parser.add_argument("--data-root", type=str, default=os.path.join(ROOT, "data"))
    parser.add_argument("--save-root", type=str, default=os.path.join(ROOT, "checkpoints", "irt_compare"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--select-metric", type=str, default="auc")
    parser.add_argument("--value-range", type=float, default=None)
    parser.add_argument("--a-range", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.save_root)

    dataset_list = [x.strip().lower() for x in args.datasets.split(",")] if args.datasets != "all" else ["math1",
                                                                                                         "math2",
                                                                                                         "frcsub",
                                                                                                         "assist17",
                                                                                                         "assist09",
                                                                                                         "junyi"]
    model_list = [x.strip().lower() for x in args.models.split(",")]

    summary = []
    for dataset_name in dataset_list:
        for model_type in model_list:
            try:
                metrics = run_one_model_one_dataset(args, dataset_name, model_type)
                summary.append(metrics)
            except Exception as e:
                print(f"\n[ERROR] dataset={dataset_name}, model={model_type}: {repr(e)}")
                summary.append({"dataset": dataset_name, "model": model_type, "error": repr(e)})

    save_json(os.path.join(args.save_root, "summary.json"), summary)
    comparison = build_comparison_table(summary)

    # 辅助打印格式化函数，防止 None 报错
    def fmt(x):
        if x is None:
            return "None"
        if isinstance(x, float) and np.isnan(x):
            return "nan"
        return f"{float(x):.4f}"

    if comparison:
        pd.DataFrame(comparison).to_csv(os.path.join(args.save_root, "comparison.csv"), index=False)
        print("\n" + "=" * 110)
        print(f"Comparison Summary (best epoch selected by valid {args.select_metric})")
        print("=" * 110)
        for row in comparison:
            print(
                f"[{row['dataset'].upper()}]\n"
                f"  IRT     -> AUC={fmt(row['irt_test_auc'])}, ACC={fmt(row['irt_test_acc'])}, F1={fmt(row['irt_test_f1'])}, RMSE={fmt(row['irt_test_rmse'])}\n"
                f"  Fluency -> AUC={fmt(row['fluency_test_auc'])}, ACC={fmt(row['fluency_test_acc'])}, F1={fmt(row['fluency_test_f1'])}, RMSE={fmt(row['fluency_test_rmse'])}\n"
                f"  Delta   -> AUC={fmt(row['delta_auc'])}, ACC={fmt(row['delta_acc'])}, F1={fmt(row['delta_f1'])}, RMSE={fmt(row['delta_rmse'])}\n"
            )


if __name__ == "__main__":
    main()