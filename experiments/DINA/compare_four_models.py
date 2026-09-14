# coding: utf-8
# File: experiments/DINA/run_neural_dina_compare.py

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

# 确保项目根目录在环境变量中
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# ===== 导入你的 Neural DINA 模型 =====
from models.dina.Neural_DINA import DINA as NeuralDINA
from models.dina.Neural_DINA_Fluency import DINAFluency as NeuralDINAFluency

# ===== 数据集加载复用 =====
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
# Metrics Setup
# =========================================================
def safe_auc(y_true, y_pred):
    if len(np.unique(y_true)) < 2: return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def safe_acc(y_true, y_prob, threshold=0.5):
    return float(accuracy_score(np.asarray(y_true).astype(int), (np.asarray(y_prob) >= threshold).astype(int)))


def safe_f1(y_true, y_prob, threshold=0.5):
    return float(
        f1_score(np.asarray(y_true).astype(int), (np.asarray(y_prob) >= threshold).astype(int), zero_division=0))


def safe_rmse(y_true, y_prob):
    return float(
        np.sqrt(mean_squared_error(np.asarray(y_true, dtype=np.float32), np.asarray(y_prob, dtype=np.float32))))


def get_score_by_metric(metrics: dict, metric_name: str) -> float:
    value = metrics.get(metric_name.lower(), float("nan"))
    if np.isnan(value): return -float("inf")
    if metric_name.lower() == "rmse": return -float(value)
    return float(value)


# =========================================================
# Train & Eval Loops
# =========================================================
def evaluate_neural_dina(model, data_loader, device="cpu"):
    model.dina_net = model.dina_net.to(device)
    model.dina_net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_data in data_loader:
            user_id, item_id, knowledge_emb, y = batch_data[:4]
            pred = model.dina_net(user_id.to(device), item_id.to(device), knowledge_emb.to(device))
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())
    return {"auc": safe_auc(y_true, y_pred), "acc": safe_acc(y_true, y_pred), "f1": safe_f1(y_true, y_pred),
            "rmse": safe_rmse(y_true, y_pred)}


def evaluate_fluency(model, data_loader, device="cpu"):
    model.dina_net = model.dina_net.to(device)
    model.dina_net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_data in data_loader:
            user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = batch_data
            pred = model.dina_net(
                user_id.to(device), item_id.to(device), knowledge_emb.to(device),
                opp.to(device), acc_prior.to(device), stability.to(device), return_aux=False
            )
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())
    return {"auc": safe_auc(y_true, y_pred), "acc": safe_acc(y_true, y_pred), "f1": safe_f1(y_true, y_pred),
            "rmse": safe_rmse(y_true, y_pred)}


def train_neural_dina_with_best_epoch(model, train_loader, valid_loader, args):
    model.dina_net = model.dina_net.to(args.device)
    optimizer = optim.Adam(model.dina_net.parameters(), lr=args.lr)
    loss_function = torch.nn.BCELoss()

    best_epoch, best_valid_metrics, best_state_dict = -1, None, None
    best_score = -float("inf")
    history = []

    for epoch_i in range(args.epochs):
        model.dina_net.train()
        losses = []
        for batch_data in train_loader:
            user_id, item_id, knowledge_emb, y = batch_data[:4]
            pred = model.dina_net(user_id.to(args.device), item_id.to(args.device), knowledge_emb.to(args.device))
            loss = loss_function(pred, y.to(args.device).float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        valid_metrics = evaluate_neural_dina(model, valid_loader, device=args.device)
        row = {
            "epoch": epoch_i, "train_loss": float(np.mean(losses)),
            "valid_auc": valid_metrics["auc"], "valid_acc": valid_metrics["acc"],
            "valid_f1": valid_metrics["f1"], "valid_rmse": valid_metrics["rmse"],
        }
        history.append(row)

        print(
            f"[Neural DINA][Epoch {epoch_i}] loss={row['train_loss']:.6f} | valid_auc={row['valid_auc']:.6f} valid_acc={row['valid_acc']:.6f}")

        score = get_score_by_metric(valid_metrics, args.select_metric)
        if score > best_score:
            best_score, best_epoch, best_valid_metrics = score, epoch_i, valid_metrics
            best_state_dict = copy.deepcopy(model.dina_net.state_dict())

    if best_state_dict is not None:
        model.dina_net.load_state_dict(best_state_dict)

    return {"best_epoch": best_epoch, "best_valid_metrics": best_valid_metrics, "history": history,
            "best_metric_name": args.select_metric}


def train_fluency_with_best_epoch(model, train_loader, valid_loader, args):
    model.dina_net = model.dina_net.to(args.device)
    optimizer = optim.Adam(model.dina_net.parameters(), lr=args.lr)

    best_epoch, best_valid_metrics, best_state_dict = -1, None, None
    best_score = -float("inf")
    history = []

    for epoch_i in range(args.epochs):
        model.dina_net.train()
        losses, ly_list = [], []

        for batch_data in train_loader:
            user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = batch_data
            out = model.dina_net(
                user_id.to(args.device), item_id.to(args.device), knowledge_emb.to(args.device),
                opp.to(args.device), acc_prior.to(args.device), stability.to(args.device), return_aux=True
            )

            ly = model.loss_y(out["p"], y.to(args.device).float())
            lopp = model.loss_opp(out["opp_hat"], out["opp_tgt"])
            lacc = model.loss_acc(out["acc_hat"], out["acc_tgt"])
            lsta = model.loss_sta(out["sta_hat"], out["sta_tgt"])

            loss = (1 - model.fluency_lambda) * ly + model.fluency_lambda * (lopp + lacc + lsta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            ly_list.append(ly.item())

        valid_metrics = evaluate_fluency(model, valid_loader, device=args.device)
        row = {
            "epoch": epoch_i, "train_loss": float(np.mean(losses)), "train_y_loss": float(np.mean(ly_list)),
            "valid_auc": valid_metrics["auc"], "valid_acc": valid_metrics["acc"],
            "valid_f1": valid_metrics["f1"], "valid_rmse": valid_metrics["rmse"],
        }
        history.append(row)

        print(
            f"[Neural DINA Fluency][Epoch {epoch_i}] loss={row['train_loss']:.6f} | valid_auc={row['valid_auc']:.6f} valid_acc={row['valid_acc']:.6f}")

        score = get_score_by_metric(valid_metrics, args.select_metric)
        if score > best_score:
            best_score, best_epoch, best_valid_metrics = score, epoch_i, valid_metrics
            best_state_dict = copy.deepcopy(model.dina_net.state_dict())

    if best_state_dict is not None:
        model.dina_net.load_state_dict(best_state_dict)

    return {"best_epoch": best_epoch, "best_valid_metrics": best_valid_metrics, "history": history,
            "best_metric_name": args.select_metric}


# =========================================================
# Dataset Wrapper
# =========================================================
def get_dataset(dataset_name, data_root, seed, use_fluency, alpha, beta):
    name = dataset_name.lower()

    # if name == "math1":
    #     return build_math1_splits_from_folder(os.path.join(data_root, "math2015", "Math1"), seed=seed,
    #                                           use_fluency=use_fluency, alpha=alpha, beta=beta, data_file="data.txt",
    #                                           q_file="q.txt", raw_file="rawdata.txt")
    # if name == "math2":
    #     return build_math2_splits_from_folder(os.path.join(data_root, "math2015", "Math2"), seed=seed,
    #                                           use_fluency=use_fluency, alpha=alpha, beta=beta)
    # if name == "frcsub":
    #     return build_frcsub_splits_from_folder(os.path.join(data_root, "math2015", "FrcSub"), seed=seed,
    #                                            use_fluency=use_fluency, alpha=alpha, beta=beta)
    # if name == "assist17":
    #     return build_assist17_splits_from_folder(os.path.join(data_root, "Assist17"), use_fluency=use_fluency,
    #                                              alpha=alpha, beta=beta, seed=seed, response_file="response.csv",
    #                                              q_file="q_matrix.csv")

    # if name == "assist09":
    #     return build_assist09_splits_from_folder(os.path.join(data_root, "assist09"), use_fluency=use_fluency,
    #                                              alpha=alpha, beta=beta)

    # 修复了冲突，这里只保留一条调用 assist09_dataset 工具的路径
    if name == "a0910":
        return build_assist09_splits_from_folder(os.path.join(data_root, "a0910"), use_fluency=use_fluency,
                                                 alpha=alpha, beta=beta)

    # if name == "junyi":
    #     data_dir = os.path.join(data_root, "junyi")
    #     return build_junyi_splits_from_csv(pd.read_csv(os.path.join(data_dir, "train.csv")),
    #                                        pd.read_csv(os.path.join(data_dir, "valid.csv")),
    #                                        pd.read_csv(os.path.join(data_dir, "test.csv")), use_fluency=use_fluency,
    #                                        alpha=alpha, beta=beta)
    # raise ValueError(f"Unsupported dataset: {dataset_name}")


# =========================================================
# Execution Pipeline
# =========================================================
def build_model(model_type, meta, args):
    if model_type == "neural_dina":
        return NeuralDINA(
            user_num=meta["student_n"],
            item_num=meta["exer_n"],
            hidden_dim=meta["knowledge_n"]
        )
    if model_type == "neural_dina_fluency":
        return NeuralDINAFluency(
            user_num=meta["student_n"],
            item_num=meta["exer_n"],
            hidden_dim=meta["knowledge_n"],
            lr=args.lr,
            fluency_lambda=args.fluency_lambda,
            dropout=args.dropout,
            flu_latent_dim=args.flu_latent_dim
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def run_one_model_one_dataset(args, dataset_name, model_type):
    use_fluency = (model_type == "neural_dina_fluency")
    train_ds, valid_ds, test_ds, meta = get_dataset(dataset_name, args.data_root, args.seed, use_fluency, args.alpha,
                                                    args.beta)
    print_meta(dataset_name, model_type, meta)

    train_loader, valid_loader, test_loader = build_loaders(train_ds, valid_ds, test_ds, args.batch_size,
                                                            args.num_workers)
    model = build_model(model_type, meta, args)

    if model_type == "neural_dina":
        train_info = train_neural_dina_with_best_epoch(model, train_loader, valid_loader, args)
        test_metrics = evaluate_neural_dina(model, test_loader, device=args.device)
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
        for m in ["neural_dina", "neural_dina_fluency"]:
            for metric in ["best_epoch", "test_auc", "test_acc", "test_f1", "test_rmse"]:
                row[f"{m}_{metric}"] = dd[m][metric] if m in dd else None

        if "neural_dina" in dd and "neural_dina_fluency" in dd:
            for metric in ["auc", "acc", "f1", "rmse"]:
                row[f"delta_{metric}"] = dd["neural_dina_fluency"][f"test_{metric}"] - dd["neural_dina"][
                    f"test_{metric}"]
        out.append(row)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Neural DINA and Neural DINA Fluency")

    parser.add_argument("--datasets", type=str, default="a0910")  # 默认跑 a0910
    parser.add_argument("--models", type=str, default="neural_dina,neural_dina_fluency")

    parser.add_argument("--data-root", type=str, default=os.path.join(ROOT, "data"))
    parser.add_argument("--save-root", type=str, default=os.path.join(ROOT, "checkpoints", "neural_dina_compare"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--select-metric", type=str, default="auc")

    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--fluency-lambda", type=float, default=0.5)
    parser.add_argument("--flu-latent-dim", type=int, default=16)

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
                print(f"\n[ERROR] {dataset_name} - {model_type}: {repr(e)}")
                summary.append({"dataset": dataset_name, "model": model_type, "error": repr(e)})

    save_json(os.path.join(args.save_root, "summary.json"), summary)
    comparison = build_comparison_table(summary)

    # 辅助打印格式化函数，防止NoneType报错
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
                f"  Neural DINA -> AUC={fmt(row['neural_dina_test_auc'])}, ACC={fmt(row['neural_dina_test_acc'])}, F1={fmt(row['neural_dina_test_f1'])}, RMSE={fmt(row['neural_dina_test_rmse'])}\n"
                f"  Fluency     -> AUC={fmt(row['neural_dina_fluency_test_auc'])}, ACC={fmt(row['neural_dina_fluency_test_acc'])}, F1={fmt(row['neural_dina_fluency_test_f1'])}, RMSE={fmt(row['neural_dina_fluency_test_rmse'])}\n"
                f"  Delta       -> AUC={fmt(row['delta_auc'])}, ACC={fmt(row['delta_acc'])}, F1={fmt(row['delta_f1'])}, RMSE={fmt(row['delta_rmse'])}\n"
            )


if __name__ == "__main__":
    main()