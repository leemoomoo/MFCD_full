# coding: utf-8
"""
run_kcd_fluency_auc.py
-------------------------------------------------------
对比 KCD(实际为 model.py 中的 NCD) 与 KCD_Fluency

支持:
- a0910 / assist09
- Assist17
- junyi

评价方式:
- VALID AUC 选择 best epoch
- TEST 输出 AUC / ACC / F1 / RMSE / MAE

输出:
- compare_kcd_fluency_auc_full.csv
- compare_kcd_fluency_auc_summary.csv
- 终端 Comparison Summary
-------------------------------------------------------
"""

import sys
import copy
import random
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
)

# =====================================================
# 0. 路径与模块导入
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
OUT_DIR = BASE_DIR / "outputs" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BASE_DIR))

print(f"BASE_DIR = {BASE_DIR}")
print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"OUT_DIR = {OUT_DIR}")

from kcd_fluency import KCD_Fluency
from model import NCD as KCD

from experiments.ncdm.assist09_dataset import build_assist09_splits_from_folder
from experiments.ncdm.Assist17_dataset import build_assist17_splits_from_folder
from experiments.ncdm.junyi_dataset import build_junyi_splits_from_csv


# =====================================================
# 1. 随机种子
# =====================================================
def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================
# 2. 通用工具
# =====================================================
def get_net(model):
    net = getattr(model, "ncdm_net", None)
    if net is None:
        net = getattr(model, "net", None)
    if net is None:
        raise AttributeError("模型没有 ncdm_net 或 net")
    return net


def load_json_int_key(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def sort_and_fill_dict(input_dict, max_key, min_key=0):
    if not input_dict:
        return {}

    example_length = len(next(iter(input_dict.values())))
    all_keys = list(range(min_key, max_key))

    filled = dict(input_dict)
    for key in all_keys:
        if key not in filled:
            filled[key] = [0] * example_length

    return {key: filled[key] for key in sorted(all_keys)}


def build_loaders_from_datasets(train_ds, valid_ds, test_ds, batch_size=64):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, valid_loader, test_loader


def unpack_batch_for_eval(batch, model_type):
    """
    KCD(NCD): 使用前4个字段
      user, item, knowledge, score

    KCD_Fluency:
      user, item, knowledge, score, opp, acc_prior, stability
    """
    if model_type.lower() == "kcd":
        if len(batch) < 4:
            raise ValueError(f"KCD 期望 batch 至少 4 个字段，实际为 {len(batch)}")
        user, item, knowledge, score = batch[:4]
        return user, item, knowledge, score, None, None, None

    if len(batch) != 7:
        raise ValueError(f"KCD_Fluency 期望 batch 长度为 7，实际为 {len(batch)}")
    user, item, knowledge, score, opp, acc_prior, stability = batch
    return user, item, knowledge, score, opp, acc_prior, stability


def _to_scalar_tensor(v, device):
    if torch.is_tensor(v):
        if v.numel() == 1:
            return v.float().to(device)
        return v.float().mean().to(device)
    return torch.tensor(float(v), dtype=torch.float32, device=device)


def _safe_pred_tensor(pred: torch.Tensor) -> torch.Tensor:
    pred = torch.nan_to_num(pred, nan=0.5, posinf=1.0, neginf=0.0)
    pred = torch.clamp(pred, min=1e-6, max=1 - 1e-6)
    return pred


def call_fluency_net(net, user, item, knowledge, opp, acc_prior, stability, want_aux=False, device="cpu"):
    """
    兼容多种 KCD_Fluency forward 写法:
    1) net(..., return_aux=True/False)
    2) net(...)
    3) 返回 pred
    4) 返回 (pred, aux)
    5) 返回 (pred, aux1, aux2, ...)
    6) 返回 dict，预测值在 key='p' 或 'pred' 或 'output'
    """
    try:
        out = net(
            user, item, knowledge,
            opp, acc_prior, stability,
            return_aux=want_aux
        )
    except TypeError:
        out = net(user, item, knowledge, opp, acc_prior, stability)

    # ---------- dict 返回 ----------
    if isinstance(out, dict):
        if "p" in out:
            pred = out["p"]
        elif "pred" in out:
            pred = out["pred"]
        elif "output" in out:
            pred = out["output"]
        else:
            raise ValueError(f"KCD_Fluency 返回 dict，但未找到预测键。keys={list(out.keys())}")

        pred = _safe_pred_tensor(pred)

        aux_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        if want_aux:
            mse = torch.nn.MSELoss()

            if "opp_hat" in out and "opp_tgt" in out:
                aux_loss = aux_loss + mse(out["opp_hat"], out["opp_tgt"])
            if "acc_hat" in out and "acc_tgt" in out:
                aux_loss = aux_loss + mse(out["acc_hat"], out["acc_tgt"])
            if "sta_hat" in out and "sta_tgt" in out:
                aux_loss = aux_loss + mse(out["sta_hat"], out["sta_tgt"])

        return pred, aux_loss

    # ---------- tuple 返回 ----------
    if isinstance(out, tuple):
        pred = out[0]
        pred = _safe_pred_tensor(pred)

        if want_aux and len(out) >= 2:
            aux_terms = [_to_scalar_tensor(v, device) for v in out[1:]]
            aux_loss = sum(aux_terms) if len(aux_terms) > 0 else torch.tensor(0.0, device=device)
        else:
            aux_loss = torch.tensor(0.0, device=device)

        return pred, aux_loss

    # ---------- 直接 tensor 返回 ----------
    pred = _safe_pred_tensor(out)
    aux_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    return pred, aux_loss


# =====================================================
# 3. 数据集根目录自动探测
# =====================================================
def locate_data_root():
    candidates = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "dataset" / "dataset" / "open",
        BASE_DIR / "dataset" / "dataset" / "open",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "找不到数据根目录，请检查以下候选路径：\n" +
        "\n".join(str(p) for p in candidates)
    )


# =====================================================
# 4. 不同数据集的读取
# =====================================================
def load_dataset_bundle(dataset_name: str):
    data_root = locate_data_root()
    name = dataset_name.lower()

    if name in {"a0910", "assist09", "assist09new"}:
        data_dir = data_root / "a0910"
        train_ds, valid_ds, test_ds, meta = build_assist09_splits_from_folder(
            data_dir=str(data_dir),
            use_fluency=True,
            alpha=1.0,
            beta=1.0,
            allow_user_oov=True,
            train_file="train.csv",
            valid_file="valid.csv",
            test_file="test.csv",
            item_file="item.csv",
        )
        return data_dir, train_ds, valid_ds, test_ds, meta

    elif name == "assist17":
        data_dir = data_root / "Assist17"
        train_ds, valid_ds, test_ds, meta = build_assist17_splits_from_folder(
            data_dir=str(data_dir),
            use_fluency=True,
            alpha=1.0,
            beta=1.0,
            allow_user_oov=True,
            seed=2025,
            response_file="response.csv",
            q_file="q_matrix.csv",
        )
        return data_dir, train_ds, valid_ds, test_ds, meta

    elif name == "junyi":
        data_dir = data_root / "junyi"

        train_path = data_dir / "train.csv"
        valid_path = data_dir / "valid.csv"
        test_path = data_dir / "test.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"文件不存在: {train_path}")
        if not valid_path.exists():
            raise FileNotFoundError(f"文件不存在: {valid_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"文件不存在: {test_path}")

        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)

        train_ds, valid_ds, test_ds, meta = build_junyi_splits_from_csv(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            use_fluency=True,
            alpha=1.0,
            beta=1.0,
            allow_user_oov=True,
        )
        return data_dir, train_ds, valid_ds, test_ds, meta

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


# =====================================================
# 5. 模型构建
# =====================================================
def build_model(
    model_type,
    data_dir,
    meta,
    lambda_kcd=1.0,
    lambda_opp=0.1,
    lambda_acc=0.05,
    lambda_sta=0.05,
    flu_latent_dim=16,
    dropout=0.5,
):
    user_n = int(meta["student_n"])
    item_n = int(meta["exer_n"])
    knowledge_n = int(meta["knowledge_n"])

    user_emb_path = Path(data_dir) / "user_emb.pkl"
    item_emb_path = Path(data_dir) / "item_emb.pkl"
    sim_stu_path = Path(data_dir) / "similar_students.json"
    sim_exer_path = Path(data_dir) / "similar_exercises.json"

    if user_emb_path.exists() and item_emb_path.exists():
        with open(user_emb_path, "rb") as f:
            usr_emb = pickle.load(f)
        with open(item_emb_path, "rb") as f:
            itm_emb = pickle.load(f)
        print("loaded existing embeddings.")
    else:
        emb_dim = 128
        rng = np.random.default_rng(2026)
        usr_emb = rng.normal(0, 0.01, size=(user_n, emb_dim)).astype(np.float32)
        itm_emb = rng.normal(0, 0.01, size=(item_n, emb_dim)).astype(np.float32)
        print("embedding files not found, using random embeddings.")

    if sim_stu_path.exists():
        similar_students = load_json_int_key(sim_stu_path)
        similar_students = sort_and_fill_dict(similar_students, user_n)
    else:
        similar_students = {i: [i] for i in range(user_n)}

    if sim_exer_path.exists():
        similar_exercises = load_json_int_key(sim_exer_path)
        similar_exercises = sort_and_fill_dict(similar_exercises, item_n)
    else:
        similar_exercises = {i: [i] for i in range(item_n)}

    if model_type.lower() == "kcd":
        log_name = str(OUT_DIR / f"{Path(data_dir).name}_KCD.log")
        best_model = str(OUT_DIR / f"{Path(data_dir).name}_KCD_best.pt")

        model = KCD(
            knowledge_n=knowledge_n,
            exer_n=item_n,
            student_n=user_n,
            log_name=log_name,
            best_model=best_model,
            usr_emb=usr_emb,
            itm_emb=itm_emb,
        )
        model_name = "KCD"

    elif model_type.lower() == "kcd_fluency":
        log_name = str(OUT_DIR / f"{Path(data_dir).name}_KCD_Fluency.log")
        best_model = str(OUT_DIR / f"{Path(data_dir).name}_KCD_Fluency_best.pt")

        model = KCD_Fluency(
            knowledge_n=knowledge_n,
            exer_n=item_n,
            student_n=user_n,
            log_name=log_name,
            best_model=best_model,
            usr_emb=usr_emb,
            itm_emb=itm_emb,
            similar_students=similar_students,
            similar_exercises=similar_exercises,
            lambda_kcd=lambda_kcd,
            lambda_opp=lambda_opp,
            lambda_acc=lambda_acc,
            lambda_sta=lambda_sta,
            flu_latent_dim=flu_latent_dim,
            dropout=dropout,
        )
        model_name = "KCD_Fluency"

    else:
        raise ValueError(f"不支持的 model_type: {model_type}")

    return model, model_name, user_n, item_n, knowledge_n


# =====================================================
# 6. 训练与评估
# =====================================================
def evaluate_metrics(model, loader, model_type, device="cuda", pred_threshold=0.5):
    net = get_net(model).to(device)
    net.eval()

    y_true, y_prob, y_pred = [], [], []

    with torch.no_grad():
        for batch in loader:
            user, item, knowledge, score, opp, acc_prior, stability = unpack_batch_for_eval(batch, model_type)

            user = user.to(device)
            item = item.to(device)
            knowledge = knowledge.to(device)
            score = score.to(device)

            if model_type.lower() == "kcd":
                prob = net(user, item, knowledge)
                prob = _safe_pred_tensor(prob)
            else:
                opp = opp.to(device)
                acc_prior = acc_prior.to(device)
                stability = stability.to(device)
                prob, _ = call_fluency_net(
                    net, user, item, knowledge, opp, acc_prior, stability,
                    want_aux=False,
                    device=device
                )

            pred = (prob >= pred_threshold).float()

            y_true.extend(score.detach().cpu().numpy().tolist())
            y_prob.extend(prob.detach().cpu().numpy().tolist())
            y_pred.extend(pred.detach().cpu().numpy().tolist())

    y_true = np.array(y_true, dtype=np.float32)
    y_prob = np.array(y_prob, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    y_true_bin = (y_true >= 0.5).astype(np.int32)

    auc = float(roc_auc_score(y_true_bin, y_prob)) if len(np.unique(y_true_bin)) > 1 else float("nan")
    acc = float(accuracy_score(y_true_bin, y_pred))
    f1 = float(f1_score(y_true_bin, y_pred, zero_division=0))
    rmse = float(np.sqrt(mean_squared_error(y_true_bin, y_prob)))
    mae = float(mean_absolute_error(y_true_bin, y_prob))

    return auc, acc, f1, rmse, mae


def train_one_epoch_custom(net, optimizer, train_loader, model_type, device="cuda"):
    net.train()
    loss_function = torch.nn.BCELoss()
    epoch_losses = []

    for batch in train_loader:
        if model_type.lower() == "kcd":
            if len(batch) < 4:
                raise ValueError(f"KCD 训练时 batch 至少应有4个字段，实际为 {len(batch)}")
            user, item, knowledge, score = batch[:4]

            user = user.to(device)
            item = item.to(device)
            knowledge = knowledge.to(device)
            y = (score > 0.5).float().to(device)

            pred = net(user, item, knowledge)
            pred = _safe_pred_tensor(pred)
            loss = loss_function(pred, y)

        else:
            if len(batch) != 7:
                raise ValueError(f"KCD_Fluency 训练时期望 batch=7，实际为 {len(batch)}")
            user, item, knowledge, score, opp, acc_prior, stability = batch

            user = user.to(device)
            item = item.to(device)
            knowledge = knowledge.to(device)
            opp = opp.to(device)
            acc_prior = acc_prior.to(device)
            stability = stability.to(device)
            y = (score > 0.5).float().to(device)

            pred, aux_loss = call_fluency_net(
                net, user, item, knowledge, opp, acc_prior, stability,
                want_aux=True,
                device=device
            )
            pred = _safe_pred_tensor(pred)
            aux_loss = torch.nan_to_num(aux_loss, nan=0.0, posinf=0.0, neginf=0.0)

            loss = loss_function(pred, y) + aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.detach().cpu().item())

    return float(np.mean(epoch_losses)) if epoch_losses else float("nan")


def train_best_by_valid_auc(
    model,
    model_type,
    train_loader,
    valid_loader,
    epochs,
    lr,
    device,
    pred_threshold=0.5,
):
    net = get_net(model).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    best_auc = -1.0
    best_state = None
    best_epoch = 0
    best_valid_metrics = None

    for ep in range(1, epochs + 1):
        print(f"\n--- Epoch {ep}/{epochs} ---")

        train_loss = train_one_epoch_custom(
            net=net,
            optimizer=optimizer,
            train_loader=train_loader,
            model_type=model_type,
            device=device,
        )
        print(f"[TRAIN] epoch={ep}, loss={train_loss:.6f}")

        auc, acc, f1, rmse, mae = evaluate_metrics(
            model=model,
            loader=valid_loader,
            model_type=model_type,
            device=device,
            pred_threshold=pred_threshold,
        )

        print(
            f"[VALID] epoch={ep}, "
            f"AUC={auc}, ACC={acc:.6f}, F1={f1:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}"
        )

        select_metric = auc if not np.isnan(auc) else -1.0
        if select_metric > best_auc:
            best_auc = select_metric
            best_epoch = ep
            best_state = copy.deepcopy(net.state_dict())
            best_valid_metrics = (auc, acc, f1, rmse, mae)

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_state is None:
        raise RuntimeError("best_state 为空，训练失败")

    net.load_state_dict(best_state)

    print(
        f"\nBEST EPOCH(by VALID AUC) = {best_epoch}, "
        f"BEST_VALID_AUC={best_valid_metrics[0]}, "
        f"ACC={best_valid_metrics[1]:.6f}, "
        f"F1={best_valid_metrics[2]:.6f}, "
        f"RMSE={best_valid_metrics[3]:.6f}, "
        f"MAE={best_valid_metrics[4]:.6f}"
    )

    return best_epoch, best_valid_metrics


# =====================================================
# 7. 单个模型 + 单个数据集运行
# =====================================================
def run_one_dataset(
    dataset_name,
    model_type="kcd_fluency",
    device="cuda",
    epochs=10,
    lr=1e-3,
    batch_size=64,
    pred_threshold=0.5,
    lambda_kcd=1.0,
    lambda_opp=0.1,
    lambda_acc=0.05,
    lambda_sta=0.05,
    flu_latent_dim=16,
    dropout=0.5,
):
    data_dir, train_ds, valid_ds, test_ds, meta = load_dataset_bundle(dataset_name)

    print("\n" + "=" * 90)
    print(f"开始数据集: {dataset_name} | 模型: {model_type}")
    print("=" * 90)
    print(meta)

    train_loader, valid_loader, test_loader = build_loaders_from_datasets(
        train_ds, valid_ds, test_ds, batch_size=batch_size
    )

    model, model_name, user_n, item_n, knowledge_n = build_model(
        model_type=model_type,
        data_dir=data_dir,
        meta=meta,
        lambda_kcd=lambda_kcd,
        lambda_opp=lambda_opp,
        lambda_acc=lambda_acc,
        lambda_sta=lambda_sta,
        flu_latent_dim=flu_latent_dim,
        dropout=dropout,
    )

    best_epoch, valid_metrics = train_best_by_valid_auc(
        model=model,
        model_type=model_type,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        pred_threshold=pred_threshold,
    )

    test_auc, test_acc, test_f1, test_rmse, test_mae = evaluate_metrics(
        model=model,
        loader=test_loader,
        model_type=model_type,
        device=device,
        pred_threshold=pred_threshold,
    )

    print(
        f"TEST {dataset_name} | {model_name}: "
        f"AUC={test_auc}, ACC={test_acc:.6f}, F1={test_f1:.6f}, "
        f"RMSE={test_rmse:.6f}, MAE={test_mae:.6f}"
    )

    return {
        "dataset": dataset_name,
        "model": model_name,
        "select_by": "valid_auc",
        "BEST_EPOCH": best_epoch,
        "VALID_AUC": valid_metrics[0],
        "VALID_ACC": valid_metrics[1],
        "VALID_F1": valid_metrics[2],
        "VALID_RMSE": valid_metrics[3],
        "VALID_MAE": valid_metrics[4],
        "TEST_AUC": test_auc,
        "TEST_ACC": test_acc,
        "TEST_F1": test_f1,
        "TEST_RMSE": test_rmse,
        "TEST_MAE": test_mae,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "user_n": user_n,
        "item_n": item_n,
        "knowledge_n": knowledge_n,
        "lambda_kcd": lambda_kcd,
        "lambda_opp": lambda_opp,
        "lambda_acc": lambda_acc,
        "lambda_sta": lambda_sta,
        "flu_latent_dim": flu_latent_dim,
        "dropout": dropout,
    }


# =====================================================
# 8. 汇总与打印
# =====================================================
def safe_float(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def build_summary_df(full_df):
    summary_rows = []

    datasets = full_df["dataset"].dropna().unique().tolist()
    for ds in datasets:
        sub = full_df[full_df["dataset"] == ds].copy()

        kcd_row = sub[sub["model"] == "KCD"]
        flu_row = sub[sub["model"] == "KCD_Fluency"]

        if len(kcd_row) == 0 or len(flu_row) == 0:
            continue

        kcd_row = kcd_row.iloc[0]
        flu_row = flu_row.iloc[0]

        row = {
            "dataset": ds,

            "KCD_epoch": kcd_row["BEST_EPOCH"],
            "KCD_sel": kcd_row["select_by"],
            "KCD_AUC": kcd_row["TEST_AUC"],
            "KCD_ACC": kcd_row["TEST_ACC"],
            "KCD_F1": kcd_row["TEST_F1"],
            "KCD_RMSE": kcd_row["TEST_RMSE"],
            "KCD_MAE": kcd_row["TEST_MAE"],

            "KCD_Fluency_epoch": flu_row["BEST_EPOCH"],
            "KCD_Fluency_sel": flu_row["select_by"],
            "KCD_Fluency_AUC": flu_row["TEST_AUC"],
            "KCD_Fluency_ACC": flu_row["TEST_ACC"],
            "KCD_Fluency_F1": flu_row["TEST_F1"],
            "KCD_Fluency_RMSE": flu_row["TEST_RMSE"],
            "KCD_Fluency_MAE": flu_row["TEST_MAE"],
        }

        row["Delta_AUC"] = safe_float(row["KCD_Fluency_AUC"]) - safe_float(row["KCD_AUC"])
        row["Delta_ACC"] = safe_float(row["KCD_Fluency_ACC"]) - safe_float(row["KCD_ACC"])
        row["Delta_F1"] = safe_float(row["KCD_Fluency_F1"]) - safe_float(row["KCD_F1"])
        row["Delta_RMSE"] = safe_float(row["KCD_Fluency_RMSE"]) - safe_float(row["KCD_RMSE"])
        row["Delta_MAE"] = safe_float(row["KCD_Fluency_MAE"]) - safe_float(row["KCD_MAE"])

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def print_comparison_summary(summary_df):
    print("\n" + "=" * 90)
    print("Comparison Summary")
    print("=" * 90)

    if len(summary_df) == 0:
        print("没有可对比结果。")
        return

    for _, row in summary_df.iterrows():
        print(f"\n{row['dataset']}")

        print(
            f"  KCD          -> epoch={row['KCD_epoch']}, "
            f"sel={row['KCD_sel']}, "
            f"AUC={safe_float(row['KCD_AUC']):.6f}, "
            f"ACC={safe_float(row['KCD_ACC']):.6f}, "
            f"F1={safe_float(row['KCD_F1']):.6f}, "
            f"RMSE={safe_float(row['KCD_RMSE']):.6f}, "
            f"MAE={safe_float(row['KCD_MAE']):.6f}"
        )

        print(
            f"  KCD_Fluency  -> epoch={row['KCD_Fluency_epoch']}, "
            f"sel={row['KCD_Fluency_sel']}, "
            f"AUC={safe_float(row['KCD_Fluency_AUC']):.6f}, "
            f"ACC={safe_float(row['KCD_Fluency_ACC']):.6f}, "
            f"F1={safe_float(row['KCD_Fluency_F1']):.6f}, "
            f"RMSE={safe_float(row['KCD_Fluency_RMSE']):.6f}, "
            f"MAE={safe_float(row['KCD_Fluency_MAE']):.6f}"
        )

        print(
            f"  Delta        -> "
            f"AUC={safe_float(row['Delta_AUC']):+.6f}, "
            f"ACC={safe_float(row['Delta_ACC']):+.6f}, "
            f"F1={safe_float(row['Delta_F1']):+.6f}, "
            f"RMSE={safe_float(row['Delta_RMSE']):+.6f}, "
            f"MAE={safe_float(row['Delta_MAE']):+.6f}"
        )


# =====================================================
# 9. 主程序
# =====================================================
if __name__ == "__main__":
    set_seed(2025)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device = {device}")

    dataset_names = [
        "a0910",
        "junyi",
        "Assist17",
    ]

    model_types = [
        "kcd",
        "kcd_fluency",
    ]

    epochs = 10
    lr = 0.001
    batch_size = 32
    pred_threshold = 0.5

    lambda_kcd = 1.0
    lambda_opp = 0.1
    lambda_acc = 0.05
    lambda_sta = 0.05
    flu_latent_dim = 16
    dropout = 0.5

    all_results = []

    for dataset_name in dataset_names:
        for model_type in model_types:
            try:
                result = run_one_dataset(
                    dataset_name=dataset_name,
                    model_type=model_type,
                    device=device,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    pred_threshold=pred_threshold,
                    lambda_kcd=lambda_kcd,
                    lambda_opp=lambda_opp,
                    lambda_acc=lambda_acc,
                    lambda_sta=lambda_sta,
                    flu_latent_dim=flu_latent_dim,
                    dropout=dropout,
                )
                all_results.append(result)

            except Exception as e:
                print(f"\n数据集 {dataset_name} / 模型 {model_type} 运行失败: {type(e).__name__}: {e}")
                all_results.append({
                    "dataset": dataset_name,
                    "model": "KCD" if model_type == "kcd" else "KCD_Fluency",
                    "select_by": "valid_auc",
                    "BEST_EPOCH": None,
                    "VALID_AUC": None,
                    "VALID_ACC": None,
                    "VALID_F1": None,
                    "VALID_RMSE": None,
                    "VALID_MAE": None,
                    "TEST_AUC": None,
                    "TEST_ACC": None,
                    "TEST_F1": None,
                    "TEST_RMSE": None,
                    "TEST_MAE": None,
                    "epochs": epochs,
                    "lr": lr,
                    "batch_size": batch_size,
                    "user_n": None,
                    "item_n": None,
                    "knowledge_n": None,
                    "lambda_kcd": lambda_kcd,
                    "lambda_opp": lambda_opp,
                    "lambda_acc": lambda_acc,
                    "lambda_sta": lambda_sta,
                    "flu_latent_dim": flu_latent_dim,
                    "dropout": dropout,
                    "error": f"{type(e).__name__}: {e}",
                })

    full_df = pd.DataFrame(all_results)
    full_csv = OUT_DIR / "compare_kcd_fluency_auc_full.csv"
    full_df.to_csv(full_csv, index=False, encoding="utf-8-sig")

    summary_df = build_summary_df(full_df)
    summary_csv = OUT_DIR / "compare_kcd_fluency_auc_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"\nfull 结果已保存至: {full_csv}")
    print(f"summary 结果已保存至: {summary_csv}")

    print_comparison_summary(summary_df)

    print("\n程序结束。")