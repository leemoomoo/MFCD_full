# coding: utf-8
"""
compare_kcscd_fluency.py
-------------------------------------------------------
对比 KCSCD vs KCSCD_Fluency

功能:
1. 统一按 VALID AUC 选择最佳 epoch
2. 在 VALID 上搜索最佳 threshold
3. TEST 上使用该 threshold 评估 AUC / ACC / F1 / RMSE / MAE
4. fluency 特征按 train -> valid -> test 连续历史构造
5. 兼容新的真实 KCSCD_Fluency 主干
-------------------------------------------------------
"""

import copy
import json
import random
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
)

from kcscd_fluency import Net as KCSCDFluencyNet
from model import Net as KCSCDNet


# =========================================================
# 0. seed
# =========================================================
def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 1. paths
# =========================================================
def find_project_root(start: Path) -> Path:
    cur = start.resolve().parent
    for p in [cur] + list(cur.parents):
        if (p / "data").exists():
            return p
    return cur


BASE_DIR = find_project_root(Path(__file__))
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs" / "compare_kcscd_fluency_true"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. generic utils
# =========================================================
def normalize_binary_score(x):
    try:
        x = float(x)
    except Exception:
        return 0.0
    return 1.0 if x >= 0.5 else 0.0


def detect_columns(df: pd.DataFrame, candidates):
    cols_lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def remap_records_columns(df: pd.DataFrame):
    user_col = detect_columns(df, ["user_id", "uid", "student_id", "user", "stu_id"])
    item_col = detect_columns(df, ["item_id", "exer_id", "problem_id", "question_id", "item", "qid"])
    score_col = detect_columns(df, ["score", "label", "correct", "is_correct", "answer", "y"])

    if user_col is None or item_col is None or score_col is None:
        if df.shape[1] >= 3:
            tmp = df.iloc[:, :3].copy()
            tmp.columns = ["user_id", "item_id", "score"]
            df2 = tmp
        else:
            raise ValueError(f"无法识别 user/item/score 列，当前列名: {list(df.columns)}")
    else:
        df2 = df[[user_col, item_col, score_col]].copy()
        df2.columns = ["user_id", "item_id", "score"]

    df2["user_id"] = pd.to_numeric(df2["user_id"], errors="coerce")
    df2["item_id"] = pd.to_numeric(df2["item_id"], errors="coerce")
    df2["score"] = df2["score"].apply(normalize_binary_score)

    df2 = df2.dropna().copy()
    df2["user_id"] = df2["user_id"].astype(int)
    df2["item_id"] = df2["item_id"].astype(int)
    df2["score"] = df2["score"].astype(float)
    return df2.reset_index(drop=True)


def split_records(records: pd.DataFrame, seed: int = 2025):
    stratify_all = records["score"] if records["score"].nunique() > 1 else None

    try:
        train_df, temp_df = train_test_split(
            records,
            test_size=0.3,
            random_state=seed,
            shuffle=True,
            stratify=stratify_all,
        )
    except Exception:
        train_df, temp_df = train_test_split(
            records,
            test_size=0.3,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    stratify_temp = temp_df["score"] if temp_df["score"].nunique() > 1 else None

    try:
        valid_df, test_df = train_test_split(
            temp_df,
            test_size=2 / 3,
            random_state=seed,
            shuffle=True,
            stratify=stratify_temp,
        )
    except Exception:
        valid_df, test_df = train_test_split(
            temp_df,
            test_size=2 / 3,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def read_table_auto(path: Path, header=None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", header=header)
    except Exception:
        return pd.read_csv(path, header=header)


def read_q_matrix_standard(q_path: Path):
    if not q_path.exists():
        raise FileNotFoundError(f"找不到 Q 矩阵文件: {q_path}")

    q_df = read_table_auto(q_path, header=None)
    q_df = q_df.apply(pd.to_numeric, errors="coerce")
    q_df = q_df.dropna(axis=1, how="all")
    q_df = q_df.dropna(axis=0, how="all")
    q_df = q_df.fillna(0)
    return q_df.values.astype(int)


def parse_k_list(s):
    s = str(s).strip()
    s = s.replace("[", "").replace("]", "").replace("{", "").replace("}", "")
    s = s.replace(";", ",").replace("，", ",").replace(" ", ",")
    arr = [x.strip() for x in s.split(",") if x.strip() != ""]
    out = []
    for x in arr:
        try:
            out.append(int(float(x)))
        except Exception:
            pass
    return out


def clip_fluency_semantics(df: pd.DataFrame):
    df = df.copy()
    df["opportunity"] = pd.to_numeric(df["opportunity"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["acc_prior"] = pd.to_numeric(df["acc_prior"], errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
    df["stability"] = pd.to_numeric(df["stability"], errors="coerce").fillna(0.25).clip(lower=0.0)
    return df


# =========================================================
# 3. dataset loading
# =========================================================
def build_data_bundle(dataset_name: str, records: pd.DataFrame, q_mat: np.ndarray, seed: int = 2025):
    train_df, valid_df, test_df = split_records(records, seed=seed)
    return build_data_bundle_from_split(dataset_name, train_df, valid_df, test_df, q_mat)


def build_data_bundle_from_split(
    dataset_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    q_mat: np.ndarray,
    explicit_item_ids=None,
):
    all_df = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)

    user_ids_sorted = sorted(all_df["user_id"].astype(int).unique().tolist())
    item_ids_sorted = sorted(all_df["item_id"].astype(int).unique().tolist())

    user_id_map = {uid: i for i, uid in enumerate(user_ids_sorted)}

    if explicit_item_ids is not None:
        explicit_item_ids = sorted([int(x) for x in explicit_item_ids])
        item_id_map = {iid: i for i, iid in enumerate(explicit_item_ids)}

        missing_items = sorted(set(item_ids_sorted) - set(explicit_item_ids))
        if missing_items:
            raise ValueError(
                f"{dataset_name}: 数据中的 item_id 在 Q 矩阵映射中不存在，示例缺失: {missing_items[:10]}"
            )

        item_n = len(explicit_item_ids)
        q_rows = q_mat.shape[0]
        if q_rows != item_n:
            raise ValueError(
                f"{dataset_name}: Q 矩阵行数与 explicit_item_ids 数量不一致，q_rows={q_rows}, item_n={item_n}"
            )

        q_mat_aligned = q_mat.copy()

    else:
        unique_item_n = len(item_ids_sorted)
        q_rows = q_mat.shape[0]

        if q_rows == unique_item_n:
            item_id_map = {iid: i for i, iid in enumerate(item_ids_sorted)}
            item_n = unique_item_n
            q_mat_aligned = q_mat.copy()
        else:
            i_min = int(all_df["item_id"].min())
            i_max = int(all_df["item_id"].max())
            expected_n = i_max - i_min + 1

            if q_rows < expected_n:
                raise ValueError(
                    f"{dataset_name}: Q 矩阵行数不足，q_rows={q_rows}, expected_items={expected_n}"
                )

            expected_items = list(range(i_min, i_max + 1))
            if item_ids_sorted != expected_items:
                raise ValueError(
                    f"{dataset_name}: item_id 非连续，且未提供 explicit_item_ids，无法安全对齐 Q 矩阵"
                )

            item_id_map = {iid: iid - i_min for iid in expected_items}
            item_n = expected_n
            q_mat_aligned = q_mat[:item_n].copy()

    user_n = len(user_ids_sorted)
    knowledge_n = q_mat_aligned.shape[1]

    meta = {
        "dataset": dataset_name,
        "student_n": user_n,
        "exercise_n": item_n,
        "knowledge_n": knowledge_n,
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_test": len(test_df),
    }

    return {
        "dataset_name": dataset_name,
        "train_df": train_df.reset_index(drop=True),
        "valid_df": valid_df.reset_index(drop=True),
        "test_df": test_df.reset_index(drop=True),
        "q_mat": q_mat_aligned,
        "user_n": user_n,
        "item_n": item_n,
        "knowledge_n": knowledge_n,
        "user_id_map": user_id_map,
        "item_id_map": item_id_map,
        "meta": meta,
    }


def load_assist17(seed=2025):
    dataset_dir = DATA_DIR / "Assist17"
    config_path = dataset_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到 config.json: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    files_cfg = cfg.get("files", {})
    q_name = files_cfg.get("q_matrix") or files_cfg.get("q_mat") or files_cfg.get("Q") or "q_matrix.csv"
    r_name = files_cfg.get("response") or files_cfg.get("responses") or files_cfg.get("record") or "response.csv"

    q_mat = read_q_matrix_standard(dataset_dir / q_name)
    resp_df = read_table_auto(dataset_dir / r_name, header=0)
    records = remap_records_columns(resp_df)

    return build_data_bundle("Assist17", records, q_mat, seed=seed)


def load_a0910(seed=2025):
    dataset_dir = DATA_DIR / "a0910"

    train_df = remap_records_columns(read_table_auto(dataset_dir / "train.csv", header=0))
    valid_df = remap_records_columns(read_table_auto(dataset_dir / "valid.csv", header=0))
    test_df = remap_records_columns(read_table_auto(dataset_dir / "test.csv", header=0))
    item_df = read_table_auto(dataset_dir / "item.csv", header=0)

    if "item_id" not in item_df.columns or "knowledge_code" not in item_df.columns:
        raise ValueError("a0910/item.csv 缺少 item_id 或 knowledge_code 列")

    item_ids = item_df["item_id"].astype(int).values
    all_k = set()
    parsed = []
    for s in item_df["knowledge_code"]:
        ks = parse_k_list(s)
        parsed.append(ks)
        all_k.update(ks)

    if len(all_k) == 0:
        raise ValueError("a0910: knowledge_code 为空")

    k_sorted = sorted(all_k)
    k_to_idx = {k: i for i, k in enumerate(k_sorted)}
    item_sorted = sorted(set(int(x) for x in item_ids))
    item_to_qrow = {item_id: i for i, item_id in enumerate(item_sorted)}

    q_mat = np.zeros((len(item_sorted), len(k_sorted)), dtype=int)
    for i, ks in zip(item_ids, parsed):
        row = item_to_qrow[int(i)]
        for k in ks:
            if k in k_to_idx:
                q_mat[row, k_to_idx[k]] = 1

    return build_data_bundle_from_split(
        "a0910", train_df, valid_df, test_df, q_mat, explicit_item_ids=item_sorted
    )


def load_junyi(seed=2025):
    dataset_dir = DATA_DIR / "junyi"

    train_df = remap_records_columns(read_table_auto(dataset_dir / "train.csv", header=0))
    valid_df = remap_records_columns(read_table_auto(dataset_dir / "valid.csv", header=0))
    test_df = remap_records_columns(read_table_auto(dataset_dir / "test.csv", header=0))
    item_df = read_table_auto(dataset_dir / "items.csv", header=0)

    if "item_id" not in item_df.columns or "knowledge_code" not in item_df.columns:
        raise ValueError("junyi/items.csv 缺少 item_id 或 knowledge_code 列")

    item_ids = item_df["item_id"].astype(int).values
    all_k = set()
    parsed = []
    for s in item_df["knowledge_code"]:
        ks = parse_k_list(s)
        parsed.append(ks)
        all_k.update(ks)

    if len(all_k) == 0:
        raise ValueError("junyi: knowledge_code 为空")

    k_sorted = sorted(all_k)
    k_to_idx = {k: i for i, k in enumerate(k_sorted)}
    item_sorted = sorted(set(int(x) for x in item_ids))
    item_to_qrow = {item_id: i for i, item_id in enumerate(item_sorted)}

    q_mat = np.zeros((len(item_sorted), len(k_sorted)), dtype=int)
    for i, ks in zip(item_ids, parsed):
        row = item_to_qrow[int(i)]
        for k in ks:
            if k in k_to_idx:
                q_mat[row, k_to_idx[k]] = 1

    return build_data_bundle_from_split(
        "junyi", train_df, valid_df, test_df, q_mat, explicit_item_ids=item_sorted
    )


def load_math1(seed=2025):
    dataset_dir = DATA_DIR / "math2015" / "Math1"
    records = remap_records_columns(read_table_auto(dataset_dir / "data.txt", header=None))
    q_mat = read_table_auto(dataset_dir / "q.txt", header=None).values.astype(int)
    return build_data_bundle("Math1", records, q_mat, seed=seed)


def load_math2(seed=2025):
    dataset_dir = DATA_DIR / "math2015" / "Math2"
    records = remap_records_columns(read_table_auto(dataset_dir / "data.txt", header=None))
    q_mat = read_table_auto(dataset_dir / "q.txt", header=None).values.astype(int)
    return build_data_bundle("Math2", records, q_mat, seed=seed)


def load_frcsub(seed=2025):
    dataset_dir = DATA_DIR / "math2015" / "FrcSub"
    records = remap_records_columns(read_table_auto(dataset_dir / "data.txt", header=None))
    q_mat = read_table_auto(dataset_dir / "q.txt", header=None).values.astype(int)
    return build_data_bundle("FrcSub", records, q_mat, seed=seed)


# =========================================================
# 4. fluency features with state propagation
# =========================================================
def init_history_state():
    return {
        "user_k_count": {},
        "user_k_correct": {},
        "user_k_sq": {},
    }


def copy_history_state(state):
    return {
        "user_k_count": copy.deepcopy(state["user_k_count"]),
        "user_k_correct": copy.deepcopy(state["user_k_correct"]),
        "user_k_sq": copy.deepcopy(state["user_k_sq"]),
    }


def build_fluency_features_with_state(
    df: pd.DataFrame,
    item_id_map: dict,
    q_mat: np.ndarray,
    init_state=None,
):
    df = df.copy().reset_index(drop=True)

    if init_state is None:
        state = init_history_state()
    else:
        state = copy_history_state(init_state)

    opp_list = []
    acc_list = []
    sta_list = []

    user_k_count = state["user_k_count"]
    user_k_correct = state["user_k_correct"]
    user_k_sq = state["user_k_sq"]

    for _, row in df.iterrows():
        u = int(row["user_id"])
        item_raw = int(row["item_id"])
        y = float(row["score"])

        if item_raw not in item_id_map:
            raise ValueError(f"item_id {item_raw} not in item_id_map")

        item_idx = item_id_map[item_raw]
        q = q_mat[item_idx]
        ks = np.where(q > 0)[0].tolist()

        if u not in user_k_count:
            user_k_count[u] = {}
            user_k_correct[u] = {}
            user_k_sq[u] = {}

        if len(ks) == 0:
            opp = 0.0
            acc = 0.5
            sta = 0.25
        else:
            counts = [user_k_count[u].get(k, 0.0) for k in ks]
            cors = [user_k_correct[u].get(k, 0.0) for k in ks]
            sqs = [user_k_sq[u].get(k, 0.0) for k in ks]

            opp = float(np.mean(counts)) if len(counts) > 0 else 0.0

            per_k_acc = []
            per_k_var = []
            for c, s1, s2 in zip(counts, cors, sqs):
                if c <= 0:
                    per_k_acc.append(0.5)
                    per_k_var.append(0.25)
                else:
                    p = s1 / c
                    var = max(0.0, min(0.25, (s2 / c) - p * p))
                    per_k_acc.append(float(p))
                    per_k_var.append(float(var))

            acc = float(np.mean(per_k_acc)) if len(per_k_acc) > 0 else 0.5
            sta = float(np.mean(per_k_var)) if len(per_k_var) > 0 else 0.25

        opp_list.append(opp)
        acc_list.append(acc)
        sta_list.append(sta)

        for k in ks:
            user_k_count[u][k] = user_k_count[u].get(k, 0.0) + 1.0
            user_k_correct[u][k] = user_k_correct[u].get(k, 0.0) + y
            user_k_sq[u][k] = user_k_sq[u].get(k, 0.0) + y * y

    df["opportunity"] = np.array(opp_list, dtype=np.float32)
    df["acc_prior"] = np.array(acc_list, dtype=np.float32)
    df["stability"] = np.array(sta_list, dtype=np.float32)
    df = clip_fluency_semantics(df)

    return df, state


# =========================================================
# 5. loaders
# =========================================================
def make_loader(
    df: pd.DataFrame,
    q_mat: np.ndarray,
    user_id_map: dict,
    item_id_map: dict,
    batch_size: int = 64,
    shuffle: bool = True,
    use_fluency: bool = True,
):
    users_raw = df["user_id"].astype(int).tolist()
    items_raw = df["item_id"].astype(int).tolist()

    users = np.array([user_id_map[u] for u in users_raw], dtype=np.int64)
    items = np.array([item_id_map[i] for i in items_raw], dtype=np.int64)
    scores = df["score"].values.astype(np.float32)

    if len(users) == 0:
        raise ValueError("DataLoader 输入为空")
    if items.max() >= len(q_mat):
        raise ValueError(f"item 索引越界: max_item={items.max()}, q_rows={len(q_mat)}")

    knowledge = torch.tensor(q_mat[items], dtype=torch.float32)

    if use_fluency:
        opp = df["opportunity"].values.astype(np.float32)
        acc = df["acc_prior"].values.astype(np.float32)
        sta = df["stability"].values.astype(np.float32)
        dataset = TensorDataset(
            torch.tensor(users, dtype=torch.long),
            torch.tensor(items, dtype=torch.long),
            knowledge,
            torch.tensor(scores, dtype=torch.float32),
            torch.tensor(opp, dtype=torch.float32),
            torch.tensor(acc, dtype=torch.float32),
            torch.tensor(sta, dtype=torch.float32),
        )
    else:
        dataset = TensorDataset(
            torch.tensor(users, dtype=torch.long),
            torch.tensor(items, dtype=torch.long),
            knowledge,
            torch.tensor(scores, dtype=torch.float32),
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# =========================================================
# 6. metrics
# =========================================================
def stabilize_output(output: torch.Tensor) -> torch.Tensor:
    output = torch.nan_to_num(output, nan=0.5, posinf=1.0, neginf=0.0)
    output = torch.clamp(output, min=1e-6, max=1 - 1e-6)
    return output


@torch.no_grad()
def collect_outputs(model, loader, model_type, device="cuda"):
    model.eval()

    y_true, y_prob = [], []

    for batch in loader:
        if model_type == "kcscd":
            stu_ids, exer_ids, knowledge_embs, ys = batch
            stu_ids = stu_ids.to(device)
            exer_ids = exer_ids.to(device)
            knowledge_embs = knowledge_embs.to(device)
            ys = ys.float().to(device)
            answers = ys.long()

            pred, _, _ = model(stu_ids, exer_ids, knowledge_embs, answers)

        else:
            stu_ids, exer_ids, knowledge_embs, ys, opp, acc, sta = batch
            stu_ids = stu_ids.to(device)
            exer_ids = exer_ids.to(device)
            knowledge_embs = knowledge_embs.to(device)
            ys = ys.float().to(device)
            opp = opp.to(device)
            acc = acc.to(device)
            sta = sta.to(device)

            pred, _, _ = model(
                stu_ids,
                exer_ids,
                knowledge_embs,
                ys.long(),
                opportunity=opp,
                acc_prior=acc,
                stability=sta,
            )

        pred = stabilize_output(pred.view(-1))

        y_true.extend(ys.detach().cpu().numpy().tolist())
        y_prob.extend(pred.detach().cpu().numpy().tolist())

    return np.array(y_true), np.array(y_prob)


def calc_metrics_from_probs(y_true_np, y_prob_np, threshold=0.5):
    y_pred_np = (y_prob_np > threshold).astype(np.float32)

    if len(np.unique(y_true_np)) > 1:
        auc = float(roc_auc_score(y_true_np, y_prob_np))
    else:
        auc = float("nan")

    acc = float(accuracy_score(y_true_np, y_pred_np))
    f1 = float(f1_score(y_true_np, y_pred_np, zero_division=0))
    rmse = float(np.sqrt(mean_squared_error(y_true_np, y_prob_np)))
    mae = float(mean_absolute_error(y_true_np, y_prob_np))

    return {
        "AUC": auc,
        "ACC": acc,
        "F1": f1,
        "RMSE": rmse,
        "MAE": mae,
        "threshold": threshold,
    }


def search_best_threshold(y_true_np, y_prob_np):
    best_t = 0.5
    best_f1 = -1.0

    for t in np.arange(0.10, 0.91, 0.02):
        y_pred = (y_prob_np > t).astype(np.float32)
        cur_f1 = f1_score(y_true_np, y_pred, zero_division=0)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_t = float(t)

    return best_t, best_f1


# =========================================================
# 7. train
# =========================================================
def train_one_epoch_kcscd(model, train_loader, optimizer, device="cuda", grad_clip=5.0):
    model.train()
    loss_y = nn.BCELoss()

    total_loss = 0.0
    total_count = 0

    for stu_ids, exer_ids, knowledge_embs, ys in train_loader:
        stu_ids = stu_ids.to(device)
        exer_ids = exer_ids.to(device)
        knowledge_embs = knowledge_embs.to(device)
        ys = ys.float().to(device)
        answers = ys.long()

        optimizer.zero_grad()

        pred, _, cl_loss = model(stu_ids, exer_ids, knowledge_embs, answers)
        pred = stabilize_output(pred.view(-1))
        cl_loss = torch.nan_to_num(cl_loss, nan=0.0, posinf=0.0, neginf=0.0)

        loss = loss_y(pred, ys) + cl_loss

        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: 非法 loss，跳过一个 batch")
            continue

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        if hasattr(model, "apply_clipper"):
            model.apply_clipper()

        bs = ys.size(0)
        total_loss += loss.detach().cpu().item() * bs
        total_count += bs

    return float("nan") if total_count == 0 else total_loss / total_count


def train_one_epoch_fluency(
    model,
    train_loader,
    optimizer,
    device="cuda",
    grad_clip=5.0,
):
    model.train()
    loss_y = nn.BCELoss()

    total_loss = 0.0
    total_count = 0

    for stu_ids, exer_ids, knowledge_embs, ys, opp, acc, sta in train_loader:
        stu_ids = stu_ids.to(device)
        exer_ids = exer_ids.to(device)
        knowledge_embs = knowledge_embs.to(device)
        ys = ys.float().to(device)
        opp = opp.to(device)
        acc = acc.to(device)
        sta = sta.to(device)

        optimizer.zero_grad()

        pred, _, cl_loss = model(
            stu_ids,
            exer_ids,
            knowledge_embs,
            ys.long(),
            opportunity=opp,
            acc_prior=acc,
            stability=sta,
        )

        pred = stabilize_output(pred.view(-1))
        cl_loss = torch.nan_to_num(cl_loss, nan=0.0, posinf=0.0, neginf=0.0)

        loss = loss_y(pred, ys) + cl_loss

        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: 非法 loss，跳过一个 batch")
            continue

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        if hasattr(model, "apply_clipper"):
            model.apply_clipper()

        bs = ys.size(0)
        total_loss += loss.detach().cpu().item() * bs
        total_count += bs

    return float("nan") if total_count == 0 else total_loss / total_count


def train_best_by_valid(
    dataset_name: str,
    model,
    train_loader,
    valid_loader,
    model_type,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc = -1.0
    best_state = None
    best_ep = 0
    best_valid_metrics = None
    best_threshold = 0.5

    for ep in range(1, epochs + 1):
        if hasattr(model, "start_new_epoch"):
            model.start_new_epoch()

        if model_type == "kcscd":
            train_loss = train_one_epoch_kcscd(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
            )
        else:
            train_loss = train_one_epoch_fluency(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
            )

        valid_y, valid_prob = collect_outputs(
            model=model,
            loader=valid_loader,
            model_type=model_type,
            device=device,
        )

        cur_auc_metrics = calc_metrics_from_probs(valid_y, valid_prob, threshold=0.5)
        cur_best_t, _ = search_best_threshold(valid_y, valid_prob)
        cur_best_metrics = calc_metrics_from_probs(valid_y, valid_prob, threshold=cur_best_t)

        print(
            f"[{dataset_name}][{model_type}][Epoch {ep:03d}] "
            f"loss={train_loss:.6f} | "
            f"VALID AUC={cur_auc_metrics['AUC']:.6f}, "
            f"BEST_T={cur_best_t:.2f}, "
            f"ACC={cur_best_metrics['ACC']:.6f}, "
            f"F1={cur_best_metrics['F1']:.6f}, "
            f"RMSE={cur_best_metrics['RMSE']:.6f}, "
            f"MAE={cur_best_metrics['MAE']:.6f}"
        )

        cur_auc = cur_auc_metrics["AUC"]
        if np.isnan(cur_auc):
            cur_auc = -1.0

        if cur_auc > best_auc:
            best_auc = cur_auc
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_valid_metrics = cur_best_metrics
            best_threshold = cur_best_t

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if best_state is None:
        raise RuntimeError("best_state is empty")

    model.load_state_dict(best_state)
    return best_ep, best_valid_metrics, best_threshold, "valid_auc"


# =========================================================
# 8. run one dataset + one model
# =========================================================
def run_one_dataset_model(
    bundle,
    model_type,
    device="cuda",
    epochs=20,
    lr=1e-3,
    batch_size=256,
    weight_decay=0.0,
    dropout=0.5,
    flu_latent_dim=8,
    gamma_init=0.05,
    delta_scale=0.5,
    gate_scale=0.2,
    warmup_epochs=3,
    lambda_opp=0.005,
    lambda_acc=0.005,
    lambda_sta=0.002,
    lambda_consistency=0.002,
):
    dataset_name = bundle["dataset_name"]
    train_df = bundle["train_df"]
    valid_df = bundle["valid_df"]
    test_df = bundle["test_df"]
    q_mat = bundle["q_mat"]
    user_n = bundle["user_n"]
    item_n = bundle["item_n"]
    knowledge_n = bundle["knowledge_n"]
    user_id_map = bundle["user_id_map"]
    item_id_map = bundle["item_id_map"]

    print("\n" + "=" * 90)
    print(f"开始数据集: {dataset_name} | 模型: {model_type}")
    print(bundle["meta"])
    print("=" * 90)

    if model_type == "kcscd_fluency":
        hist_state = init_history_state()

        train_df, hist_state = build_fluency_features_with_state(
            train_df, item_id_map, q_mat, init_state=hist_state
        )
        valid_df, hist_state = build_fluency_features_with_state(
            valid_df, item_id_map, q_mat, init_state=hist_state
        )
        test_df, hist_state = build_fluency_features_with_state(
            test_df, item_id_map, q_mat, init_state=hist_state
        )

        train_loader = make_loader(
            train_df, q_mat, user_id_map, item_id_map,
            batch_size=batch_size, shuffle=True, use_fluency=True
        )
        valid_loader = make_loader(
            valid_df, q_mat, user_id_map, item_id_map,
            batch_size=batch_size, shuffle=False, use_fluency=True
        )
        test_loader = make_loader(
            test_df, q_mat, user_id_map, item_id_map,
            batch_size=batch_size, shuffle=False, use_fluency=True
        )

        args = SimpleNamespace(
            n_student=user_n,
            n_exercise=item_n,
            n_skill=knowledge_n,
            dropout=dropout,
            flu_latent_dim=flu_latent_dim,
            gamma_init=gamma_init,
            delta_scale=delta_scale,
            gate_scale=gate_scale,
            warmup_epochs=warmup_epochs,
            lambda_rt=lambda_opp,
            lambda_hint=lambda_acc,
            lambda_att=lambda_sta,
            lambda_consistency=lambda_consistency,
        )
        model = KCSCDFluencyNet(args).to(device)

    else:
        train_loader = make_loader(
            train_df, q_mat, user_id_map, item_id_map,
            batch_size=batch_size, shuffle=True, use_fluency=False
        )
        valid_loader = make_loader(
            valid_df, q_mat, user_id_map, item_id_map,
            batch_size=batch_size, shuffle=False, use_fluency=False
        )
        test_loader = make_loader(
            test_df, q_mat, user_id_map, item_id_map,
            batch_size=batch_size, shuffle=False, use_fluency=False
        )

        args = SimpleNamespace(
            n_student=user_n,
            n_exercise=item_n,
            n_skill=knowledge_n,
        )
        model = KCSCDNet(args).to(device)

    best_ep, best_valid, best_threshold, selected_by = train_best_by_valid(
        dataset_name=dataset_name,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        model_type=model_type,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    test_y, test_prob = collect_outputs(
        model=model,
        loader=test_loader,
        model_type=model_type,
        device=device,
    )
    test_metrics = calc_metrics_from_probs(test_y, test_prob, threshold=best_threshold)

    ckpt_path = OUT_DIR / f"{model_type}_{dataset_name.lower()}_best_true.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "dataset_meta": bundle["meta"],
            "best_epoch": best_ep,
            "best_valid": best_valid,
            "best_threshold": best_threshold,
            "test_metrics": test_metrics,
            "model_type": model_type,
        },
        ckpt_path,
    )

    print(
        f"[TEST] {dataset_name} | {model_type} | "
        f"BEST_T={best_threshold:.2f} | "
        f"AUC={test_metrics['AUC']:.6f}, ACC={test_metrics['ACC']:.6f}, "
        f"F1={test_metrics['F1']:.6f}, RMSE={test_metrics['RMSE']:.6f}, "
        f"MAE={test_metrics['MAE']:.6f} | "
        f"BEST_EPOCH={best_ep}, SELECTED_BY={selected_by}"
    )

    return {
        "dataset": dataset_name,
        "model": "KCSCD" if model_type == "kcscd" else "KCSCD_Fluency",
        "BEST_EPOCH": best_ep,
        "selected_by": selected_by,
        "BEST_THRESHOLD": best_threshold,
        "VALID_AUC": best_valid["AUC"],
        "VALID_ACC": best_valid["ACC"],
        "VALID_F1": best_valid["F1"],
        "VALID_RMSE": best_valid["RMSE"],
        "VALID_MAE": best_valid["MAE"],
        "TEST_AUC": test_metrics["AUC"],
        "TEST_ACC": test_metrics["ACC"],
        "TEST_F1": test_metrics["F1"],
        "TEST_RMSE": test_metrics["RMSE"],
        "TEST_MAE": test_metrics["MAE"],
        "student_n": user_n,
        "exercise_n": item_n,
        "knowledge_n": knowledge_n,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "flu_latent_dim": flu_latent_dim,
        "gamma_init": gamma_init,
        "delta_scale": delta_scale,
        "gate_scale": gate_scale,
        "warmup_epochs": warmup_epochs,
        "lambda_opp": lambda_opp,
        "lambda_acc": lambda_acc,
        "lambda_sta": lambda_sta,
        "lambda_consistency": lambda_consistency,
        "checkpoint": str(ckpt_path),
        "error": None,
    }


# =========================================================
# 9. summary
# =========================================================
def safe_float(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def build_summary_df(df: pd.DataFrame):
    summary_rows = []
    dataset_order = ["Math1", "Math2", "FrcSub", "Assist17", "a0910", "junyi"]

    for ds in dataset_order:
        sub = df[df["dataset"] == ds].copy()
        base_row = sub[sub["model"] == "KCSCD"]
        flu_row = sub[sub["model"] == "KCSCD_Fluency"]

        if len(base_row) == 0 or len(flu_row) == 0:
            continue

        base_row = base_row.iloc[0]
        flu_row = flu_row.iloc[0]

        row = {
            "dataset": ds,

            "KCSCD_epoch": base_row["BEST_EPOCH"],
            "KCSCD_sel": base_row["selected_by"],
            "KCSCD_threshold": base_row["BEST_THRESHOLD"],
            "KCSCD_AUC": base_row["TEST_AUC"],
            "KCSCD_ACC": base_row["TEST_ACC"],
            "KCSCD_F1": base_row["TEST_F1"],
            "KCSCD_RMSE": base_row["TEST_RMSE"],
            "KCSCD_MAE": base_row["TEST_MAE"],

            "KCSCD_Fluency_epoch": flu_row["BEST_EPOCH"],
            "KCSCD_Fluency_sel": flu_row["selected_by"],
            "KCSCD_Fluency_threshold": flu_row["BEST_THRESHOLD"],
            "KCSCD_Fluency_AUC": flu_row["TEST_AUC"],
            "KCSCD_Fluency_ACC": flu_row["TEST_ACC"],
            "KCSCD_Fluency_F1": flu_row["TEST_F1"],
            "KCSCD_Fluency_RMSE": flu_row["TEST_RMSE"],
            "KCSCD_Fluency_MAE": flu_row["TEST_MAE"],
        }

        row["Delta_AUC"] = safe_float(row["KCSCD_Fluency_AUC"]) - safe_float(row["KCSCD_AUC"])
        row["Delta_ACC"] = safe_float(row["KCSCD_Fluency_ACC"]) - safe_float(row["KCSCD_ACC"])
        row["Delta_F1"] = safe_float(row["KCSCD_Fluency_F1"]) - safe_float(row["KCSCD_F1"])
        row["Delta_RMSE"] = safe_float(row["KCSCD_Fluency_RMSE"]) - safe_float(row["KCSCD_RMSE"])
        row["Delta_MAE"] = safe_float(row["KCSCD_Fluency_MAE"]) - safe_float(row["KCSCD_MAE"])

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def print_comparison_summary(summary_df: pd.DataFrame):
    print("\n" + "=" * 90)
    print("KCSCD vs KCSCD_Fluency Comparison Summary")
    print("=" * 90)

    if len(summary_df) == 0:
        print("没有可对比结果。")
        return

    for _, row in summary_df.iterrows():
        print(f"\n{row['dataset']}")
        print(
            f"  KCSCD          -> epoch={row['KCSCD_epoch']}, "
            f"sel={row['KCSCD_sel']}, "
            f"thr={safe_float(row['KCSCD_threshold']):.2f}, "
            f"AUC={safe_float(row['KCSCD_AUC']):.6f}, "
            f"ACC={safe_float(row['KCSCD_ACC']):.6f}, "
            f"F1={safe_float(row['KCSCD_F1']):.6f}, "
            f"RMSE={safe_float(row['KCSCD_RMSE']):.6f}, "
            f"MAE={safe_float(row['KCSCD_MAE']):.6f}"
        )
        print(
            f"  KCSCD_Fluency  -> epoch={row['KCSCD_Fluency_epoch']}, "
            f"sel={row['KCSCD_Fluency_sel']}, "
            f"thr={safe_float(row['KCSCD_Fluency_threshold']):.2f}, "
            f"AUC={safe_float(row['KCSCD_Fluency_AUC']):.6f}, "
            f"ACC={safe_float(row['KCSCD_Fluency_ACC']):.6f}, "
            f"F1={safe_float(row['KCSCD_Fluency_F1']):.6f}, "
            f"RMSE={safe_float(row['KCSCD_Fluency_RMSE']):.6f}, "
            f"MAE={safe_float(row['KCSCD_Fluency_MAE']):.6f}"
        )
        print(
            f"  Delta          -> "
            f"AUC={safe_float(row['Delta_AUC']):+.6f}, "
            f"ACC={safe_float(row['Delta_ACC']):+.6f}, "
            f"F1={safe_float(row['Delta_F1']):+.6f}, "
            f"RMSE={safe_float(row['Delta_RMSE']):+.6f}, "
            f"MAE={safe_float(row['Delta_MAE']):+.6f}"
        )


# =========================================================
# 10. main
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--flu_latent_dim", type=int, default=8)
    parser.add_argument("--gamma_init", type=float, default=0.05)
    parser.add_argument("--delta_scale", type=float, default=0.5)
    parser.add_argument("--gate_scale", type=float, default=0.2)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--lambda_opp", type=float, default=0.005)
    parser.add_argument("--lambda_acc", type=float, default=0.005)
    parser.add_argument("--lambda_sta", type=float, default=0.002)
    parser.add_argument("--lambda_consistency", type=float, default=0.002)

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"BASE_DIR = {BASE_DIR}")
    print(f"DATA_DIR = {DATA_DIR}")
    print(f"OUT_DIR  = {OUT_DIR}")
    print(f"Device   = {device}")

    dataset_loaders = [
        ("a0910", load_a0910),
        ("Assist17", load_assist17),
        ("junyi", load_junyi),
        ("Math1", load_math1),
        ("Math2", load_math2),
        ("FrcSub", load_frcsub),
    ]

    model_types = ["kcscd", "kcscd_fluency"]
    all_results = []

    for name, loader_fn in dataset_loaders:
        for model_type in model_types:
            try:
                bundle = loader_fn(seed=args.seed)
                result = run_one_dataset_model(
                    bundle=bundle,
                    model_type=model_type,
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    weight_decay=args.weight_decay,
                    dropout=args.dropout,
                    flu_latent_dim=args.flu_latent_dim,
                    gamma_init=args.gamma_init,
                    delta_scale=args.delta_scale,
                    gate_scale=args.gate_scale,
                    warmup_epochs=args.warmup_epochs,
                    lambda_opp=args.lambda_opp,
                    lambda_acc=args.lambda_acc,
                    lambda_sta=args.lambda_sta,
                    lambda_consistency=args.lambda_consistency,
                )
                all_results.append(result)

            except Exception as e:
                print(f"\n数据集 {name} / 模型 {model_type} 运行失败: {type(e).__name__}: {e}")
                all_results.append({
                    "dataset": name,
                    "model": "KCSCD" if model_type == "kcscd" else "KCSCD_Fluency",
                    "BEST_EPOCH": None,
                    "selected_by": None,
                    "BEST_THRESHOLD": None,
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
                    "student_n": None,
                    "exercise_n": None,
                    "knowledge_n": None,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "weight_decay": args.weight_decay,
                    "dropout": args.dropout,
                    "flu_latent_dim": args.flu_latent_dim,
                    "gamma_init": args.gamma_init,
                    "delta_scale": args.delta_scale,
                    "gate_scale": args.gate_scale,
                    "warmup_epochs": args.warmup_epochs,
                    "lambda_opp": args.lambda_opp,
                    "lambda_acc": args.lambda_acc,
                    "lambda_sta": args.lambda_sta,
                    "lambda_consistency": args.lambda_consistency,
                    "checkpoint": None,
                    "error": f"{type(e).__name__}: {e}",
                })

    full_df = pd.DataFrame(all_results)
    full_csv = OUT_DIR / "compare_kcscd_fluency_full_true.csv"
    full_df.to_csv(full_csv, index=False, encoding="utf-8-sig")
    print(f"\n完整结果已保存至: {full_csv}")

    summary_df = build_summary_df(full_df)
    summary_csv = OUT_DIR / "compare_kcscd_fluency_summary_true.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"对比汇总已保存至: {summary_csv}")

    print_comparison_summary(summary_df)
    print("\n程序结束。")