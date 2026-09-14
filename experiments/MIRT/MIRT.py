# coding: utf-8
import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
from EduCDM import MIRT

logging.getLogger().setLevel(logging.INFO)

BATCH_SIZE = 256
SEED = 42
VALID_RATIO = 0.1
LATENT_DIM = 123


def get_project_root():
    # experiments/MIRT/MIRT.py -> project_root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_1d_numpy(x):
    """Series/list/np -> 1D numpy"""
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    elif isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    else:
        x = np.asarray(x)
    return np.asarray(x).reshape(-1)


def transform(u, i, y, batch_size, shuffle=False):
    u = _to_1d_numpy(u).astype(np.int64)
    i = _to_1d_numpy(i).astype(np.int64)
    y = _to_1d_numpy(y).astype(np.float32)

    ds = TensorDataset(torch.from_numpy(u), torch.from_numpy(i), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _norm_id_col(s: pd.Series) -> pd.Series:
    """统一 ID 为去空格字符串，避免 train/test dtype 不一致导致 map 全 NaN"""
    return s.astype(str).str.strip()


def build_id_map(values: pd.Series):
    values = _norm_id_col(values)
    uniq = pd.unique(values.values)
    return {v: i for i, v in enumerate(uniq)}


def split_train_valid(df, valid_ratio=0.1, seed=42):
    idx = np.arange(len(df))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    n_valid = int(len(df) * valid_ratio)
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]

    return df.iloc[train_idx].reset_index(drop=True), df.iloc[valid_idx].reset_index(drop=True)


def remap_train(df, user_map, item_map, name="train"):
    """
    train：严格映射到 train 的 user/item map（不会有 OOV）
    """
    required = ["user_id", "problem_id", "correct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}. Got: {list(df.columns)}")

    df = df.copy()
    df["user_id"] = _norm_id_col(df["user_id"])
    df["problem_id"] = _norm_id_col(df["problem_id"])

    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df.dropna(subset=["correct"])
    df["correct"] = df["correct"].astype(int)

    df["user_id_m"] = df["user_id"].map(user_map)
    df["item_id_m"] = df["problem_id"].map(item_map)

    before = len(df)
    df = df.dropna(subset=["user_id_m", "item_id_m"])
    after = len(df)
    print(f"[{name}] after mapping dropna: {before} -> {after}")

    df["user_id_m"] = df["user_id_m"].astype(int)
    df["item_id_m"] = df["item_id_m"].astype(int)
    return df


def remap_test_with_oov_user(df, user_map, item_map, user_oov_id, name="test"):
    """
    test：item 必须在 train 出现过（过滤 unseen item）
          user 允许全新（unseen user -> OOV）
    """
    required = ["user_id", "problem_id", "correct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}. Got: {list(df.columns)}")

    df = df.copy()
    df["user_id"] = _norm_id_col(df["user_id"])
    df["problem_id"] = _norm_id_col(df["problem_id"])

    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df.dropna(subset=["correct"])
    df["correct"] = df["correct"].astype(int)

    # 过滤未见过的题目
    before = len(df)
    df = df[df["problem_id"].isin(item_map)]
    after = len(df)
    print(f"[filter {name} unseen items] {before} -> {after}")

    # item 映射（必定存在）
    df["item_id_m"] = df["problem_id"].map(item_map)

    # user 映射：未见过 -> OOV
    df["user_id_m"] = df["user_id"].map(user_map)
    miss_user = int(df["user_id_m"].isna().sum())
    print(f"[{name}] unseen users mapped to OOV: {miss_user} / {len(df)}")

    df["user_id_m"] = df["user_id_m"].fillna(user_oov_id).astype(int)

    # item 保险 dropna
    before2 = len(df)
    df = df.dropna(subset=["item_id_m"])
    after2 = len(df)
    print(f"[{name}] after item dropna: {before2} -> {after2}")

    df["item_id_m"] = df["item_id_m"].astype(int)
    return df


def _safe_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except Exception:
        return float("nan")


def _find_inner_net(model, sample_user, sample_item):
    """
    不同 EduCDM 版本字段名不同：自动扫描 model 内所有 torch.nn.Module，
    用 (user,item) 试跑 forward，能跑通且输出长度匹配 batch 的就认为是内部网络。
    """
    candidates = []
    for name, obj in vars(model).items():
        if isinstance(obj, torch.nn.Module):
            candidates.append((name, obj))

    # 常见字段名优先
    preferred = ["model", "mirt", "mirt_model", "_model", "net", "mirt_net"]
    ordered = []
    used = set()
    for p in preferred:
        for n, o in candidates:
            if n == p and n not in used:
                ordered.append((n, o))
                used.add(n)
    for n, o in candidates:
        if n not in used:
            ordered.append((n, o))
            used.add(n)

    for name, net in ordered:
        try:
            net.eval()
            with torch.no_grad():
                out = net(sample_user, sample_item)
            if torch.is_tensor(out):
                out = out.view(-1)
                if out.numel() == sample_user.numel():
                    return name, net
        except Exception:
            continue

    cand_names = [n for n, _ in candidates]
    raise AttributeError(
        f"Cannot locate inner torch.nn.Module for MIRT. "
        f"Found nn.Module fields={cand_names}."
    )


@torch.no_grad()
def evaluate_4metrics(model, loader):
    """
    返回 (auc, acc, f1, rmse)
    """
    it = iter(loader)
    u0, i0, y0 = next(it)
    u0 = u0.view(-1)
    i0 = i0.view(-1)

    _, net = _find_inner_net(model, u0, i0)
    net.eval()

    y_true, y_pred = [], []

    # 第一批
    p0 = net(u0, i0).view(-1)
    y_pred.extend(p0.detach().cpu().numpy().tolist())
    y_true.extend(y0.detach().cpu().numpy().tolist())

    # 后续
    for u, i, y in it:
        u = u.view(-1)
        i = i.view(-1)
        p = net(u, i).view(-1)
        y_pred.extend(p.detach().cpu().numpy().tolist())
        y_true.extend(y.detach().cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_hat = (y_pred >= 0.5).astype(np.int32)

    auc = _safe_auc(y_true, y_pred)
    acc = accuracy_score(y_true, y_hat)
    f1 = f1_score(y_true, y_hat, zero_division=0)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return auc, acc, f1, rmse


def fmt4(m):
    auc, acc, f1, rmse = m
    return f"auc={auc:.6f} acc={acc:.6f} f1={f1:.6f} rmse={rmse:.6f}"


def main():
    set_seed(SEED)

    root = get_project_root()
    train_path = os.path.join(root, "data", "assist09", "splits", "assist09_train.csv")
    test_path = os.path.join(root, "data", "assist09", "splits", "assist09_test.csv")

    train_raw = pd.read_csv(train_path, encoding="utf-8")
    test_raw = pd.read_csv(test_path, encoding="utf-8")

    # train 建 map
    user_map = build_id_map(train_raw["user_id"])
    item_map = build_id_map(train_raw["problem_id"])
    user_oov_id = len(user_map)

    train_df = remap_train(train_raw, user_map, item_map, name="train")
    if len(train_df) == 0:
        print("[FATAL] train is empty after mapping.")
        return

    train_df, valid_df = split_train_valid(train_df, valid_ratio=VALID_RATIO, seed=SEED)

    test_df = remap_test_with_oov_user(
        test_raw, user_map, item_map, user_oov_id=user_oov_id, name="test"
    )
    if len(test_df) == 0:
        print("[WARN] test is empty after filtering/mapping. Skip test metrics.")
        return

    student_n = len(user_map) + 1  # +1 for OOV user
    exer_n = len(item_map)

    print(f"student_n={student_n}, exer_n={exer_n}, latent_dim={LATENT_DIM}")
    print(f"train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    train_loader = transform(train_df["user_id_m"], train_df["item_id_m"], train_df["correct"],
                             batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = transform(valid_df["user_id_m"], valid_df["item_id_m"], valid_df["correct"],
                             batch_size=BATCH_SIZE, shuffle=False)
    test_loader = transform(test_df["user_id_m"], test_df["item_id_m"], test_df["correct"],
                            batch_size=BATCH_SIZE, shuffle=False)

    # 注意：EduCDM.MIRT 不是 torch.nn.Module，没有 .to()
    cdm = MIRT(student_n, exer_n, LATENT_DIM)

    cdm.train(train_loader, valid_loader, epoch=2)
    cdm.save("mirt.params")
    cdm.load("mirt.params")

    va = evaluate_4metrics(cdm, valid_loader)
    te = evaluate_4metrics(cdm, test_loader)

    print(f"[Valid] {fmt4(va)}")
    print(f"[Test ] {fmt4(te)}")


if __name__ == "__main__":
    main()