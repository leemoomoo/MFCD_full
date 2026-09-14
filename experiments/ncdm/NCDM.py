# coding: utf-8
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error

from models.ncdm.NCDM import NCDM


def get_project_root():
    # .../experiments/ncdm/run_NCDM_on_fluency_dataset.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # reproducible (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Assist09NCDMDataset(Dataset):
    """
    读取 Fluency 数据集（包含额外行为列），但只返回原始 NCDM 需要的4个：
      user_id, item_id, knowledge_emb(one-hot), y
    """
    def __init__(self, csv_path, user_map=None, item_map=None, skill_map=None, user_oov_id=None):
        df = pd.read_csv(csv_path, encoding="utf-8")

        required = ["user_id", "problem_id", "skill_id", "correct"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")

        # label
        df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
        df = df.dropna(subset=["correct"])
        df["correct"] = df["correct"].astype(int)

        # ids normalize to int (避免 train/test 类型不一致)
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
        df["problem_id"] = pd.to_numeric(df["problem_id"], errors="coerce")
        df["skill_id"] = pd.to_numeric(df["skill_id"], errors="coerce")
        df = df.dropna(subset=["user_id", "problem_id", "skill_id"])
        df["user_id"] = df["user_id"].astype(int)
        df["problem_id"] = df["problem_id"].astype(int)
        df["skill_id"] = df["skill_id"].astype(int)

        is_train = (user_map is None or item_map is None or skill_map is None)

        def build_map(series):
            uniq = pd.unique(series.values)
            return {v: i for i, v in enumerate(uniq)}

        if is_train:
            user_map = build_map(df["user_id"])
            item_map = build_map(df["problem_id"])
            skill_map = build_map(df["skill_id"])

        self.user_map = user_map
        self.item_map = item_map
        self.skill_map = skill_map

        # OOV user（按学生切分时 test 全是新学生）
        if user_oov_id is None:
            user_oov_id = len(user_map)
        self.user_oov_id = user_oov_id

        # test: item/skill 必须在 train 出现，否则 embedding 越界
        if not is_train:
            before = len(df)
            df = df[df["problem_id"].isin(item_map)]
            df = df[df["skill_id"].isin(skill_map)]
            after = len(df)
            print(f"[{os.path.basename(csv_path)}] filtered unseen item/skill: {before} -> {after}")

        # map ids
        self.users = np.array([user_map.get(v, self.user_oov_id) for v in df["user_id"].values], dtype=np.int64)
        self.items = np.array([item_map[v] for v in df["problem_id"].values], dtype=np.int64)
        self.skills = np.array([skill_map[v] for v in df["skill_id"].values], dtype=np.int64)

        self.y = df["correct"].values.astype(np.float32)
        self.knowledge_n = len(skill_map)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        u = torch.tensor(self.users[idx], dtype=torch.long)
        i = torch.tensor(self.items[idx], dtype=torch.long)

        k = torch.zeros(self.knowledge_n, dtype=torch.float32)
        s = int(self.skills[idx])
        k[s] = 1.0

        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return u, i, k, y


def _safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_pred))
    except Exception:
        return float("nan")


def fmt(m: dict) -> str:
    return f"auc={m['auc']:.6f} acc={m['acc']:.6f} f1={m['f1']:.6f} rmse={m['rmse']:.6f}"


@torch.no_grad()
def evaluate_4metrics(model: NCDM, loader: DataLoader, device: str, silence: bool, desc: str):
    model.ncdm_net = model.ncdm_net.to(device)
    model.ncdm_net.eval()

    y_true, y_pred = [], []

    for batch in tqdm(loader, desc=desc, disable=silence, leave=False):
        user_id, item_id, knowledge_emb, y = batch
        user_id = user_id.to(device)
        item_id = item_id.to(device)
        knowledge_emb = knowledge_emb.to(device)
        y = y.to(device)

        pred = model.ncdm_net(user_id, item_id, knowledge_emb)
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        y_true.extend(y.detach().cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_hat = (y_pred >= 0.5).astype(np.int64)

    return {
        "auc": _safe_auc(y_true, y_pred),
        "acc": float(accuracy_score(y_true, y_hat)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def train_one_epoch(model: NCDM, loader: DataLoader, optimizer: optim.Optimizer, device: str,
                    silence: bool, epoch_i: int) -> float:
    model.ncdm_net = model.ncdm_net.to(device)
    model.ncdm_net.train()

    loss_fn = nn.BCELoss()
    losses = []

    for batch in tqdm(loader, desc=f"Train Epoch {epoch_i:02d}", disable=silence, leave=False):
        user_id, item_id, knowledge_emb, y = batch
        user_id = user_id.to(device)
        item_id = item_id.to(device)
        knowledge_emb = knowledge_emb.to(device)
        y = y.to(device).float()

        pred = model.ncdm_net(user_id, item_id, knowledge_emb)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def main():
    root = get_project_root()

    train_path = os.path.join(root, "data", "assist09", "splits", "assist09_train.csv")
    test_path = os.path.join(root, "data", "assist09", "splits", "assist09_test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test not found: {test_path}")

    # config（其余不变）
    batch_size = 512
    epochs = 10
    lr = 0.002
    seed = 42
    silence = False  # 你想关闭 tqdm/进度就改 True

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    train_ds = Assist09NCDMDataset(train_path)
    student_n = len(train_ds.user_map) + 1   # +1 for OOV user
    user_oov_id = student_n - 1

    test_ds = Assist09NCDMDataset(
        test_path,
        user_map=train_ds.user_map,
        item_map=train_ds.item_map,
        skill_map=train_ds.skill_map,
        user_oov_id=user_oov_id
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    knowledge_n = train_ds.knowledge_n
    exer_n = len(train_ds.item_map)

    print("=== Data Info ===")
    print("train:", len(train_ds), "test:", len(test_ds))
    print("knowledge_n:", knowledge_n, "exer_n:", exer_n, "student_n:", student_n)
    print("device:", device)

    # model (原始 NCDM)
    model = NCDM(knowledge_n=knowledge_n, exer_n=exer_n, student_n=student_n)
    optimizer = optim.Adam(model.ncdm_net.parameters(), lr=lr)

    # ===== train loop with per-epoch metrics =====
    best_ep = -1
    best_test = None
    best_auc = -1.0

    for ep in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device=device, silence=silence, epoch_i=ep)

        tr = evaluate_4metrics(model, train_loader, device=device, silence=silence, desc=f"Eval Train {ep:02d}")
        te = evaluate_4metrics(model, test_loader, device=device, silence=silence, desc=f"Eval Test  {ep:02d}")

        print(f"[Epoch {ep:02d}] loss={loss:.6f} | Train: {fmt(tr)} | Test: {fmt(te)}")

        if (not np.isnan(te["auc"])) and (te["auc"] > best_auc):
            best_auc = te["auc"]
            best_ep = ep
            best_test = te

    # final test
    te_final = evaluate_4metrics(model, test_loader, device=device, silence=silence, desc="Final Test Eval")
    print(f"[Final Test] {fmt(te_final)}")

    if best_test is not None:
        print(f"[Best Test] epoch={best_ep:02d} {fmt(best_test)}")
    else:
        print("[Best Test] not available (AUC is NaN)")

    # save ckpt (仍然保存)
    out_dir = os.path.join(root, "outputs", "snapshots")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "ncdm_assist09.pt")
    model.save(ckpt_path)
    print("Saved to:", ckpt_path)


if __name__ == "__main__":
    main()
