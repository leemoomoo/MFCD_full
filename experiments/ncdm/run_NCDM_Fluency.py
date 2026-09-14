# coding: utf-8
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
from tqdm import tqdm


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


PROJECT_ROOT = get_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ncdm.NCDM_Fluency import NCDMFluency


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Assist09FluencyDataset(Dataset):
    def __init__(
        self,
        csv_path,
        user_map=None,
        item_map=None,
        skill_map=None,
        user_oov_id=None,
        user_col="user_id",
        item_col="problem_id",
        skill_col="skill_id",
        label_col="correct",
        opportunity_col="opportunity",
        rt_col="ms_first_response",
        attempt_col="attempt_count",
        hint_col="hint_count",
        rt_cap_ms=120000.0,
    ):
        df = pd.read_csv(csv_path, encoding="utf-8")

        required = [
            user_col,
            item_col,
            skill_col,
            label_col,
            opportunity_col,
            rt_col,
            attempt_col,
            hint_col,
        ]

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} 缺少字段: {missing}")

        df = df.dropna(subset=[user_col, item_col, skill_col, label_col]).copy()

        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        df[skill_col] = pd.to_numeric(df[skill_col], errors="coerce")
        df = df.dropna(subset=[label_col, skill_col]).copy()

        df[label_col] = df[label_col].astype(int)
        df[skill_col] = df[skill_col].astype(int)

        for col in [opportunity_col, rt_col, attempt_col, hint_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        is_train = user_map is None or item_map is None or skill_map is None

        def build_map(values):
            uniq = pd.unique(values)
            return {v: i for i, v in enumerate(uniq)}

        if is_train:
            user_map = build_map(df[user_col].values)
            item_map = build_map(df[item_col].values)
            skill_map = build_map(df[skill_col].values)

        self.user_map = user_map
        self.item_map = item_map
        self.skill_map = skill_map

        if user_oov_id is None:
            user_oov_id = len(user_map)
        self.user_oov_id = user_oov_id

        if not is_train:
            before = len(df)
            df = df[df[item_col].isin(item_map)]
            df = df[df[skill_col].isin(skill_map)]
            after = len(df)
            print(f"[{os.path.basename(csv_path)}] 过滤未知 item/skill: {before} -> {after}")

        df = df.reset_index(drop=True)

        # acc_prior: 当前学生在当前知识点之前的历史正确率。
        # 第一次出现没有历史，默认 0.5。
        group_keys = [user_col, skill_col]
        prior_sum = df.groupby(group_keys)[label_col].cumsum() - df[label_col]
        prior_cnt = df.groupby(group_keys).cumcount()

        df["_acc_prior"] = np.where(
            prior_cnt.values > 0,
            prior_sum.values / np.maximum(prior_cnt.values, 1),
            0.5,
        ).astype(np.float32)

        # stability: 用响应时间、额外尝试次数、提示次数构造一个 0~0.25 的“不稳定度”代理。
        # 你的模型内部会做：f_sta = 1 - clamp(4 * stability, 0, 1)
        rt = np.clip(df[rt_col].values.astype(np.float32), 0.0, float(rt_cap_ms))
        rt_norm = np.log1p(rt) / np.log1p(float(rt_cap_ms))

        att_extra = np.clip(df[attempt_col].values.astype(np.float32) - 1.0, 0.0, None)
        att_norm = np.log1p(att_extra) / np.log1p(10.0)

        hint = np.clip(df[hint_col].values.astype(np.float32), 0.0, None)
        hint_norm = np.log1p(hint) / np.log1p(10.0)

        instability = (rt_norm + att_norm + hint_norm) / 3.0
        df["_stability"] = (0.25 * np.clip(instability, 0.0, 1.0)).astype(np.float32)

        self.users = np.array(
            [user_map.get(v, self.user_oov_id) for v in df[user_col].values],
            dtype=np.int64,
        )

        self.items = np.array(
            [item_map[v] for v in df[item_col].values],
            dtype=np.int64,
        )

        self.skills = np.array(
            [skill_map[v] for v in df[skill_col].values],
            dtype=np.int64,
        )

        self.y = df[label_col].values.astype(np.float32)
        self.opp = df[opportunity_col].values.astype(np.float32)
        self.acc_prior = df["_acc_prior"].values.astype(np.float32)
        self.stability = df["_stability"].values.astype(np.float32)

        self.knowledge_n = len(skill_map)
        self.exer_n = len(item_map)
        self.student_n = len(user_map) + 1

        print(
            f"[Dataset] {os.path.basename(csv_path)} "
            f"samples={len(self.y)}, students={self.student_n}, "
            f"items={self.exer_n}, skills={self.knowledge_n}"
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        user_id = torch.tensor(self.users[idx], dtype=torch.long)
        item_id = torch.tensor(self.items[idx], dtype=torch.long)

        knowledge_emb = torch.zeros(self.knowledge_n, dtype=torch.float32)
        knowledge_emb[int(self.skills[idx])] = 1.0

        return (
            user_id,
            item_id,
            knowledge_emb,
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.opp[idx], dtype=torch.float32),
            torch.tensor(self.acc_prior[idx], dtype=torch.float32),
            torch.tensor(self.stability[idx], dtype=torch.float32),
        )


def safe_auc(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_pred)


@torch.no_grad()
def evaluate(model, loader, device, desc="Evaluating"):
    model.net.eval()

    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc=desc, leave=False):
        user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = batch

        pred = model.net(
            user_id.to(device),
            item_id.to(device),
            knowledge_emb.to(device),
            opp.to(device),
            acc_prior.to(device),
            stability.to(device),
            return_aux=False,
        )

        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_hat = (y_pred >= 0.5).astype(int)

    return {
        "auc": safe_auc(y_true, y_pred),
        "acc": accuracy_score(y_true, y_hat),
        "f1": f1_score(y_true, y_hat, zero_division=0),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def fmt_metrics(m):
    return (
        f"auc={m['auc']:.6f} "
        f"acc={m['acc']:.6f} "
        f"f1={m['f1']:.6f} "
        f"rmse={m['rmse']:.6f}"
    )


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.net.train()
    losses = []
    ly_list = []
    lopp_list = []
    lacc_list = []
    lsta_list = []

    for batch in tqdm(loader, desc=f"Train Epoch {epoch}", leave=False):
        user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = batch

        y = y.to(device).float()

        out = model.net(
            user_id.to(device),
            item_id.to(device),
            knowledge_emb.to(device),
            opp.to(device),
            acc_prior.to(device),
            stability.to(device),
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

    return {
        "loss": float(np.mean(losses)),
        "y": float(np.mean(ly_list)),
        "opp": float(np.mean(lopp_list)),
        "acc": float(np.mean(lacc_list)),
        "sta": float(np.mean(lsta_list)),
    }


def main():
    root = get_project_root()

    train_path = os.path.join(root, "data", "assist09", "splits", "assist09_train.csv")
    test_path = os.path.join(root, "data", "assist09", "splits", "assist09_test.csv")

    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds = Assist09FluencyDataset(train_path)

    test_ds = Assist09FluencyDataset(
        test_path,
        user_map=train_ds.user_map,
        item_map=train_ds.item_map,
        skill_map=train_ds.skill_map,
        user_oov_id=train_ds.user_oov_id,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=512,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )

    model = NCDMFluency(
        knowledge_n=train_ds.knowledge_n,
        exer_n=train_ds.exer_n,
        student_n=train_ds.student_n,
        lr=0.002,
        fluency_lambda=0.5,
        dropout=0.5,
    )

    model.net = model.net.to(device)
    optimizer = optim.Adam(model.net.parameters(), lr=model.lr)

    epochs = 50
    best_auc = -1.0
    best_epoch = -1
    best_metrics = None

    save_dir = os.path.join(root, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_ncdm_fluency_assist09.pt")

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            desc=f"Eval Epoch {epoch}",
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"loss={train_metrics['loss']:.6f} "
            f"(y={train_metrics['y']:.6f}, "
            f"opp={train_metrics['opp']:.6f}, "
            f"acc={train_metrics['acc']:.6f}, "
            f"sta={train_metrics['sta']:.6f}) "
            f"| Test: {fmt_metrics(test_metrics)}"
        )

        auc = test_metrics["auc"]
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_metrics = test_metrics
            model.save(save_path)

    print()

    if best_metrics is not None:
        print(f"[Best Test] epoch={best_epoch:02d} {fmt_metrics(best_metrics)}")
        print(f"Best model saved to: {save_path}")
    else:
        print("[Best Test] 没有得到有效 AUC，请检查测试集标签是否只有单一类别。")


if __name__ == "__main__":
    main()