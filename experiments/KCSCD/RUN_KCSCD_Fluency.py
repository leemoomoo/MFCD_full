# coding: utf-8
import os
import sys
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
from tqdm import tqdm


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


PROJECT_ROOT = get_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from KCSCD_Fluency2 import Net


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Assist09KCSCDFluencyDataset(Dataset):
    def __init__(
        self,
        csv_path,
        user_map=None,
        item_map=None,
        skill_map=None,
        user_oov_id=None,
    ):
        df = pd.read_csv(csv_path, encoding="utf-8")

        required = [
            "user_id",
            "problem_id",
            "skill_id",
            "correct",
            "opportunity",
            "ms_first_response",
            "attempt_count",
            "hint_count",
        ]

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} 缺少字段: {missing}")

        df = df.dropna(subset=["user_id", "problem_id", "skill_id", "correct"]).copy()

        df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
        df["skill_id"] = pd.to_numeric(df["skill_id"], errors="coerce")
        df = df.dropna(subset=["correct", "skill_id"]).copy()

        df["correct"] = df["correct"].astype(int)
        df["skill_id"] = df["skill_id"].astype(int)

        for col in ["opportunity", "ms_first_response", "attempt_count", "hint_count"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        is_train = user_map is None or item_map is None or skill_map is None

        def build_map(values):
            uniq = pd.unique(values)
            return {v: i for i, v in enumerate(uniq)}

        if is_train:
            user_map = build_map(df["user_id"].values)
            item_map = build_map(df["problem_id"].values)
            skill_map = build_map(df["skill_id"].values)

        self.user_map = user_map
        self.item_map = item_map
        self.skill_map = skill_map

        if user_oov_id is None:
            user_oov_id = len(user_map)
        self.user_oov_id = user_oov_id

        if not is_train:
            before = len(df)
            df = df[df["problem_id"].isin(item_map)].copy()
            df = df[df["skill_id"].isin(skill_map)].copy()
            after = len(df)
            print(f"[{os.path.basename(csv_path)}] 过滤未知 item/skill: {before} -> {after}")

        self.users = np.array(
            [user_map.get(v, self.user_oov_id) for v in df["user_id"].values],
            dtype=np.int64,
        )
        self.items = np.array(
            [item_map[v] for v in df["problem_id"].values],
            dtype=np.int64,
        )
        self.skills = np.array(
            [skill_map[v] for v in df["skill_id"].values],
            dtype=np.int64,
        )

        self.y = df["correct"].values.astype(np.float32)
        self.opp = df["opportunity"].values.astype(np.float32)
        self.rt = df["ms_first_response"].values.astype(np.float32)
        self.attempt = df["attempt_count"].values.astype(np.float32)
        self.hint = df["hint_count"].values.astype(np.float32)

        self.n_student = len(user_map) + 1
        self.n_exercise = len(item_map)
        self.n_skill = len(skill_map)

        print(
            f"[Dataset] {os.path.basename(csv_path)} "
            f"samples={len(self.y)}, students={self.n_student}, "
            f"items={self.n_exercise}, skills={self.n_skill}"
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        stu_id = torch.tensor(self.users[idx], dtype=torch.long)
        exer_id = torch.tensor(self.items[idx], dtype=torch.long)

        qe = torch.zeros(self.n_skill, dtype=torch.float32)
        qe[int(self.skills[idx])] = 1.0

        return (
            stu_id,
            exer_id,
            qe,
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.opp[idx], dtype=torch.float32),
            torch.tensor(self.rt[idx], dtype=torch.float32),
            torch.tensor(self.attempt[idx], dtype=torch.float32),
            torch.tensor(self.hint[idx], dtype=torch.float32),
        )


def safe_auc(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_pred)


@torch.no_grad()
def evaluate(model, loader, device, desc="Evaluating"):
    model.eval()

    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc=desc, leave=False):
        stu_id, exer_id, qe, y, opp, rt, attempt, hint = batch

        output, _, _ = model(
            stu_id.to(device),
            exer_id.to(device),
            qe.to(device),
            y.to(device),
            opportunity=opp.to(device),
            ms_first_response=rt.to(device),
            attempt_count=attempt.to(device),
            hint_count=hint.to(device),
        )

        pred = output.view(-1)

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


def train_one_epoch(model, loader, optimizer, bce_loss, device, epoch, cl_weight=1.0):
    model.train()
    if hasattr(model, "start_new_epoch"):
        model.start_new_epoch()

    losses = []
    bce_losses = []
    cl_losses = []

    for batch in tqdm(loader, desc=f"Train Epoch {epoch}", leave=False):
        stu_id, exer_id, qe, y, opp, rt, attempt, hint = batch

        y = y.to(device).float()

        output, _, cl_loss = model(
            stu_id.to(device),
            exer_id.to(device),
            qe.to(device),
            y,
            opportunity=opp.to(device),
            ms_first_response=rt.to(device),
            attempt_count=attempt.to(device),
            hint_count=hint.to(device),
        )

        pred = output.view(-1)
        loss_bce = bce_loss(pred, y)
        loss = loss_bce + cl_weight * cl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if hasattr(model, "apply_clipper"):
            model.apply_clipper()

        losses.append(loss.item())
        bce_losses.append(loss_bce.item())
        cl_losses.append(cl_loss.item())

    return {
        "loss": float(np.mean(losses)),
        "bce": float(np.mean(bce_losses)),
        "cl": float(np.mean(cl_losses)),
    }



def main():
    root = get_project_root()
    train_path = os.path.join(root, "data", "assist09", "splits", "assist09_train.csv")
    test_path = os.path.join(root, "data", "assist09", "splits", "assist09_test.csv")
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    import time
    log_dir = os.path.join(output_dir, "KCSCD_vs_Fluency_" + time.strftime("%Y-%m-%d-%H-%M"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "comparison.log")
    flog = open(log_file, "w", encoding="utf-8")
    def log(s): print(s); flog.write(str(s) + "\n"); flog.flush()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    train_ds = Assist09KCSCDFluencyDataset(train_path)
    test_ds = Assist09KCSCDFluencyDataset(test_path,
        user_map=train_ds.user_map, item_map=train_ds.item_map,
        skill_map=train_ds.skill_map, user_oov_id=train_ds.user_oov_id)

    base_args = dict(
        n_skill=train_ds.n_skill, n_exercise=train_ds.n_exercise, n_student=train_ds.n_student,
        prednet_len1=512, prednet_len2=256, dropout=0.5, flu_latent_dim=8,
        lambda_opp=0.005, lambda_rt=0.005, lambda_att=0.002, lambda_hint=0.002,
        lambda_consistency=0.002, gamma_init=0.05, delta_scale=0.5, gate_scale=0.2,
        warmup_epochs=3, rt_cap_ms=120000.0, gate_alpha=0.5, gate_hidden_ratio=0.5,
    )

    lr_list = [0.001, 0.002, 0.005]
    epochs = 50
    configs = [
        ("KCSCD",       lambda args: Net(args, use_fluency=False)),
        ("KCSCD+Fluency", lambda args: Net(args, use_fluency=True)),
    ]
    results = {}

    for lr in lr_list:
        for name, build_fn in configs:
            set_seed(42)
            log(f"\n===== Training {name} lr={lr} =====")
            train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

            args = SimpleNamespace(**base_args)
            model = build_fn(args).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            bce_loss = nn.BCELoss()
            best_auc = -1.0; best_metrics = None
            best_path = os.path.join(log_dir, f"{name.replace('+','_')}_lr{lr}_best.pt")

            for epoch in range(epochs):
                model.train()
                losses = []
                for batch in train_loader:
                    stu_id, exer_id, qe, y, opp, rt, attempt, hint = batch
                    y = y.to(device).float()

                    if name == "KCSCD":
                        output, _, cl_loss = model(
                            stu_id.to(device), exer_id.to(device), qe.to(device), y)
                    else:
                        output, _, cl_loss = model(
                            stu_id.to(device), exer_id.to(device), qe.to(device), y,
                            opportunity=opp.to(device), ms_first_response=rt.to(device),
                            attempt_count=attempt.to(device), hint_count=hint.to(device))

                    pred = output.view(-1)
                    loss_bce = bce_loss(pred, y)
                    loss = loss_bce + 1.0 * cl_loss
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    if hasattr(model, "apply_clipper"): model.apply_clipper()
                    losses.append(loss.item())

                model.eval()
                y_true, y_pred = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        stu_id, exer_id, qe, y, opp, rt, attempt, hint = batch
                        if name == "KCSCD":
                            output, _, _ = model(
                                stu_id.to(device), exer_id.to(device), qe.to(device), y.to(device))
                        else:
                            output, _, _ = model(
                                stu_id.to(device), exer_id.to(device), qe.to(device), y.to(device),
                                opportunity=opp.to(device), ms_first_response=rt.to(device),
                                attempt_count=attempt.to(device), hint_count=hint.to(device))
                        y_pred.extend(output.view(-1).cpu().tolist())
                        y_true.extend(y.tolist())

                y_true_np = np.array(y_true); y_pred_np = np.array(y_pred); y_hat = (y_pred_np >= 0.5).astype(int)
                auc = safe_auc(y_true_np, y_pred_np)
                acc = accuracy_score(y_true_np, y_hat)
                f1 = f1_score(y_true_np, y_hat, zero_division=0)
                rmse = float(np.sqrt(mean_squared_error(y_true_np, y_pred_np)))

                log(f"  [{name} lr={lr} Epoch {epoch:02d}] loss={float(np.mean(losses)):.6f}  auc={auc:.6f}  acc={acc:.6f}  rmse={rmse:.6f}  f1={f1:.6f}")
                if not np.isnan(auc) and auc > best_auc:
                    best_auc = auc; best_metrics = {"auc": auc, "acc": acc, "rmse": rmse, "f1": f1}
                    torch.save(model.state_dict(), best_path)

            results[(name, lr)] = best_metrics
            log(f"  [{name} lr={lr}] BEST: auc={best_metrics['auc']:.6f} acc={best_metrics['acc']:.6f} rmse={best_metrics['rmse']:.6f} f1={best_metrics['f1']:.6f}")

    flog.close()

    # Summary
    sep = "=" * 70
    print("\n" + sep)
    print(" KCSCD vs KCSCD+Fluency COMPARISON SUMMARY")
    print(sep)
    print(f"{'LR':<12} {'Model':<20} {'AUC':<10} {'ACC':<10} {'F1':<10} {'RMSE':<10}")
    print("-" * 70)
    for lr in lr_list:
        for name, _ in configs:
            r = results.get((name, lr))
            if r: print(f"{lr:<12} {name:<20} {r['auc']:<10.6f} {r['acc']:<10.6f} {r['f1']:<10.6f} {r['rmse']:<10.6f}")
    print(sep)
    for name, _ in configs:
        valid = {lr: results[(name, lr)] for lr in lr_list if (name, lr) in results and results[(name, lr)]}
        if valid:
            bl = max(valid, key=lambda l: valid[l]["auc"])
            r = valid[bl]
            print(f"Best {name}: lr={bl}, AUC={r['auc']:.6f}, ACC={r['acc']:.6f}, F1={r['f1']:.6f}, RMSE={r['rmse']:.6f}")

    # Save CSV
    rows = [{"Model": n, "LR": lr, **results[(n, lr)]} for lr in lr_list for n, _ in configs if (n, lr) in results and results[(n, lr)]]
    csv_path = os.path.join(log_dir, "kcscd_comparison.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Chart
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(lr_list)); width = 0.35
        for i, metric in enumerate(["auc", "rmse"]):
            ax = axes[i]
            for j, (name, _) in enumerate(configs):
                vals = [results[(name, lr)][metric] if (name, lr) in results and results[(name, lr)] else 0 for lr in lr_list]
                ax.bar(x + j * width, vals, width, label=name)
            ax.set_xlabel("Learning Rate"); ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()} Comparison")
            ax.set_xticks(x + width / 2); ax.set_xticklabels([str(lr) for lr in lr_list])
            ax.legend()
        chart_path = os.path.join(log_dir, "kcscd_comparison.png")
        plt.tight_layout(); plt.savefig(chart_path, dpi=150); plt.close()
        print(f"Chart saved to {chart_path}")
    except ImportError:
        print("matplotlib not available, skipping chart")

if __name__ == "__main__":
    main()
