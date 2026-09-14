# coding: utf-8
import os
import sys
import random
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


from models.mirt.MIRT_Fluency2 import MIRTFluency


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Assist09MIRTFluencyDataset(Dataset):
    def __init__(
        self,
        csv_path,
        user_map=None,
        item_map=None,
        user_oov_id=None,
        user_col="user_id",
        item_col="problem_id",
        label_col="correct",
        opportunity_col="opportunity",
        rt_col="ms_first_response",
        attempt_col="attempt_count",
        hint_col="hint_count",
    ):
        df = pd.read_csv(csv_path, encoding="utf-8")

        required = [
            user_col,
            item_col,
            label_col,
            opportunity_col,
            rt_col,
            attempt_col,
            hint_col,
        ]

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} 缺少字段: {missing}")

        df = df.dropna(subset=[user_col, item_col, label_col]).copy()

        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        df = df.dropna(subset=[label_col]).copy()
        df[label_col] = df[label_col].astype(int)

        for col in [opportunity_col, rt_col, attempt_col, hint_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        is_train = user_map is None or item_map is None

        def build_map(values):
            uniq = pd.unique(values)
            return {v: i for i, v in enumerate(uniq)}

        if is_train:
            user_map = build_map(df[user_col].values)
            item_map = build_map(df[item_col].values)

        self.user_map = user_map
        self.item_map = item_map

        if user_oov_id is None:
            user_oov_id = len(user_map)
        self.user_oov_id = user_oov_id

        if not is_train:
            before = len(df)
            df = df[df[item_col].isin(item_map)].copy()
            after = len(df)
            print(f"[{os.path.basename(csv_path)}] 过滤未知 item: {before} -> {after}")

        self.users = np.array(
            [user_map.get(v, self.user_oov_id) for v in df[user_col].values],
            dtype=np.int64,
        )

        self.items = np.array(
            [item_map[v] for v in df[item_col].values],
            dtype=np.int64,
        )

        self.y = df[label_col].values.astype(np.float32)
        self.opp = df[opportunity_col].values.astype(np.float32)
        self.rt = df[rt_col].values.astype(np.float32)
        self.attempt = df[attempt_col].values.astype(np.float32)
        self.hint = df[hint_col].values.astype(np.float32)

        self.user_num = len(user_map) + 1
        self.item_num = len(item_map)

        print(
            f"[Dataset] {os.path.basename(csv_path)} "
            f"samples={len(self.y)}, users={self.user_num}, items={self.item_num}"
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
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
    model.irt_net.eval()

    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc=desc, leave=False):
        user_id, item_id, y, opp, rt, attempt, hint = batch

        pred = model.irt_net(
            user_id.to(device),
            item_id.to(device),
            opp.to(device),
            rt.to(device),
            attempt.to(device),
            hint.to(device),
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
    model.irt_net.train()

    losses = []
    ly_list = []
    lopp_list = []
    lrt_list = []
    latt_list = []
    lhint_list = []

    for batch in tqdm(loader, desc=f"Train Epoch {epoch}", leave=False):
        user_id, item_id, y, opp, rt, attempt, hint = batch

        y = y.to(device).float()

        out = model.irt_net(
            user_id.to(device),
            item_id.to(device),
            opp.to(device),
            rt.to(device),
            attempt.to(device),
            hint.to(device),
            return_aux=True,
        )

        ly = model.loss_y(out["p"], y)
        lopp = model.loss_opp(out["opp_hat"], out["opp_tgt"])
        lrt = model.loss_rt(out["rt_hat"], out["rt_tgt"])
        latt = model.loss_att(out["att_hat"], out["att_tgt"])
        lhint = model.loss_hint(out["hint_hat"], out["hint_tgt"])

        loss = (
            ly
            + model.lambda_opp * lopp
            + model.lambda_rt * lrt
            + model.lambda_att * latt
            + model.lambda_hint * lhint
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        ly_list.append(ly.item())
        lopp_list.append(lopp.item())
        lrt_list.append(lrt.item())
        latt_list.append(latt.item())
        lhint_list.append(lhint.item())

    return {
        "loss": float(np.mean(losses)),
        "y": float(np.mean(ly_list)),
        "opp": float(np.mean(lopp_list)),
        "rt": float(np.mean(lrt_list)),
        "att": float(np.mean(latt_list)),
        "hint": float(np.mean(lhint_list)),
    }



def main():
    root = get_project_root()
    train_path = os.path.join(root, "data", "assist09", "splits", "assist09_train.csv")
    test_path = os.path.join(root, "data", "assist09", "splits", "assist09_test.csv")
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    import time
    log_dir = os.path.join(output_dir, "MIRT_vs_Fluency_" + time.strftime("%Y-%m-%d-%H-%M"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "comparison.log")
    flog = open(log_file, "w", encoding="utf-8")
    def log(s): print(s); flog.write(str(s) + "\n"); flog.flush()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    train_ds = Assist09MIRTFluencyDataset(train_path)
    test_ds = Assist09MIRTFluencyDataset(test_path,
        user_map=train_ds.user_map, item_map=train_ds.item_map,
        user_oov_id=train_ds.user_oov_id)

    lr_list = [0.001, 0.002, 0.005]
    epochs = 30
    configs = [
        ("MIRT", lambda: MIRTFluency(
            user_num=train_ds.user_num, item_num=train_ds.item_num, latent_dim=16,
            dropout=0.5, use_fluency=False)),
        ("MIRT+Fluency", lambda: MIRTFluency(
            user_num=train_ds.user_num, item_num=train_ds.item_num, latent_dim=16,
            dropout=0.5, use_fluency=True)),
    ]
    results = {}

    for lr in lr_list:
        for name, build_fn in configs:
            set_seed(42)
            log(f"\n===== Training {name} lr={lr} =====")
            train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

            model = build_fn()
            model.irt_net = model.irt_net.to(device)
            optimizer = torch.optim.Adam(model.irt_net.parameters(), lr=lr)
            bce_loss = nn.BCELoss()
            best_auc = -1.0; best_metrics = None
            best_path = os.path.join(log_dir, f"{name.replace('+','_')}_lr{lr}_best.pt")

            for epoch in range(epochs):
                model.irt_net.train()
                losses = []
                for batch in train_loader:
                    user_id, item_id, y, opp, rt, attempt, hint = batch
                    y = y.to(device).float()

                    if name == "MIRT":
                        pred = model.irt_net(
                            user_id.to(device), item_id.to(device),
                            opp.to(device), rt.to(device), attempt.to(device), hint.to(device))
                        loss = bce_loss(pred, y)
                    else:
                        out = model.irt_net(
                            user_id.to(device), item_id.to(device),
                            opp.to(device), rt.to(device), attempt.to(device), hint.to(device),
                            return_aux=True)
                        ly = bce_loss(out["p"], y)
                        lopp = model.loss_opp(out["opp_hat"], out["opp_tgt"])
                        lrt = model.loss_rt(out["rt_hat"], out["rt_tgt"])
                        latt = model.loss_att(out["att_hat"], out["att_tgt"])
                        lhint = model.loss_hint(out["hint_hat"], out["hint_tgt"])
                        loss = ly + model.lambda_opp * lopp + model.lambda_rt * lrt + model.lambda_att * latt + model.lambda_hint * lhint

                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    losses.append(loss.item())

                # Evaluate
                model.irt_net.eval()
                y_true, y_pred = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        user_id, item_id, y, opp, rt, attempt, hint = batch
                        pred = model.irt_net(
                            user_id.to(device), item_id.to(device),
                            opp.to(device), rt.to(device), attempt.to(device), hint.to(device))
                        y_pred.extend(pred.view(-1).cpu().tolist())
                        y_true.extend(y.tolist())

                y_true = np.array(y_true); y_pred = np.array(y_pred); y_hat = (y_pred >= 0.5).astype(int)
                try: auc = roc_auc_score(y_true, y_pred)
                except ValueError: auc = float("nan")
                acc = accuracy_score(y_true, y_hat)
                f1 = f1_score(y_true, y_hat, zero_division=0)
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

                log(f"  [{name} lr={lr} Epoch {epoch:02d}] loss={float(np.mean(losses)):.6f}  auc={auc:.6f}  acc={acc:.6f}  rmse={rmse:.6f}  f1={f1:.6f}")
                if not np.isnan(auc) and auc > best_auc:
                    best_auc = auc; best_metrics = {"auc": auc, "acc": acc, "rmse": rmse, "f1": f1}
                    torch.save(model.irt_net.state_dict(), best_path)

            results[(name, lr)] = best_metrics
            log(f"  [{name} lr={lr}] BEST: auc={best_metrics['auc']:.6f} acc={best_metrics['acc']:.6f} rmse={best_metrics['rmse']:.6f} f1={best_metrics['f1']:.6f}")

    flog.close()

    sep = "=" * 70
    print("\n" + sep)
    print(" MIRT vs MIRT+Fluency COMPARISON SUMMARY")
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

    rows = [{"Model": n, "LR": lr, **results[(n, lr)]} for lr in lr_list for n, _ in configs if (n, lr) in results and results[(n, lr)]]
    csv_path = os.path.join(log_dir, "mirt_comparison.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

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
        chart_path = os.path.join(log_dir, "mirt_comparison.png")
        plt.tight_layout(); plt.savefig(chart_path, dpi=150); plt.close()
        print(f"Chart saved to {chart_path}")
    except ImportError:
        print("matplotlib not available, skipping chart")

if __name__ == "__main__":
    main()
