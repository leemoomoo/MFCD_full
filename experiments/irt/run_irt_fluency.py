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

# ==========================================
# 导入您的两个模型
# ==========================================
from models.irt.IRT import IRT
from models.irt.IRT_fluency import IRTFluency


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Assist09Dataset(Dataset):
    def __init__(
            self,
            csv_path,
            user_map=None,
            item_map=None,
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
            user_col, item_col, skill_col, label_col,
            opportunity_col, rt_col, attempt_col, hint_col,
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

        df = df.reset_index(drop=True)

        group_keys = [user_col, skill_col]
        prior_sum = df.groupby(group_keys)[label_col].cumsum() - df[label_col]
        prior_cnt = df.groupby(group_keys).cumcount()

        df["_acc_prior"] = np.where(
            prior_cnt.values > 0,
            prior_sum.values / np.maximum(prior_cnt.values, 1),
            0.5,
        ).astype(np.float32)

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

        self.y = df[label_col].values.astype(np.float32)
        self.opp = df[opportunity_col].values.astype(np.float32)
        self.acc_prior = df["_acc_prior"].values.astype(np.float32)
        self.stability = df["_stability"].values.astype(np.float32)

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
            torch.tensor(self.acc_prior[idx], dtype=torch.float32),
            torch.tensor(self.stability[idx], dtype=torch.float32),
        )


def safe_auc(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_pred)


def fmt_metrics(m):
    return (
        f"auc={m['auc']:.6f} "
        f"acc={m['acc']:.6f} "
        f"f1={m['f1']:.6f} "
        f"rmse={m['rmse']:.6f}"
    )


# ==========================================
# 核心修复点：智能提取真实的 PyTorch Module
# ==========================================
def get_nn_model(wrapper_obj):
    """
    自适应获取底层的 PyTorch nn.Module 模型。
    通过动态属性探测，解决不同模型代码结构不一致的问题。
    """
    # 1. 如果它本身就是 nn.Module，直接返回
    if isinstance(wrapper_obj, torch.nn.Module):
        return wrapper_obj

    # 2. 检查常见的属性命名习惯
    for attr_name in ['net', 'model', 'irt_model', 'nn_model', 'network']:
        if hasattr(wrapper_obj, attr_name):
            attr = getattr(wrapper_obj, attr_name)
            if isinstance(attr, torch.nn.Module):
                return attr

    # 3. 终极兜底：遍历该对象的所有属性，揪出那个是 nn.Module 的对象
    for attr_name in dir(wrapper_obj):
        # 排除 Python 的魔法方法（如 __init__, __class__ 等）
        if not attr_name.startswith('__'):
            try:
                attr = getattr(wrapper_obj, attr_name)
                if isinstance(attr, torch.nn.Module):
                    return attr
            except Exception:
                pass

    raise ValueError(f"在 {type(wrapper_obj)} 中无法找到继承自 torch.nn.Module 的属性，请检查您的底层模型代码实现。")


def forward_model(model_name, wrapper_obj, batch, device):
    user_id, item_id, y, opp, acc_prior, stability = batch
    nn_model = get_nn_model(wrapper_obj)

    if model_name == "IRT":
        # 传统 IRT 一般只需要 user_id 和 item_id
        pred = nn_model(
            user_id.to(device),
            item_id.to(device)
        )
    elif model_name == "IRTFluency":
        # IRTFluency 需要所有扩展特征
        pred = nn_model(
            user_id.to(device),
            item_id.to(device),
            opp.to(device),
            acc_prior.to(device),
            stability.to(device),
        )
    else:
        raise ValueError(f"未知的模型类型: {model_name}")

    return pred, y.to(device).float()


@torch.no_grad()
def evaluate(model_name, wrapper_obj, loader, device, desc="Evaluating"):
    nn_model = get_nn_model(wrapper_obj)
    nn_model.eval()

    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc=desc, leave=False):
        pred, y = forward_model(model_name, wrapper_obj, batch, device)

        y_true.extend(y.cpu().numpy().tolist())
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


def train_one_epoch(model_name, wrapper_obj, loader, optimizer, loss_fn, device, epoch):
    nn_model = get_nn_model(wrapper_obj)
    nn_model.train()
    losses = []

    for batch in tqdm(loader, desc=f"Train Epoch {epoch}", leave=False):
        pred, y = forward_model(model_name, wrapper_obj, batch, device)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


def train_and_eval_model(model_name, wrapper_obj, train_loader, test_loader, device, save_dir, epochs=50):
    print(f"\n{'=' * 40}")
    print(f" 开始训练模型: {model_name} ")
    print(f"{'=' * 40}")

    # 提取真正的网络结构并推入设备（GPU/CPU）
    nn_model = get_nn_model(wrapper_obj)
    nn_model = nn_model.to(device)

    optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()

    best_auc = -1.0
    best_epoch = -1
    best_metrics = None
    save_path = os.path.join(save_dir, f"best_{model_name.lower()}_assist09.pt")

    for epoch in range(epochs):
        loss = train_one_epoch(
            model_name=model_name,
            wrapper_obj=wrapper_obj,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
        )

        test_metrics = evaluate(
            model_name=model_name,
            wrapper_obj=wrapper_obj,
            loader=test_loader,
            device=device,
            desc=f"Eval {model_name} Epoch {epoch}",
        )

        print(f"[{model_name} Epoch {epoch:02d}] loss={loss:.6f} | Test: {fmt_metrics(test_metrics)}")

        auc = test_metrics["auc"]
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_metrics = test_metrics

            # 自适应保存模型权重：若外层封装了 save 方法则调用，否则用 torch 原生保存机制
            if hasattr(wrapper_obj, "save"):
                wrapper_obj.save(save_path)
            else:
                torch.save(nn_model.state_dict(), save_path)

    if best_metrics is not None:
        print(f"\n[Best Test - {model_name}] epoch={best_epoch:02d} {fmt_metrics(best_metrics)}")
        print(f"[{model_name}] 模型已保存至: {save_path}")
    else:
        print(f"\n[{model_name}] 没有得到有效 AUC，请检查测试集标签是否只有单一类别。")

    return best_metrics


def main():
    root = get_project_root()
    train_path = os.path.join(root, "data", "assist09", "splits", "assist09_train.csv")
    test_path = os.path.join(root, "data", "assist09", "splits", "assist09_test.csv")
    save_dir = os.path.join(root, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 准备数据
    train_ds = Assist09Dataset(train_path)
    test_ds = Assist09Dataset(
        test_path,
        user_map=train_ds.user_map,
        item_map=train_ds.item_map,
        user_oov_id=train_ds.user_oov_id,
    )

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    # 2. 实例化需要对比的模型：IRT 和 IRTFluency
    models_to_compare = {
        "IRT": IRT(
            user_num=train_ds.user_num,
            item_num=train_ds.item_num
        ),
        "IRTFluency": IRTFluency(
            user_num=train_ds.user_num,
            item_num=train_ds.item_num,
            value_range=4.0,
            a_range=1.0
        )
    }

    # 3. 运行对比循环
    comparison_results = {}
    for m_name, m_instance in models_to_compare.items():
        best_res = train_and_eval_model(
            model_name=m_name,
            wrapper_obj=m_instance,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            save_dir=save_dir,
            epochs=50
        )
        comparison_results[m_name] = best_res

    # 4. 打印最终对比总结表
    print("\n\n" + "=" * 50)
    print(" 模型性能对比总结 (Best Test Metrics)")
    print("=" * 50)

    header = f"{'Model':<12} | {'AUC':<8} | {'ACC':<8} | {'F1':<8} | {'RMSE':<8}"
    print(header)
    print("-" * 50)

    for m_name, res in comparison_results.items():
        if res:
            row = f"{m_name:<12} | {res['auc']:.6f} | {res['acc']:.6f} | {res['f1']:.6f} | {res['rmse']:.6f}"
            print(row)
        else:
            print(f"{m_name:<12} | N/A (Failed to get valid metrics)")
    print("=" * 50)


if __name__ == "__main__":
    main()