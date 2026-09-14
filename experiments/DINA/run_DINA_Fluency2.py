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
# ⚠️ 请根据您本地实际的模型文件名修改以下导入路径
# 假设您的模型分别为 DINA 和 DINAChannel
# ==========================================
try:
    from models.dina.DINA import DINA
    from models.dina.DINA_channel import DINAChannel
except ImportError:
    print("⚠️ 警告：找不到 DINA 或 DINAChannel 模型文件，请检查导入路径！")
    # 如果都在同一个文件里，可能长这样：
    # from models.dina.DINA_Fluency2 import DINA, DINAChannel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Assist09DINADataset(Dataset):
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
            df = df[df[item_col].isin(item_map)].copy()
            df = df[df[skill_col].isin(skill_map)].copy()
            after = len(df)
            print(f"[{os.path.basename(csv_path)}] 过滤未知 item/skill: {before} -> {after}")

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
        self.rt = df[rt_col].values.astype(np.float32)
        self.attempt = df[attempt_col].values.astype(np.float32)
        self.hint = df[hint_col].values.astype(np.float32)

        self.user_num = len(user_map) + 1
        self.item_num = len(item_map)
        self.knowledge_n = len(skill_map)

        print(
            f"[Dataset] {os.path.basename(csv_path)} "
            f"samples={len(self.y)}, users={self.user_num}, "
            f"items={self.item_num}, skills={self.knowledge_n}"
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
            torch.tensor(self.rt[idx], dtype=torch.float32),
            torch.tensor(self.attempt[idx], dtype=torch.float32),
            torch.tensor(self.hint[idx], dtype=torch.float32),
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
# 核心组件：自适应模型提取与前向传播
# ==========================================
def get_nn_model(wrapper_obj):
    if isinstance(wrapper_obj, torch.nn.Module):
        return wrapper_obj
    for attr_name in ['dina_net', 'net', 'model', 'nn_model']:
        if hasattr(wrapper_obj, attr_name):
            attr = getattr(wrapper_obj, attr_name)
            if isinstance(attr, torch.nn.Module):
                return attr
    for attr_name in dir(wrapper_obj):
        if not attr_name.startswith('__'):
            try:
                attr = getattr(wrapper_obj, attr_name)
                if isinstance(attr, torch.nn.Module):
                    return attr
            except Exception:
                pass
    raise ValueError(f"无法在 {type(wrapper_obj)} 中找到 PyTorch Module")


def forward_model(model_name, nn_model, batch, device, return_aux=False):
    user_id, item_id, knowledge, y, opp, rt, attempt, hint = batch

    if "channel" in model_name.lower():
        # DINAChannel 需要所有扩展特征
        out = nn_model(
            user_id.to(device),
            item_id.to(device),
            knowledge.to(device),
            opp.to(device),
            rt.to(device),
            attempt.to(device),
            hint.to(device),
            return_aux=return_aux,
        )
    else:
        # 传统 DINA 只需 user, item, knowledge
        try:
            out = nn_model(
                user_id.to(device),
                item_id.to(device),
                knowledge.to(device),
                return_aux=return_aux
            )
        except TypeError:
            # 兼容不支持 return_aux 的原生网络写法
            out = nn_model(
                user_id.to(device),
                item_id.to(device),
                knowledge.to(device)
            )

    return out, y.to(device).float()


@torch.no_grad()
def evaluate(model_name, wrapper_obj, loader, device, desc="Evaluating"):
    nn_model = get_nn_model(wrapper_obj)
    nn_model.eval()

    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc=desc, leave=False):
        out, y = forward_model(model_name, nn_model, batch, device, return_aux=False)

        # 兼容字典输出和张量输出
        pred = out["p"] if isinstance(out, dict) else out

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


def train_one_epoch(model_name, wrapper_obj, loader, optimizer, device, epoch):
    nn_model = get_nn_model(wrapper_obj)
    nn_model.train()
    fallback_loss_fn = torch.nn.BCELoss()
    losses = []

    for batch in tqdm(loader, desc=f"Train Epoch {epoch}", leave=False):
        out, y = forward_model(model_name, nn_model, batch, device, return_aux=True)

        # ===============================================
        # 智能 Loss 计算机制：
        # 如果模型有封装的多通道 loss 函数，就用多通道。否则退化为普通 BCE
        # ===============================================
        if hasattr(wrapper_obj, "loss_y") and isinstance(out, dict) and "opp_hat" in out:
            ly = wrapper_obj.loss_y(out["p"], y)
            lopp = wrapper_obj.loss_opp(out["opp_hat"], out["opp_tgt"])
            lrt = wrapper_obj.loss_rt(out["rt_hat"], out["rt_tgt"])
            latt = wrapper_obj.loss_att(out["att_hat"], out["att_tgt"])
            lhint = wrapper_obj.loss_hint(out["hint_hat"], out["hint_tgt"])

            loss = (
                    ly
                    + getattr(wrapper_obj, 'lambda_opp', 0.1) * lopp
                    + getattr(wrapper_obj, 'lambda_rt', 0.05) * lrt
                    + getattr(wrapper_obj, 'lambda_att', 0.05) * latt
                    + getattr(wrapper_obj, 'lambda_hint', 0.05) * lhint
            )
        else:
            # 传统 DINA 只计算 BCE
            pred = out["p"] if isinstance(out, dict) else out
            loss = fallback_loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return {"loss": float(np.mean(losses))}


def train_and_eval_model(model_name, wrapper_obj, train_loader, test_loader, device, save_dir, epochs=50):
    print(f"\n{'=' * 40}")
    print(f" 开始训练模型: {model_name} ")
    print(f"{'=' * 40}")

    nn_model = get_nn_model(wrapper_obj)
    nn_model = nn_model.to(device)

    # 获取特定模型的初始学习率（如果没有设置，默认0.001）
    lr = getattr(wrapper_obj, "lr", 0.001)
    optimizer = optim.Adam(nn_model.parameters(), lr=lr)

    best_auc = -1.0
    best_epoch = -1
    best_metrics = None
    save_path = os.path.join(save_dir, f"best_{model_name.lower()}_assist09.pt")

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model_name=model_name,
            wrapper_obj=wrapper_obj,
            loader=train_loader,
            optimizer=optimizer,
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

        print(f"[{model_name} Epoch {epoch:02d}] loss={train_metrics['loss']:.6f} | Test: {fmt_metrics(test_metrics)}")

        auc = test_metrics["auc"]
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_metrics = test_metrics

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
    train_ds = Assist09DINADataset(train_path)
    test_ds = Assist09DINADataset(
        test_path,
        user_map=train_ds.user_map,
        item_map=train_ds.item_map,
        skill_map=train_ds.skill_map,
        user_oov_id=train_ds.user_oov_id,
    )

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    # ==========================================
    # 2. 实例化需要对比的模型：DINA 和 DINAChannel
    # 如果您的模型名称是 IDINE，请将字典的 Key 和类实例化改成对应的名称
    # ==========================================
    models_to_compare = {}

    try:
        models_to_compare["DINA"] = DINA(
            user_num=train_ds.user_num,
            item_num=train_ds.item_num,
            hidden_dim=train_ds.knowledge_n,
            # 其他基础参数按需添加
        )
    except NameError:
        print("未找到基础 DINA 模型，跳过实例化...")

    try:
        models_to_compare["DINAChannel"] = DINAChannel(
            user_num=train_ds.user_num,
            item_num=train_ds.item_num,
            hidden_dim=train_ds.knowledge_n,
            ste=False,
            lr=0.001,
            max_slip=0.4,
            max_guess=0.4,
            dropout=0.5,
            fluency_lambda=0.5,
            lambda_opp=0.1,
            lambda_rt=0.05,
            lambda_att=0.05,
            lambda_hint=0.05,
            flu_latent_dim=16,
            gamma_init=1.0,
            rt_cap_ms=120000.0,
        )
    except NameError:
        print("未找到 DINAChannel 模型，跳过实例化...")

    # 如果字典为空，说明导入失败了，需要您检查开头的 import 语句
    if not models_to_compare:
        print("\n❌ 错误：模型实例化失败，请检查脚本顶部 from models.dina... 的导入路径和类名！")
        sys.exit(1)

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
    header = f"{'Model':<15} | {'AUC':<8} | {'ACC':<8} | {'F1':<8} | {'RMSE':<8}"
    print(header)
    print("-" * 50)
    for m_name, res in comparison_results.items():
        if res:
            row = f"{m_name:<15} | {res['auc']:.6f} | {res['acc']:.6f} | {res['f1']:.6f} | {res['rmse']:.6f}"
            print(row)
        else:
            print(f"{m_name:<15} | N/A (Failed)")
    print("=" * 50)


if __name__ == "__main__":
    main()