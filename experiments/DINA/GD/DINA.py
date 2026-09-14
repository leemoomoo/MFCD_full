# coding: utf-8
# fixed GDDINA runner with AUC / ACC / F1 / RMSE

import ast
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from EduCDM import GDDINA
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error

logging.getLogger().setLevel(logging.INFO)

# =========================
# 1. 读取数据
# =========================
train_data = pd.read_csv("../../../data/a0910/train.csv")
valid_data = pd.read_csv("../../../data/a0910/valid.csv")
test_data = pd.read_csv("../../../data/a0910/test.csv")
item_data = pd.read_csv("../../../data/a0910/item.csv")


# =========================
# 2. knowledge_code -> multi-hot
# =========================
def parse_code(x):
    if isinstance(x, list):
        return x
    return list(ast.literal_eval(str(x)))


codes = item_data["knowledge_code"].apply(parse_code)
knowledge_num = max(max(c) if len(c) > 0 else 0 for c in codes)
print("knowledge_num:", knowledge_num)


def code2vector(x):
    vector = [0] * knowledge_num
    for k in parse_code(x):
        k = int(k)
        if 1 <= k <= knowledge_num:
            vector[k - 1] = 1
    return vector


item_data["knowledge"] = item_data["knowledge_code"].apply(code2vector)
item_data.drop(columns=["knowledge_code"], inplace=True)


# =========================
# 3. merge item knowledge
# =========================
train_data = pd.merge(train_data, item_data, on="item_id", how="inner")
valid_data = pd.merge(valid_data, item_data, on="item_id", how="inner")
test_data = pd.merge(test_data, item_data, on="item_id", how="inner")


# =========================
# 4. 统一映射 user/item id
# =========================
all_users = pd.concat([
    train_data["user_id"],
    valid_data["user_id"],
    test_data["user_id"]
]).unique()

all_items = pd.concat([
    train_data["item_id"],
    valid_data["item_id"],
    test_data["item_id"]
]).unique()

user_map = {u: i for i, u in enumerate(all_users)}
item_map = {q: i for i, q in enumerate(all_items)}

for df in [train_data, valid_data, test_data]:
    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)

student_num = len(user_map)
item_num = len(item_map)

print("students:", student_num)
print("items:", item_num)


# =========================
# 5. score 清理
# =========================
for df in [train_data, valid_data, test_data]:
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df.dropna(subset=["score"], inplace=True)
    df["score"] = df["score"].astype(np.float32)


# =========================
# 6. DataLoader
# =========================
batch_size = 32

def build_loader(df, shuffle=False):
    user = torch.tensor(df["user_id"].values, dtype=torch.int64)
    item = torch.tensor(df["item_id"].values, dtype=torch.int64)
    score = torch.tensor(df["score"].values, dtype=torch.float32)
    knowledge = torch.tensor(
        np.asarray(df["knowledge"].tolist(), dtype=np.float32),
        dtype=torch.float32
    )

    dataset = TensorDataset(user, item, knowledge, score)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


train = build_loader(train_data, shuffle=True)
valid = build_loader(valid_data, shuffle=False)
test = build_loader(test_data, shuffle=False)


# =========================
# 7. 训练模型
# =========================
cdm = GDDINA(student_num, item_num, knowledge_num)

cdm.train(train, valid, epoch=2)
cdm.save("gddina.params")


# =========================
# 8. 自定义评估
# =========================
def get_predict_module(model):
    candidate_names = [
        "net",
        "model",
        "dina_net",
        "gddina_net",
        "_net",
        "_model"
    ]

    for name in candidate_names:
        if hasattr(model, name):
            module = getattr(model, name)
            if hasattr(module, "eval") and callable(module):
                return module

    for name in dir(model):
        if name.startswith("__"):
            continue
        try:
            obj = getattr(model, name)
        except Exception:
            continue
        if hasattr(obj, "eval") and callable(obj):
            return obj

    raise AttributeError(
        "没有在 GDDINA 对象中找到可调用的 torch 模型。请打印 dir(cdm) 查看内部属性。"
    )


def evaluate_metrics(model, loader, threshold=0.5):
    predict_module = get_predict_module(model)
    predict_module.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():
        for user, item, knowledge, score in loader:
            user = user.long()
            item = item.long()
            knowledge = knowledge.float()

            pred = predict_module(user, item, knowledge)
            prob = pred.detach().cpu().numpy().reshape(-1)

            y_prob.extend(prob.tolist())
            y_true.extend(score.cpu().numpy().reshape(-1).tolist())

    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= threshold).astype(np.int32)

    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true, y_prob)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rmse = np.sqrt(mean_squared_error(y_true, y_prob))

    print("\nPrediction diagnostics:")
    print("y_true positive rate:", float(np.mean(y_true)))
    print("y_pred positive rate:", float(np.mean(y_pred)))
    print("y_prob min/max/mean:", float(np.min(y_prob)), float(np.max(y_prob)), float(np.mean(y_prob)))

    return auc, acc, f1, rmse


# =========================
# 9. 测试
# =========================
cdm.load("gddina.params")

auc, acc, f1, rmse = evaluate_metrics(cdm, test)

print("\nResult")
print("AUC : %.6f" % auc if not np.isnan(auc) else "AUC : nan")
print("ACC : %.6f" % acc)
print("F1  : %.6f" % f1)
print("RMSE: %.6f" % rmse)