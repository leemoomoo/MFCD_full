# coding: utf-8
"""
compare_cdnd_baseline.py
--------------------------------
只运行 CDND baseline（已删除 channel 模块）
直接调用 model.train(train_data, test_data=valid, epoch=EPOCHS, ...)
"""

import os
import random
import logging
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import dataloader
from model import cdnd


def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LR = 0.002
EPOCHS = 10

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs", "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "compare_cdnd_baseline.csv")

logging.getLogger().setLevel(logging.INFO)
os.makedirs(os.path.join(THIS_DIR, "params"), exist_ok=True)


class ZeroBasedLoader:
    def __init__(self, base_loader, user_offset: int, item_offset: int):
        self.base_loader = base_loader
        self.user_offset = user_offset
        self.item_offset = item_offset

    def __iter__(self):
        for user_id, item_id, knowledge_emb, y in self.base_loader:
            yield (user_id - self.user_offset,
                   item_id - self.item_offset,
                   knowledge_emb, y)

    def __len__(self):
        return len(self.base_loader)


def infer_ids_and_knowledge_dim(loaders):
    u_min, i_min = 10**18, 10**18
    u_max, i_max = -1, -1
    k_dim = None
    for loader in loaders:
        for user_id, item_id, knowledge_emb, _ in loader:
            user_id = user_id.detach().cpu()
            item_id = item_id.detach().cpu()
            u_min = min(u_min, int(user_id.min().item()))
            u_max = max(u_max, int(user_id.max().item()))
            i_min = min(i_min, int(item_id.min().item()))
            i_max = max(i_max, int(item_id.max().item()))
            if k_dim is None:
                k_dim = int(knowledge_emb.shape[1])
    if k_dim is None:
        raise RuntimeError("Failed to infer knowledge_dim")
    return u_min, u_max, i_min, i_max, k_dim


def evaluate_metrics_from_model(model, loader, device=DEVICE):
    auc, acc, rmse, f1 = model.eval(loader, device=device)
    net = getattr(model, "acd_net", None)
    item_n = None
    if net is not None and hasattr(net, "exercise_k") and isinstance(net.exercise_k, torch.nn.Embedding):
        item_n = int(net.exercise_k.num_embeddings)
    rems = 0.0
    if item_n is not None:
        y_true, y_prob = [], []
        net = net.to(device)
        net.eval()
        with torch.no_grad():
            for user, item, kq, y in loader:
                user = user.to(device)
                item = item.to(device)
                kq = kq.to(device)
                pred = net(user, item, kq)
                y_prob.append(pred.cpu())
                y_true.append(y.cpu())
        y_true = torch.cat(y_true)
        y_prob = torch.cat(y_prob)
        simple_mask = item < item_n // 2
        hard_mask = item >= item_n // 2
        try:
            if simple_mask.sum() > 0 and hard_mask.sum() > 0:
                auc_simple = roc_auc_score(y_true[simple_mask].numpy(), y_prob[simple_mask].numpy())
                auc_hard = roc_auc_score(y_true[hard_mask].numpy(), y_prob[hard_mask].numpy())
                rems = abs(auc_simple - auc_hard)
        except Exception:
            rems = 0.0
    return auc, acc, f1, rmse, rems


def main():
    set_seed(2025)
    print(f"DEVICE = {DEVICE}")

    train_raw, valid_raw = dataloader.CD_DL()
    if hasattr(dataloader, "Test_DL"):
        test_raw = dataloader.Test_DL()
    else:
        print("dataloader.Test_DL() not found, using valid as test")
        test_raw = valid_raw

    u_min, u_max, i_min, i_max, knowledge_n = infer_ids_and_knowledge_dim(
        [train_raw, valid_raw, test_raw]
    )
    user_offset = 1 if u_min == 1 else 0
    item_offset = 1 if i_min == 1 else 0
    student_n = (u_max - user_offset) + 1
    exer_n = (i_max - item_offset) + 1
    emb_dim = knowledge_n

    print(f"[INFER] student_n={student_n}, exer_n={exer_n}, knowledge_n={knowledge_n}")

    train_loader = ZeroBasedLoader(train_raw, user_offset, item_offset)
    valid_loader = ZeroBasedLoader(valid_raw, user_offset, item_offset)
    test_loader  = ZeroBasedLoader(test_raw,  user_offset, item_offset)

    # -------- baseline CDND --------
    print("\n================= TRAIN CDND baseline =================")
    model = cdnd.ACD(student_n, exer_n, knowledge_n, emb_dim)
    model.train(train_data=train_loader, test_data=valid_loader,
                epoch=EPOCHS, device=DEVICE, lr=LR)

    src_path = os.path.join("params", "arcd.params")
    if os.path.exists(src_path):
        best_path = os.path.join("params", "arcd_best.params")
        shutil.copyfile(src_path, best_path)
        model.load(best_path)
        print(f"Best checkpoint: {best_path}")
    else:
        print("No checkpoint found, using final model")

    auc, acc, f1, rmse, rems = evaluate_metrics_from_model(model, test_loader, device=DEVICE)
    print(f"CDND_baseline TEST: AUC={auc:.6f}, ACC={acc:.6f}, F1={f1:.6f}, RMSE={rmse:.6f}, REMS={rems:.6f}")

    results = [{
        "model": "CDND_baseline",
        "AUC": auc, "ACC": acc, "F1": f1,
        "RMSE": rmse, "REMS": rems,
        "epochs": EPOCHS, "lr": LR,
    }]

    df = pd.DataFrame(results)
    print("\nResults:")
    print(df)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
