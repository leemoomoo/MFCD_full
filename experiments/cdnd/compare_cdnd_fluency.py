# coding: utf-8
"""
compare_cdnd_fluency.py
对比 CDND baseline (ACD) vs CDND+Fluency (ACDWithFluency) on a0910
"""
import os, random, logging, shutil
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import dataloader
from model import cdnd
from model import cdnd_fluency


def set_seed(seed=2025):
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
FLU_LATENT_DIM = 16
AUX_LOSS_WEIGHT = 0.1

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs", "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "compare_cdnd_fluency.csv")
os.makedirs(os.path.join(THIS_DIR, "params"), exist_ok=True)
logging.getLogger().setLevel(logging.INFO)


class ZeroBasedLoader:
    def __init__(self, base_loader, user_offset, item_offset):
        self.base_loader = base_loader
        self.user_offset = user_offset
        self.item_offset = item_offset

    def __iter__(self):
        for user_id, item_id, knowledge_emb, y in self.base_loader:
            yield (user_id - self.user_offset,
                   item_id - self.item_offset, knowledge_emb, y)

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


def evaluate_metrics(model, loader, device=DEVICE):
    """Returns (auc, acc, f1, rmse)."""
    auc, acc, rmse, f1 = model.eval(loader, device=device)
    return auc, acc, f1, rmse


def train_and_save_best(model, train_loader, valid_loader, tag, save_name):
    model.train(train_data=train_loader, test_data=valid_loader,
                epoch=EPOCHS, device=DEVICE, lr=LR)
    src = os.path.join("params", save_name)
    if os.path.exists(src):
        dst = src.replace(".params", "_best.params")
        shutil.copyfile(src, dst)
        model.load(dst)
        print(f"[{tag}] Loaded best from {dst}")
    return model


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

    results = []

    # ====== 1) Baseline ACD ======
    print("\n" + "=" * 50)
    print("TRAIN CDND BASELINE (ACD)")
    print("=" * 50)
    m1 = cdnd.ACD(student_n, exer_n, knowledge_n, emb_dim)
    train_and_save_best(m1, train_loader, valid_loader, "BASELINE", "arcd.params")
    auc1, acc1, f1_1, rmse1 = evaluate_metrics(m1, test_loader)
    print(f"BASELINE: AUC={auc1:.6f} ACC={acc1:.6f} F1={f1_1:.6f} RMSE={rmse1:.6f}")
    results.append({"model": "CDND_baseline", "AUC": auc1, "ACC": acc1, "F1": f1_1, "RMSE": rmse1})

    # ====== 2) ACD + Fluency ======
    print("\n" + "=" * 50)
    print("TRAIN CDND FLUENCY (ACDWithFluency)")
    print("=" * 50)
    m2 = cdnd_fluency.ACDWithFluency(
        student_n, exer_n, knowledge_n, emb_dim,
        flu_latent_dim=FLU_LATENT_DIM, aux_loss_weight=AUX_LOSS_WEIGHT,
    )
    train_and_save_best(m2, train_loader, valid_loader, "FLUENCY", "arcd_fluency.params")
    auc2, acc2, f1_2, rmse2 = evaluate_metrics(m2, test_loader)
    print(f"FLUENCY: AUC={auc2:.6f} ACC={acc2:.6f} F1={f1_2:.6f} RMSE={rmse2:.6f}")
    results.append({"model": "CDND_fluency", "AUC": auc2, "ACC": acc2, "F1": f1_2, "RMSE": rmse2})

    # Save
    df = pd.DataFrame(results)
    print("\nComparison:")
    print(df)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
