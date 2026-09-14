# coding: utf-8
# coding: utf-8
import os
import sys
import itertools
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ================= 关键：添加这一段路径配置 =================
model_dir = r"D:\12487\NCDM PLUS2\models\ncdm"
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)
# ========================================================

# 现在可以正常导入了
from NCDM_Fluency import NCDMFluency
from assist09_dataset import build_assist09_splits_from_folder

# 设置搜索空间
SEARCH_SPACE = {
    "lr": [0.001, 0.002, 0.005],
    "fluency_lambda": [0.3, 0.5, 0.7, 0.9],
    "dropout": [0.3, 0.5]
}


def run_search(train_csv, test_csv, save_dir="./hp_results"):
    # 1. 加载数据
    # 这里记得保持和你之前成功的加载方式一致
    dataset_params = {"skill_col": "skill_id", "item_col": "problem_id"}

    print("Loading datasets...")
    train_ds, _, test_ds, meta = build_assist09_splits_from_folder(
        os.path.dirname(train_csv), use_fluency=True,
        train_file=os.path.basename(train_csv),
        valid_file="valid.csv",
        test_file=os.path.basename(test_csv),
        item_file="item.csv",
    )

    # 2. 生成组合
    keys, values = zip(*SEARCH_SPACE.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    all_results = []

    # 3. 循环实验
    for i, config in enumerate(combinations):
        print(f"\n[{i + 1}/{len(combinations)}] Running config: {config}")

        model = NCDMFluency(
            knowledge_n=train_ds.knowledge_n,
            exer_n=train_ds.exer_n,
            student_n=train_ds.student_n,
            lr=config['lr'],
            fluency_lambda=config['fluency_lambda'],
            dropout=config['dropout']
        )

        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        model.train(train_loader, test_loader, epoch=5, device="cuda", silence=True)
        auc, acc = model.eval(test_loader, device="cuda", silence=True)

        result = {**config, "auc": auc, "acc": acc}
        all_results.append(result)

        # 实时保存
        pd.DataFrame(all_results).to_csv(os.path.join(save_dir, "search_results.csv"), index=False)


if __name__ == "__main__":
    os.makedirs("./hp_results", exist_ok=True)
    run_search(
        r"D:\12487\NCDM PLUS2\data\a0910\train.csv",
        r"D:\12487\NCDM PLUS2\data\a0910\test.csv"
    )