# -*- coding: utf-8 -*-
# Split ASSIST09 by STUDENT (user_id) into train/test = 80/20
# and keep only columns needed for NCDM + Fluency.

import os
import pandas as pd
import numpy as np

# ===== 路径：以脚本所在目录为基准（你 qiege.py 在 data/assist09 下）=====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = r"D:\12487\NCDM PLUS2\data\assist09\assist09.csv"
OUT_DIR = r"D:\12487\NCDM PLUS2\data\assist09\splits"

# NCDM + Fluency 最小字段集
KEEP_COLS = [
    "user_id",
    "problem_id",
    "skill_id",
    "correct",
    "opportunity",
    "opportunity_original",
    "ms_first_response",
    "attempt_count",
    "hint_count",
]

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def read_csv_robust(path: str) -> pd.DataFrame:
    # ASSIST09 常见编码问题：latin1 更稳
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Cannot find input csv: {INPUT_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)

    df = read_csv_robust(INPUT_PATH)

    # 只保留存在的列
    keep = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"[Warn] Missing columns (ignored): {missing}")
    df = df[keep].copy()

    # correct 清洗
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df.dropna(subset=["correct"])
    df["correct"] = df["correct"].astype(int)

    # 丢掉 skill_id 缺失（否则无法构造 q 向量）
    df = df.dropna(subset=["skill_id"])

    # 尽量把 skill_id 转成数值（ASSIST09 常见是数值 id）
    df["skill_id"] = pd.to_numeric(df["skill_id"], errors="coerce")
    df = df.dropna(subset=["skill_id"])
    df["skill_id"] = df["skill_id"].astype(int)

    # 行为列转数值（缺失填 0）
    for col in ["opportunity", "opportunity_original", "ms_first_response", "attempt_count", "hint_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # ===== 按学生切分 =====
    users = df["user_id"].unique()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(users)

    n_train_users = int(len(users) * TRAIN_RATIO)
    train_users = set(users[:n_train_users])
    test_users = set(users[n_train_users:])

    train_df = df[df["user_id"].isin(train_users)].copy()
    test_df = df[df["user_id"].isin(test_users)].copy()

    # sanity check：学生不重叠
    overlap = set(train_df["user_id"].unique()).intersection(set(test_df["user_id"].unique()))
    assert len(overlap) == 0, f"Train/Test user overlap found: {len(overlap)}"

    train_path = os.path.join(OUT_DIR, "assist09_train.csv")
    test_path = os.path.join(OUT_DIR, "assist09_test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print("Done: split by user_id (student-level).")
    print(f"Total rows: {len(df)} | users: {len(users)}")
    print(f"Train rows: {len(train_df)} | train users: {len(train_users)}")
    print(f"Test  rows: {len(test_df)} | test users: {len(test_users)}")
    print("Saved:")
    print(" -", train_path)
    print(" -", test_path)


if __name__ == "__main__":
    main()
