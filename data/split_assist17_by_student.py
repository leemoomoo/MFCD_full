# coding: utf-8
# File: data/split_assist17_by_student.py

import os
import argparse
import numpy as np
import pandas as pd


def pick_col(df: pd.DataFrame, candidates, name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find {name}. Tried {candidates}. Got columns={list(df.columns)[:80]}...")


def split_indices_per_student(df: pd.DataFrame, user_col: str, seed: int, train_ratio: float):
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []

    for _, g in df.groupby(user_col, sort=False):
        idxs = g.index.values
        rng.shuffle(idxs)
        n = len(idxs)
        if n < 2:
            # 无法按 8:2 切，直接丢到 train（也可以选择丢掉）
            train_idx.append(int(idxs[0]))
            continue

        n_train = int(round(n * train_ratio))
        n_train = max(1, min(n - 1, n_train))  # 确保 train/test 都非空
        train_idx.extend(idxs[:n_train].tolist())
        test_idx.extend(idxs[n_train:].tolist())

    return train_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="assist17.csv",
                        help="Path relative to data/ (default: assist17.csv)")
    parser.add_argument("--out_dir", type=str, default="Assist17_split",
                        help="Output directory relative to data/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    DATA_DIR = os.path.abspath(os.path.dirname(__file__))

    in_path = args.input
    if not os.path.isabs(in_path):
        in_path = os.path.join(DATA_DIR, in_path)
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"File not found: {in_path}")

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(DATA_DIR, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(in_path)

    # --- find required cols ---
    ucol = pick_col(df, ["studentId", "user_id", "user", "uid"], "studentId")
    icol = pick_col(df, ["problemId", "item_id", "problem_id", "exercise_id"], "problemId")
    scol = pick_col(df, ["skill", "kc", "knowledge_id", "skill_id"], "skill")
    ycol = pick_col(df, ["correct", "label", "is_correct", "score"], "correct")

    # optional cols (for fluency)
    opp_col = next((c for c in ["totalFrSkillOpportunities", "opportunity"] if c in df.columns), None)
    time_col = next((c for c in ["timeTaken", "ms_first_response", "response_time", "rt"] if c in df.columns), None)
    wrong8_col = next((c for c in ["frPast8WrongCount", "past8WrongCount"] if c in df.columns), None)
    hint_col = next((c for c in ["frIsHelpRequestScaffolding", "helpAccessUnder2Sec", "endsWithScaffolding"] if c in df.columns), None)

    # --- clean NA + binarize label ---
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[ucol, icol, scol, ycol]).copy()
    df[ycol] = (df[ycol].astype(float) > 0.5).astype(int)
    after = len(df)

    # reset index for stable slicing
    df = df.reset_index(drop=True)

    # --- per-student split 8:2 ---
    train_idx, test_idx = split_indices_per_student(df, ucol, seed=args.seed, train_ratio=args.train_ratio)
    df_train = df.iloc[train_idx].sort_index()
    df_test = df.iloc[test_idx].sort_index()

    # --- rename to canonical names ---
    rename_map = {
        ucol: "studentId",
        icol: "problemId",
        scol: "skill",
        ycol: "correct",
    }
    if opp_col is not None:
        rename_map[opp_col] = "opportunity"
    if time_col is not None:
        rename_map[time_col] = "timeTaken"
    if wrong8_col is not None:
        rename_map[wrong8_col] = "past8Wrong"
    if hint_col is not None:
        rename_map[hint_col] = "hintLike"

    df_train = df_train.rename(columns=rename_map)
    df_test = df_test.rename(columns=rename_map)

    # --- DROP ALL UNNEEDED COLS: keep only minimal set ---
    keep = ["studentId", "problemId", "skill", "correct",
            "opportunity", "timeTaken", "past8Wrong", "hintLike"]
    keep_train = [c for c in keep if c in df_train.columns]
    keep_test = [c for c in keep if c in df_test.columns]

    df_train = df_train[keep_train].copy()
    df_test = df_test[keep_test].copy()

    # save
    train_path = os.path.join(out_dir, "train.csv")
    test_path = os.path.join(out_dir, "test.csv")
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"[Input] {in_path}")
    print(f"[Clean] dropped {before - after} rows with missing core fields")
    print(f"[Saved] {train_path} cols={list(df_train.columns)} rows={len(df_train)} users={df_train['studentId'].nunique()}")
    print(f"[Saved] {test_path}  cols={list(df_test.columns)} rows={len(df_test)} users={df_test['studentId'].nunique()}")
    print(f"[Label] train_pos_rate={df_train['correct'].mean():.4f} test_pos_rate={df_test['correct'].mean():.4f}")


if __name__ == "__main__":
    main()
