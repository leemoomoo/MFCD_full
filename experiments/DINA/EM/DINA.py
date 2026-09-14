# coding: utf-8
# Run EM-DINA on assist09 with AUC / ACC / F1 / RMSE

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from EduCDM import EMDINA as DINA
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error


TOP_K = 5
TEST_RATIO = 0.2
MIN_INTER_PER_USER = 3
EPOCH = 20
EPSILON = 1e-3
RANDOM_SEED = 42
THRESHOLD = 0.5

TRAIN_PATH = "../../../data/assist09/splits/assist09_train.csv"
TEST_PATH = "../../../data/assist09/splits/assist09_test.csv"
SAVE_PATH = f"dina_assist09_top{TOP_K}.params"


def print_stats(title, df):
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)
    print("size:", len(df))
    if len(df) > 0:
        print("users:", df["user_id"].nunique())
        print("problems:", df["problem_id"].nunique())
        print("skills:", df["skill_id"].nunique())


def clean_df(df):
    required_cols = ["user_id", "problem_id", "skill_id", "correct"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}")

    df = df.copy()
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df.dropna(subset=["user_id", "problem_id", "skill_id", "correct"]).copy()
    df = df[df["correct"].isin([0, 1])].copy()

    # 同一学生-题目保留最后一次作答
    df = df.sort_index().drop_duplicates(
        subset=["user_id", "problem_id"], keep="last"
    ).copy()
    return df


def split_by_user(df, test_ratio=0.2, min_inter=3, seed=42):
    rng = np.random.RandomState(seed)
    train_parts = []
    test_parts = []

    user_groups = df.groupby("user_id")

    for _, group in tqdm(user_groups, total=df["user_id"].nunique(), desc="Splitting by user"):
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(group)

        if n < min_inter:
            train_parts.append(group)
            continue

        test_n = max(1, int(round(n * test_ratio)))
        if test_n >= n:
            test_n = n - 1

        test_idx = rng.choice(n, size=test_n, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[test_idx] = True

        test_group = group[mask].copy()
        train_group = group[~mask].copy()

        if len(train_group) == 0:
            train_group = group.iloc[:1].copy()
            test_group = group.iloc[1:].copy()

        train_parts.append(train_group)
        if len(test_group) > 0:
            test_parts.append(test_group)

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).reset_index(drop=True)

    return train_df, test_df


def build_q_matrix(df_q, problem_map, skill_map):
    q_m = np.zeros((len(problem_map), len(skill_map)), dtype=int)

    for row in tqdm(df_q.itertuples(index=False), total=len(df_q), desc="Building Q-matrix"):
        p = row.problem_id
        s = row.skill_id
        if p in problem_map and s in skill_map:
            q_m[problem_map[p], skill_map[s]] = 1

    return q_m


def build_response_matrix(train_df, user_map, problem_map):
    R = -1 * np.ones((len(user_map), len(problem_map)), dtype=float)

    for row in tqdm(train_df.itertuples(index=False), total=len(train_df), desc="Building response matrix R"):
        u = row.user_id
        p = row.problem_id
        c = float(row.correct)
        if u in user_map and p in problem_map:
            R[user_map[u], problem_map[p]] = c

    return R


def build_test_set(test_df, user_map, problem_map):
    test_set = []

    for row in tqdm(test_df.itertuples(index=False), total=len(test_df), desc="Building test_set"):
        u = row.user_id
        p = row.problem_id
        c = float(row.correct)

        if u in user_map and p in problem_map:
            test_set.append({
                "user_id": user_map[u],
                "item_id": problem_map[p],
                "score": float(c)
            })

    return test_set


def evaluate_metrics(cdm, test_set, threshold=0.5):
    slip = np.clip(np.asarray(cdm.slip, dtype=np.float64), 1e-5, 1 - 1e-5)
    guess = np.clip(np.asarray(cdm.guess, dtype=np.float64), 1e-5, 1 - 1e-5)
    eta = np.asarray(cdm.eta, dtype=np.float64)

    pred_score = (1 - slip) * eta + guess * (1 - eta)
    pred_score = np.clip(pred_score, 1e-5, 1 - 1e-5)

    y_true = []
    y_prob = []

    for rec in tqdm(test_set, desc="Evaluating metrics"):
        stu = rec["user_id"]
        item = rec["item_id"]
        true_score = int(rec["score"])

        prob = float(pred_score[int(cdm.theta[stu]), item])
        prob = float(np.clip(prob, 1e-5, 1 - 1e-5))

        y_true.append(true_score)
        y_prob.append(prob)

    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
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


def main():
    logging.getLogger().setLevel(logging.INFO)

    # =========================
    # 1. 读取原始数据
    # =========================
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)

    print_stats("Raw Train", raw_train)
    print_stats("Raw Test", raw_test)

    # =========================
    # 2. 清洗
    # =========================
    raw_train = clean_df(raw_train)
    raw_test = clean_df(raw_test)

    print_stats("Cleaned Train", raw_train)
    print_stats("Cleaned Test", raw_test)

    # =========================
    # 3. 合并
    # =========================
    full_df = pd.concat([raw_train, raw_test], axis=0).reset_index(drop=True)
    full_df = full_df.sort_index().drop_duplicates(
        subset=["user_id", "problem_id"], keep="last"
    ).copy()

    print_stats("Merged Full Data", full_df)

    # =========================
    # 4. 选 Top-K 高频 skill
    # =========================
    skill_counts = full_df["skill_id"].value_counts()
    print("\nTop 20 skills by frequency:")
    print(skill_counts.head(20))

    top_skills = skill_counts.head(TOP_K).index.tolist()
    full_df = full_df[full_df["skill_id"].isin(top_skills)].copy()

    print_stats(f"After Top-{TOP_K} Skill Filtering", full_df)

    # =========================
    # 5. 按学生重划分 train/test
    # =========================
    train_df, test_df = split_by_user(
        full_df,
        test_ratio=TEST_RATIO,
        min_inter=MIN_INTER_PER_USER,
        seed=RANDOM_SEED
    )

    print_stats("Re-split Train", train_df)
    print_stats("Re-split Test", test_df)

    # =========================
    # 6. 建立映射
    # =========================
    train_users = train_df["user_id"].unique()
    train_problems = train_df["problem_id"].unique()
    skills = sorted(pd.concat([train_df["skill_id"], test_df["skill_id"]]).unique())

    user_map = {u: i for i, u in enumerate(train_users)}
    problem_map = {p: i for i, p in enumerate(train_problems)}
    skill_map = {s: i for i, s in enumerate(skills)}

    test_df = test_df[
        test_df["user_id"].isin(train_users) &
        test_df["problem_id"].isin(train_problems)
    ].copy()

    print("\nInitial mapping stats:")
    print("students:", len(train_users))
    print("problems:", len(train_problems))
    print("skills:", len(skills))
    print("aligned test size:", len(test_df))

    # =========================
    # 7. 构造 Q-matrix
    # =========================
    q_source = pd.concat([train_df, test_df], axis=0)
    q_m = build_q_matrix(q_source, problem_map, skill_map)

    print("\nInitial Q-matrix shape:", q_m.shape)
    print("zero-q items:", int(np.sum(q_m.sum(axis=1) == 0)))

    valid_problem_mask = q_m.sum(axis=1) > 0
    old_problem_list = list(problem_map.keys())
    valid_problem_ids = [old_problem_list[i] for i in np.where(valid_problem_mask)[0]]
    valid_problem_ids_set = set(valid_problem_ids)

    train_df = train_df[train_df["problem_id"].isin(valid_problem_ids_set)].copy()
    test_df = test_df[test_df["problem_id"].isin(valid_problem_ids_set)].copy()

    train_problems = train_df["problem_id"].unique()
    problem_map = {p: i for i, p in enumerate(train_problems)}

    q_source = pd.concat([train_df, test_df], axis=0)
    q_m = build_q_matrix(q_source, problem_map, skill_map)

    print("\nAfter removing zero-q items:")
    print("Q-matrix shape:", q_m.shape)
    print("zero-q items:", int(np.sum(q_m.sum(axis=1) == 0)))

    # =========================
    # 8. 构造响应矩阵 R
    # =========================
    R = build_response_matrix(train_df, user_map, problem_map)

    valid_student_mask = (R != -1).sum(axis=1) > 0
    old_user_list = list(user_map.keys())
    valid_user_ids = [old_user_list[i] for i in np.where(valid_student_mask)[0]]
    valid_user_ids_set = set(valid_user_ids)

    train_df = train_df[train_df["user_id"].isin(valid_user_ids_set)].copy()
    test_df = test_df[test_df["user_id"].isin(valid_user_ids_set)].copy()

    train_users = train_df["user_id"].unique()
    user_map = {u: i for i, u in enumerate(train_users)}

    R = build_response_matrix(train_df, user_map, problem_map)

    stu_num = len(user_map)
    prob_num = len(problem_map)
    know_num = len(skill_map)

    print("\nFinal matrix stats:")
    print("R shape:", R.shape)
    print("Q-matrix shape:", q_m.shape)
    print("students:", stu_num)
    print("problems:", prob_num)
    print("skills:", know_num)
    print("students with no response:", int(np.sum((R != -1).sum(axis=1) == 0)))
    print("items with no response:", int(np.sum((R != -1).sum(axis=0) == 0)))

    observed = (R != -1)
    print("observed responses:", int(np.sum(observed)))
    print("avg responses per student:", float(np.mean(observed.sum(axis=1))))
    print("avg responses per item:", float(np.mean(observed.sum(axis=0))))
    print("R min/max:", float(np.min(R[observed])), float(np.max(R[observed])))
    print("q ones per item min/max:", int(q_m.sum(axis=1).min()), int(q_m.sum(axis=1).max()))

    # =========================
    # 9. 构造测试集
    # =========================
    test_set = build_test_set(test_df, user_map, problem_map)
    print("final test size:", len(test_set))

    if len(test_set) == 0:
        raise ValueError("test_set 为空，无法评估。")

    # =========================
    # 10. 训练
    # =========================
    print("\nStart training...")
    cdm = DINA(
        R,
        q_m,
        stu_num,
        prob_num,
        know_num,
        skip_value=-1
    )

    cdm.train(epoch=EPOCH, epsilon=EPSILON)
    cdm.save(SAVE_PATH)

    # =========================
    # 11. 评估
    # =========================
    print("\nStart evaluating...")
    cdm.load(SAVE_PATH)

    auc, acc, f1, rmse = evaluate_metrics(cdm, test_set, threshold=THRESHOLD)

    print("\nResult:")
    print(f"AUC : {auc:.6f}" if not np.isnan(auc) else "AUC : nan")
    print(f"ACC : {acc:.6f}")
    print(f"F1  : {f1:.6f}")
    print(f"RMSE: {rmse:.6f}")


if __name__ == "__main__":
    main()