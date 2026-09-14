# coding: utf-8
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Assist09RawFluencyDataset(Dataset):
    """
    适配字段类似下面的 ASSIST09 原始数据：

    order_id, assignment_id, user_id, assistment_id, problem_id,
    original, correct, attempt_count, ms_first_response,
    tutor_mode, answer_type, sequence_id, student_class_id,
    position, type

    返回：
    user_id, item_id, knowledge_emb, y, opportunity, rt, attempt, hint
    """

    def __init__(
        self,
        csv_path,
        user_map=None,
        item_map=None,
        skill_map=None,
        user_oov_id=None,
        item_col="problem_id",
        skill_col="type",
        user_col="user_id",
        label_col="correct",
        rt_col="ms_first_response",
        attempt_col="attempt_count",
        hint_col="hint_count",
        order_col="order_id",
    ):
        df = pd.read_csv(csv_path, encoding="utf-8")

        required = [
            user_col,
            item_col,
            skill_col,
            label_col,
            rt_col,
            attempt_col,
        ]

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} 缺少必要字段: {missing}")

        # -------- 基础清洗 --------
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        df = df.dropna(subset=[label_col])
        df[label_col] = df[label_col].astype(int)

        df = df.dropna(subset=[user_col, item_col, skill_col])

        df[rt_col] = pd.to_numeric(df[rt_col], errors="coerce").fillna(0.0)
        df[attempt_col] = pd.to_numeric(df[attempt_col], errors="coerce").fillna(1.0)

        if hint_col in df.columns:
            df[hint_col] = pd.to_numeric(df[hint_col], errors="coerce").fillna(0.0)
        else:
            df[hint_col] = 0.0

        # 保证时间顺序，用于计算 opportunity
        if order_col in df.columns:
            df = df.sort_values(order_col).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

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

        # 测试集/验证集：过滤掉训练集中没见过的题目和知识点
        if not is_train:
            before = len(df)
            df = df[df[item_col].isin(item_map)]
            df = df[df[skill_col].isin(skill_map)]
            after = len(df)
            print(f"[{os.path.basename(csv_path)}] 过滤未知 item/skill: {before} -> {after}")

        # -------- 计算 opportunity --------
        # 含义：当前学生在当前知识点上第几次练习
        df["_opportunity"] = (
            df.groupby([user_col, skill_col]).cumcount() + 1
        ).astype(np.float32)

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
        self.opp = df["_opportunity"].values.astype(np.float32)
        self.rt = df[rt_col].values.astype(np.float32)
        self.attempt = df[attempt_col].values.astype(np.float32)
        self.hint = df[hint_col].values.astype(np.float32)

        self.knowledge_n = len(skill_map)
        self.exer_n = len(item_map)
        self.student_n = len(user_map) + 1

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        user_id = torch.tensor(self.users[idx], dtype=torch.long)
        item_id = torch.tensor(self.items[idx], dtype=torch.long)

        knowledge_emb = torch.zeros(self.knowledge_n, dtype=torch.float32)
        knowledge_emb[int(self.skills[idx])] = 1.0

        y = torch.tensor(self.y[idx], dtype=torch.float32)
        opp = torch.tensor(self.opp[idx], dtype=torch.float32)
        rt = torch.tensor(self.rt[idx], dtype=torch.float32)
        attempt = torch.tensor(self.attempt[idx], dtype=torch.float32)
        hint = torch.tensor(self.hint[idx], dtype=torch.float32)

        return user_id, item_id, knowledge_emb, y, opp, rt, attempt, hint