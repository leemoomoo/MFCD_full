# coding: utf-8
import os
import time
import json
import pickle
import argparse
import ast
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from EduCDM import CDM


# =========================
# Logger
# =========================
class Logger:
    def __init__(self):
        self.log_file = None
        self.log_dir = None

    def log(self, s):
        if self.log_file is not None:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(s)

    def set_logdir(self, log_dir):
        self.log_dir = log_dir

    def set_filename(self, filename):
        self.log_file = filename

    def close(self):
        pass


logger = Logger()


def log(s):
    print(s)
    logger.log(str(s) + '\n')


# =========================
# Utils
# =========================
def transform(user, item, item2knowledge, score, batch_size, knowledge_n,
              opportunity=None, acc_prior=None, stability=None):
    knowledge_emb = torch.zeros((len(item), knowledge_n), dtype=torch.float32)

    user = np.array(user)
    item = np.array(item)
    score = np.array(score)

    for idx in range(len(item)):
        item_id = item[idx]
        if item_id in item2knowledge:
            knowledge_emb[idx][np.array(item2knowledge[item_id], dtype=int)] = 1.0

    tensors = [
        torch.tensor(user, dtype=torch.int64),
        torch.tensor(item, dtype=torch.int64),
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    ]

    if opportunity is not None and acc_prior is not None and stability is not None:
        tensors.extend([
            torch.tensor(np.array(opportunity), dtype=torch.float32),
            torch.tensor(np.array(acc_prior), dtype=torch.float32),
            torch.tensor(np.array(stability), dtype=torch.float32),
        ])

    data_set = TensorDataset(*tensors)
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def load_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def safe_load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        content = f.read().strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end + 1])
        raise ValueError(f"无法解析 JSON 文件: {path}\n内容:\n{content}")


def sort_and_fill_dict(input_dict, max_key, min_key=0):
    if not input_dict:
        return {}

    example_length = len(next(iter(input_dict.values())))
    all_keys = list(range(min_key, max_key))

    for key in all_keys:
        if key not in input_dict:
            input_dict[key] = [0] * example_length

    sorted_dict = {key: input_dict[key] for key in sorted(all_keys)}
    return sorted_dict


def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少文件: {path}")


# =========================
# Loss / basic modules
# =========================
def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
    normed_embeds1 = embeds1 / torch.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
    normed_embeds2 = embeds2 / torch.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
    normed_all_embeds2 = all_embeds2 / torch.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))

    # 保留你原代码风格
    nume_term = torch.exp(normed_embeds1 * normed_embeds2 / temp).sum()
    deno_term = torch.sum(torch.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1)
    cl_loss = -torch.log(nume_term / (deno_term + 1e-8) + 1e-8).mean()
    return cl_loss


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


# =========================
# Channel Gate
# =========================
class ChannelGate(nn.Module):
    """Input: core_feat [B,K] -> gate [B,K] in (0,1). Activated on mask=1."""
    def __init__(self, k_n, hidden_ratio=0.5):
        super().__init__()
        hidden = max(8, int(k_n * hidden_ratio))
        self.fc1 = nn.Linear(k_n, hidden)
        self.fc2 = nn.Linear(hidden, k_n)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, core_feat, mask):
        h = F.relu(self.fc1(core_feat))
        gate = torch.sigmoid(self.fc2(h))
        gate = gate * mask
        return gate


# =========================
# Channel Gate
# =========================
class FluencyMixin(nn.Module):
    def _build_fluency_module(self, emb_num, knowledge_dim,
                              flu_latent_dim=16, dropout=0.5, gamma_init=1.0):
        self.student_flu_bias = nn.Embedding(emb_num, flu_latent_dim)

        self.beh_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, flu_latent_dim),
            nn.ReLU(),
        )

        self.flu_score_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.opp_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.acc_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.sta_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.fuse_gate = nn.Sequential(
            nn.Linear(flu_latent_dim + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    @staticmethod
    def _safe_log1p(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=0.0)
        return torch.log1p(x)

    @staticmethod
    def _to_col(t: torch.Tensor) -> torch.Tensor:
        return t.view(-1, 1) if t.dim() == 1 else t

    def _fluency_forward(self, stu_id, e_difficulty, k_difficulty,
                         opportunity, acc_prior, stability):
        opp = self._to_col(opportunity.float())
        acc = self._to_col(acc_prior.float())
        sta = self._to_col(stability.float())

        f_opp = self._safe_log1p(opp)
        f_acc = torch.clamp(acc, 0.0, 1.0)

        # 维持你之前 fluency 代码的定义
        f_sta = 1.0 - torch.clamp(4.0 * torch.clamp(sta, min=0.0), 0.0, 1.0)

        beh = torch.cat([f_opp, f_acc, f_sta], dim=1)

        z = self.beh_encoder(beh)
        z = z + torch.tanh(self.student_flu_bias(stu_id))

        flu = torch.sigmoid(self.flu_score_head(z))

        opp_hat = self.opp_head(z)
        acc_hat = torch.sigmoid(self.acc_head(z))
        sta_hat = torch.sigmoid(self.sta_head(z))

        gate_in = torch.cat(
            [z, e_difficulty, k_difficulty.mean(dim=1, keepdim=True)],
            dim=1
        )
        gate = torch.sigmoid(self.fuse_gate(gate_in))

        return {
            "z": z,
            "flu": flu,
            "gate": gate,
            "opp_hat": opp_hat.view(-1),
            "acc_hat": acc_hat.view(-1),
            "sta_hat": sta_hat.view(-1),
            "opp_tgt": f_opp.view(-1),
            "acc_tgt": f_acc.view(-1),
            "sta_tgt": f_sta.view(-1),
        }


# =========================
# KCD + Fluency
# =========================
class KCDNetFluency(FluencyMixin, nn.Module):
    """
    单文件整合版:
    - NCD backbone
    - KCD contrastive alignment
    - Fluency branch
    """

    def __init__(self, knowledge_n, exer_n, student_n,
                 usr_emb, itm_emb, similar_students, similar_exercises,
                 flu_latent_dim=16, dropout=0.5):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256

        super(KCDNetFluency, self).__init__()

        # ---------- NCD backbone ----------
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)

        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=dropout)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=dropout)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # ---------- KCD side info ----------
        self.register_buffer(
            "similar_students",
            torch.tensor([similar_students[key] for key in similar_students], dtype=torch.long)
        )
        self.register_buffer(
            "similar_exercises",
            torch.tensor([similar_exercises[key] for key in similar_exercises], dtype=torch.long)
        )
        self.register_buffer(
            "usrprf_embeds_raw",
            torch.tensor(usr_emb, dtype=torch.float32)
        )
        self.register_buffer(
            "itmprf_embeds_raw",
            torch.tensor(itm_emb, dtype=torch.float32)
        )

        self.mlp_stu = nn.Sequential(
            nn.Linear(self.usrprf_embeds_raw.shape[1],
                      (self.usrprf_embeds_raw.shape[1] + self.stu_dim) // 2),
            nn.ReLU(),
            nn.Linear((self.usrprf_embeds_raw.shape[1] + self.stu_dim) // 2, self.stu_dim)
        )
        self.mlp_pro = nn.Sequential(
            nn.Linear(self.itmprf_embeds_raw.shape[1],
                      (self.itmprf_embeds_raw.shape[1] + 2000) // 2),
            nn.ReLU(),
            nn.Linear((self.itmprf_embeds_raw.shape[1] + 2000) // 2, 1 + self.knowledge_dim)
        )

        # ---------- Fluency ----------
        self._build_fluency_module(
            emb_num=self.emb_num,
            knowledge_dim=self.knowledge_dim,
            flu_latent_dim=flu_latent_dim,
            dropout=dropout,
        )

        self._init_weight()

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def _init_weight(self):
        for m in self.mlp_pro:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        for m in self.mlp_stu:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, stu_id, input_exercise, input_knowledge_point,
                opportunity, acc_prior, stability,
                return_aux=False):
        stu_id = torch.clamp(stu_id, 0, self.emb_num - 1)
        input_exercise = torch.clamp(input_exercise, 0, self.exer_n - 1)

        # ---------- NCD ----------
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))

        input_x = input_knowledge_point.float() * (stat_emb - k_difficulty) * e_difficulty
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        o_logit = self.prednet_full3(input_x)
        o = torch.sigmoid(o_logit)

        # ---------- Fluency ----------
        flu_out = self._fluency_forward(
            stu_id, e_difficulty, k_difficulty,
            opportunity, acc_prior, stability
        )

        gamma = F.softplus(self.gamma)
        delta = (flu_out["flu"] - 0.5) * 4.0
        p_logit = o_logit + gamma * flu_out["gate"] * delta
        p = torch.sigmoid(p_logit)

        # ---------- KCD contrastive ----------
        pro_emb = torch.cat(
            (self.e_difficulty(input_exercise), self.k_difficulty(input_exercise)),
            dim=1
        )

        usrprf_embeds = self.mlp_stu(self.usrprf_embeds_raw)
        itmprf_embeds = self.mlp_pro(self.itmprf_embeds_raw)

        usr_prf_embeds = usrprf_embeds[stu_id]
        itm_prf_embeds = itmprf_embeds[input_exercise]

        simi_user_idx = self.similar_students[stu_id]
        simi_exer_idx = self.similar_exercises[input_exercise]

        simi_user = usrprf_embeds[simi_user_idx].reshape(-1, self.stu_dim)
        simi_exer = itmprf_embeds[simi_exer_idx].reshape(-1, 1 + self.knowledge_dim)

        kd_loss = cal_infonce_loss(stu_emb, usr_prf_embeds, usrprf_embeds, 0.5) / stu_emb.shape[0] + \
                  cal_infonce_loss(pro_emb, itm_prf_embeds, itmprf_embeds, 0.5) / pro_emb.shape[0]

        simi_loss = cal_infonce_loss(stu_emb, usr_prf_embeds, simi_user, 0.5) / stu_emb.shape[0] + \
                    cal_infonce_loss(pro_emb, itm_prf_embeds, simi_exer, 0.5) / pro_emb.shape[0]

        kcd_loss = 0.04 * simi_loss + 0.015 * kd_loss

        if not return_aux:
            return p.view(-1), kcd_loss

        return {
            "p": p.view(-1),
            "o": o.view(-1),
            "flu": flu_out["flu"].view(-1),
            "gate": flu_out["gate"].view(-1),
            "opp_hat": flu_out["opp_hat"],
            "acc_hat": flu_out["acc_hat"],
            "sta_hat": flu_out["sta_hat"],
            "opp_tgt": flu_out["opp_tgt"],
            "acc_tgt": flu_out["acc_tgt"],
            "sta_tgt": flu_out["sta_tgt"],
            "kcd_loss": kcd_loss
        }

    def get_embeddings(self):
        return self.student_emb



class KCDNetWithChannel(KCDNetFluency):
    """KCDNetFluency + ChannelGate inserted after input_x."""
    def __init__(self, knowledge_n, exer_n, student_n,
                 usr_emb, itm_emb, similar_students, similar_exercises,
                 flu_latent_dim=16, dropout=0.5,
                 gate_alpha=0.5, gate_hidden_ratio=0.5):
        super().__init__(knowledge_n, exer_n, student_n,
                         usr_emb, itm_emb, similar_students, similar_exercises,
                         flu_latent_dim=flu_latent_dim, dropout=dropout)
        self.gate_alpha = gate_alpha
        self.channel_gate = ChannelGate(k_n=self.knowledge_dim, hidden_ratio=gate_hidden_ratio)

    def forward(self, stu_id, input_exercise, input_knowledge_point,
                opportunity, acc_prior, stability, return_aux=False):
        stu_id = torch.clamp(stu_id, 0, self.emb_num - 1)
        input_exercise = torch.clamp(input_exercise, 0, self.exer_n - 1)
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))
        input_x = input_knowledge_point.float() * (stat_emb - k_difficulty) * e_difficulty

        # Channel Gate
        mask = (input_knowledge_point > 0).float()
        gate = self.channel_gate(input_x, mask)
        scale = 1.0 + self.gate_alpha * (gate - 0.5)
        scale = torch.clamp(scale, min=0.0)
        input_x = input_x * scale

        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        o_logit = self.prednet_full3(input_x)
        o = torch.sigmoid(o_logit)

        flu_out = self._fluency_forward(stu_id, e_difficulty, k_difficulty, opportunity, acc_prior, stability)
        gamma = F.softplus(self.gamma)
        delta = (flu_out["flu"] - 0.5) * 4.0
        p_logit = o_logit + gamma * flu_out["gate"] * delta
        p = torch.sigmoid(p_logit)

        pro_emb = torch.cat((self.e_difficulty(input_exercise), self.k_difficulty(input_exercise)), dim=1)
        usrprf_embeds = self.mlp_stu(self.usrprf_embeds_raw)
        itmprf_embeds = self.mlp_pro(self.itmprf_embeds_raw)
        usr_prf_embeds = usrprf_embeds[stu_id]
        itm_prf_embeds = itmprf_embeds[input_exercise]
        simi_user_idx = self.similar_students[stu_id]
        simi_exer_idx = self.similar_exercises[input_exercise]
        simi_user = usrprf_embeds[simi_user_idx].reshape(-1, self.stu_dim)
        simi_exer = itmprf_embeds[simi_exer_idx].reshape(-1, 1 + self.knowledge_dim)
        kd_loss = (cal_infonce_loss(stu_emb, usr_prf_embeds, usrprf_embeds, 0.5) / stu_emb.shape[0] +
                   cal_infonce_loss(pro_emb, itm_prf_embeds, itmprf_embeds, 0.5) / pro_emb.shape[0])
        simi_loss = (cal_infonce_loss(stu_emb, usr_prf_embeds, simi_user, 0.5) / stu_emb.shape[0] +
                     cal_infonce_loss(pro_emb, itm_prf_embeds, simi_exer, 0.5) / pro_emb.shape[0])
        kcd_loss = 0.04 * simi_loss + 0.015 * kd_loss
        if not return_aux:
            return p.view(-1), kcd_loss
        return {"p": p.view(-1), "o": o.view(-1), "flu": flu_out["flu"].view(-1),
                "gate": flu_out["gate"].view(-1),
                "opp_hat": flu_out["opp_hat"], "acc_hat": flu_out["acc_hat"], "sta_hat": flu_out["sta_hat"],
                "opp_tgt": flu_out["opp_tgt"], "acc_tgt": flu_out["acc_tgt"], "sta_tgt": flu_out["sta_tgt"],
                "kcd_loss": kcd_loss}


class KCD_Fluency(CDM):
    def __init__(self, knowledge_n, exer_n, student_n,
                 log_name, best_model,
                 usr_emb, itm_emb, similar_students, similar_exercises,
                 lambda_kcd=1.0, lambda_opp=0.1, lambda_acc=0.05, lambda_sta=0.05,
                 flu_latent_dim=16, dropout=0.5):
        super(KCD_Fluency, self).__init__()
        self.ncdm_net = KCDNetFluency(
            knowledge_n, exer_n, student_n,
            usr_emb, itm_emb, similar_students, similar_exercises,
            flu_latent_dim=flu_latent_dim, dropout=dropout
        )

        self.log_name = log_name
        self.best_model = best_model

        self.lambda_kcd = float(lambda_kcd)
        self.lambda_opp = float(lambda_opp)
        self.lambda_acc = float(lambda_acc)
        self.lambda_sta = float(lambda_sta)

        self.loss_y = nn.BCELoss()
        self.loss_opp = nn.MSELoss()
        self.loss_acc = nn.SmoothL1Loss()
        self.loss_sta = nn.SmoothL1Loss()

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        last_time = time.time()
        best_auc = 0.0

        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)

        for epoch_i in range(epoch):
            epoch_losses = []
            ly_list, lkcd_list, lopp_list, lacc_list, lsta_list = [], [], [], [], []

            for batch_data in tqdm(train_data, f"Epoch {epoch_i}", disable=silence):
                if len(batch_data) != 7:
                    raise ValueError(
                        "当前 fluency 模型要求 dataloader 返回 7 个字段: "
                        "user_id, item_id, knowledge_emb, y, opportunity, acc_prior, stability"
                    )

                user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = batch_data

                user_id = user_id.to(device)
                item_id = item_id.to(device)
                knowledge_emb = knowledge_emb.to(device)
                y = (y > 0.5).float().to(device)

                opp = opp.to(device)
                acc_prior = acc_prior.to(device)
                stability = stability.to(device)

                out = self.ncdm_net(
                    user_id, item_id, knowledge_emb,
                    opp, acc_prior, stability,
                    return_aux=True
                )

                ly = self.loss_y(out["p"], y)
                lopp = self.loss_opp(out["opp_hat"], out["opp_tgt"])
                lacc = self.loss_acc(out["acc_hat"], out["acc_tgt"])
                lsta = self.loss_sta(out["sta_hat"], out["sta_tgt"])

                loss = ly + self.lambda_kcd * out["kcd_loss"] + \
                       self.lambda_opp * lopp + self.lambda_acc * lacc + self.lambda_sta * lsta

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                ly_list.append(ly.item())
                lkcd_list.append(out["kcd_loss"].item())
                lopp_list.append(lopp.item())
                lacc_list.append(lacc.item())
                lsta_list.append(lsta.item())

            now = time.time()
            msg = (
                f"[Epoch {epoch_i}] loss={float(np.mean(epoch_losses)):.6f} "
                f"(y={float(np.mean(ly_list)):.6f}, "
                f"kcd={float(np.mean(lkcd_list)):.6f}, "
                f"opp={float(np.mean(lopp_list)):.6f}, "
                f"acc={float(np.mean(lacc_list)):.6f}, "
                f"sta={float(np.mean(lsta_list)):.6f}) "
                f"time cost: {(now - last_time):.6f}"
            )
            print(msg)
            with open(self.log_name, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")
            last_time = now

            if test_data is not None:
                auc, accuracy, rmse, f1 = self.eval(
                    test_data,
                    device=device,
                    silence=silence
                )
                if auc > best_auc:
                    best_auc = auc
                    print("Update best auc!")
                    with open(self.log_name, 'a', encoding='utf-8') as f:
                        f.write("Update best auc!\n")
                    self.save(self.best_model)

                msg = "[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, f1: %.6f" % (
                    epoch_i, auc, accuracy, rmse, f1
                )
                print(msg)
                with open(self.log_name, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")

    @torch.no_grad()
    def eval(self, test_data, device="cpu", threshold=0.5, save=False, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()

        y_true, y_pred = [], []

        for batch_data in tqdm(test_data, "Evaluating", disable=silence):
            if len(batch_data) != 7:
                raise ValueError(
                    "当前 fluency 模型要求 dataloader 返回 7 个字段: "
                    "user_id, item_id, knowledge_emb, y, opportunity, acc_prior, stability"
                )

            user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = batch_data

            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)

            opp = opp.to(device)
            acc_prior = acc_prior.to(device)
            stability = stability.to(device)

            pred, _ = self.ncdm_net(
                user_id,
                item_id,
                knowledge_emb,
                opp,
                acc_prior,
                stability,
                return_aux=False
            )

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        # ===== binary =====
        y_true_bin = (np.array(y_true) >= 0.5).astype(int)
        y_pred_bin = (np.array(y_pred) >= threshold).astype(int)

        # ===== metrics =====
        auc = roc_auc_score(y_true_bin, y_pred)
        acc = accuracy_score(y_true_bin, y_pred_bin)
        rmse = np.sqrt(mean_squared_error(y_true_bin, y_pred))
        f1 = f1_score(y_true_bin, y_pred_bin)

        with open(self.log_name, 'a', encoding='utf-8') as f:
            f.write(
                "Test auc: %.6f, accuracy: %.6f, rmse: %.6f, f1: %.6f\n"
                % (auc, acc, rmse, f1)
            )

        return auc, acc, rmse, f1

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath, map_location=None):
        self.ncdm_net.load_state_dict(torch.load(filepath, map_location=map_location))
        logging.info("load parameters from %s" % filepath)

    def get_embeddings(self):
        return self.ncdm_net.get_embeddings()


# =========================
# ASSIST09 Dataset Adapter
# =========================
from torch.utils.data import Dataset, DataLoader



class KCD_Channel(CDM):
    """KCD variant with ChannelGate."""
    def __init__(self, knowledge_n, exer_n, student_n,
                 log_name, best_model,
                 usr_emb, itm_emb, similar_students, similar_exercises,
                 lambda_kcd=1.0, lambda_opp=0.1, lambda_acc=0.05, lambda_sta=0.05,
                 flu_latent_dim=16, dropout=0.5,
                 gate_alpha=0.5, gate_hidden_ratio=0.5):
        super(KCD_Channel, self).__init__()
        self.ncdm_net = KCDNetWithChannel(
            knowledge_n, exer_n, student_n,
            usr_emb, itm_emb, similar_students, similar_exercises,
            flu_latent_dim=flu_latent_dim, dropout=dropout,
            gate_alpha=gate_alpha, gate_hidden_ratio=gate_hidden_ratio)
        self.log_name = log_name
        self.best_model = best_model
        self.lambda_kcd = float(lambda_kcd)
        self.lambda_opp = float(lambda_opp)
        self.lambda_acc = float(lambda_acc)
        self.lambda_sta = float(lambda_sta)
        self.loss_y = nn.BCELoss()
        self.loss_opp = nn.MSELoss()
        self.loss_acc = nn.SmoothL1Loss()
        self.loss_sta = nn.SmoothL1Loss()

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        last_time = time.time()
        best_auc = 0.0
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        for epoch_i in range(epoch):
            epoch_losses = []
            ly_list, lkcd_list, lopp_list, lacc_list, lsta_list = [], [], [], [], []
            for batch_data in tqdm(train_data, f"Epoch {epoch_i}", disable=silence):
                user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = batch_data
                user_id = user_id.to(device); item_id = item_id.to(device)
                knowledge_emb = knowledge_emb.to(device)
                y = (y > 0.5).float().to(device)
                opp = opp.to(device); acc_prior = acc_prior.to(device); stability = stability.to(device)
                out = self.ncdm_net(user_id, item_id, knowledge_emb, opp, acc_prior, stability, return_aux=True)
                ly = self.loss_y(out["p"], y)
                lopp = self.loss_opp(out["opp_hat"], out["opp_tgt"])
                lacc = self.loss_acc(out["acc_hat"], out["acc_tgt"])
                lsta = self.loss_sta(out["sta_hat"], out["sta_tgt"])
                loss = ly + self.lambda_kcd * out["kcd_loss"] + self.lambda_opp * lopp + self.lambda_acc * lacc + self.lambda_sta * lsta
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_losses.append(loss.item()); ly_list.append(ly.item())
                lkcd_list.append(out["kcd_loss"].item()); lopp_list.append(lopp.item())
                lacc_list.append(lacc.item()); lsta_list.append(lsta.item())
            now = time.time()
            msg = "[Epoch %d] loss=%.6f (y=%.6f, kcd=%.6f, opp=%.6f, acc=%.6f, sta=%.6f) time=%.6f" % (
                epoch_i, float(np.mean(epoch_losses)), float(np.mean(ly_list)),
                float(np.mean(lkcd_list)), float(np.mean(lopp_list)),
                float(np.mean(lacc_list)), float(np.mean(lsta_list)), (now - last_time))
            print(msg)
            with open(self.log_name, 'a', encoding='utf-8') as f: f.write(msg + '\n')
            last_time = now
            if test_data is not None:
                auc, accuracy, rmse, f1 = self.eval(test_data, device=device, silence=silence)
                if auc > best_auc:
                    best_auc = auc; print("Update best auc!")
                    with open(self.log_name, 'a', encoding='utf-8') as f: f.write("Update best auc!\n")
                    self.save(self.best_model)
                msg = "[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, f1: %.6f" % (epoch_i, auc, accuracy, rmse, f1)
                print(msg)
                with open(self.log_name, 'a', encoding='utf-8') as f: f.write(msg + '\n')

    @torch.no_grad()
    def eval(self, test_data, device="cpu", threshold=0.5, save=False, silence=False):
        self.ncdm_net = self.ncdm_net.to(device); self.ncdm_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating", disable=silence):
            user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = batch_data
            user_id = user_id.to(device); item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)
            opp = opp.to(device); acc_prior = acc_prior.to(device); stability = stability.to(device)
            pred, _ = self.ncdm_net(user_id, item_id, knowledge_emb, opp, acc_prior, stability, return_aux=False)
            y_pred.extend(pred.detach().cpu().tolist()); y_true.extend(y.tolist())
        y_true_bin = (np.array(y_true) >= 0.5).astype(int); y_pred_bin = (np.array(y_pred) >= threshold).astype(int)
        auc = roc_auc_score(y_true_bin, y_pred); acc = accuracy_score(y_true_bin, y_pred_bin)
        rmse = float(np.sqrt(mean_squared_error(y_true_bin, y_pred))); f1_val = f1_score(y_true_bin, y_pred_bin)
        with open(self.log_name, 'a', encoding='utf-8') as f:
            f.write("Test auc: %.6f, accuracy: %.6f, rmse: %.6f, f1: %.6f\n" % (auc, acc, rmse, f1_val))
        return auc, acc, rmse, f1_val

    def save(self, filepath): torch.save(self.ncdm_net.state_dict(), filepath); logging.info("save parameters to %s" % filepath)
    def load(self, filepath, map_location=None): self.ncdm_net.load_state_dict(torch.load(filepath, map_location=map_location)); logging.info("load parameters from %s" % filepath)
    def get_embeddings(self): return self.ncdm_net.get_embeddings()


class Assist09KCDDataset(Dataset):
    def __init__(
        self,
        csv_path,
        user_map=None,
        item_map=None,
        skill_map=None,
        user_oov_id=None,
        rt_cap_ms=120000.0,
    ):
        df = pd.read_csv(csv_path, encoding="utf-8")

        required = [
            "user_id",
            "problem_id",
            "skill_id",
            "correct",
            "opportunity",
            "ms_first_response",
            "attempt_count",
            "hint_count",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} 缺少字段: {missing}")

        df = df.dropna(subset=["user_id", "problem_id", "skill_id", "correct"]).copy()

        df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
        df["skill_id"] = pd.to_numeric(df["skill_id"], errors="coerce")
        df = df.dropna(subset=["correct", "skill_id"]).copy()

        df["correct"] = df["correct"].astype(int)
        df["skill_id"] = df["skill_id"].astype(int)

        for col in ["opportunity", "ms_first_response", "attempt_count", "hint_count"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        is_train = user_map is None or item_map is None or skill_map is None

        def build_map(values):
            uniq = pd.unique(values)
            return {v: i for i, v in enumerate(uniq)}

        if is_train:
            user_map = build_map(df["user_id"].values)
            item_map = build_map(df["problem_id"].values)
            skill_map = build_map(df["skill_id"].values)

        self.user_map = user_map
        self.item_map = item_map
        self.skill_map = skill_map

        if user_oov_id is None:
            user_oov_id = len(user_map)
        self.user_oov_id = user_oov_id

        if not is_train:
            before = len(df)
            df = df[df["problem_id"].isin(item_map)].copy()
            df = df[df["skill_id"].isin(skill_map)].copy()
            after = len(df)
            print(f"[{os.path.basename(csv_path)}] 过滤未知 item/skill: {before} -> {after}")

        df = df.reset_index(drop=True)

        # acc_prior：当前学生在当前知识点之前的历史正确率，首次默认 0.5
        group_keys = ["user_id", "skill_id"]
        prior_sum = df.groupby(group_keys)["correct"].cumsum() - df["correct"]
        prior_cnt = df.groupby(group_keys).cumcount()

        df["_acc_prior"] = np.where(
            prior_cnt.values > 0,
            prior_sum.values / np.maximum(prior_cnt.values, 1),
            0.5,
        ).astype(np.float32)

        # stability：由 rt / attempt / hint 构造不稳定度，压到 [0, 0.25]
        rt = np.clip(df["ms_first_response"].values.astype(np.float32), 0.0, float(rt_cap_ms))
        rt_norm = np.log1p(rt) / np.log1p(float(rt_cap_ms))

        att_extra = np.clip(df["attempt_count"].values.astype(np.float32) - 1.0, 0.0, None)
        att_norm = np.log1p(att_extra) / np.log1p(10.0)

        hint = np.clip(df["hint_count"].values.astype(np.float32), 0.0, None)
        hint_norm = np.log1p(hint) / np.log1p(10.0)

        instability = (rt_norm + att_norm + hint_norm) / 3.0
        df["_stability"] = (0.25 * np.clip(instability, 0.0, 1.0)).astype(np.float32)

        self.users = np.array(
            [user_map.get(v, self.user_oov_id) for v in df["user_id"].values],
            dtype=np.int64,
        )
        self.items = np.array(
            [item_map[v] for v in df["problem_id"].values],
            dtype=np.int64,
        )
        self.skills = np.array(
            [skill_map[v] for v in df["skill_id"].values],
            dtype=np.int64,
        )

        self.y = df["correct"].values.astype(np.float32)
        self.opp = df["opportunity"].values.astype(np.float32)
        self.acc_prior = df["_acc_prior"].values.astype(np.float32)
        self.stability = df["_stability"].values.astype(np.float32)

        self.student_n = len(user_map) + 1
        self.exer_n = len(item_map)
        self.knowledge_n = len(skill_map)

        print(
            f"[Dataset] {os.path.basename(csv_path)} "
            f"samples={len(self.y)}, students={self.student_n}, "
            f"items={self.exer_n}, skills={self.knowledge_n}"
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
            torch.tensor(self.acc_prior[idx], dtype=torch.float32),
            torch.tensor(self.stability[idx], dtype=torch.float32),
        )


def make_neighbor_dict(n, topk=10):
    result = {}
    for i in range(n):
        result[i] = [(i + j) % n for j in range(topk)]
    return result


# =========================
# Main for ASSIST09
# =========================




if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-e", "--epoch", default=40, type=int)
    argparser.add_argument("-b", "--batch-size", default=256, type=int)
    argparser.add_argument("-g", "--gpu", type=str, default="cuda")
    argparser.add_argument("--lambda-kcd", default=1.0, type=float)
    argparser.add_argument("--lambda-opp", default=0.1, type=float)
    argparser.add_argument("--lambda-acc", default=0.05, type=float)
    argparser.add_argument("--lambda-sta", default=0.05, type=float)
    argparser.add_argument("--flu-latent-dim", default=16, type=int)
    argparser.add_argument("--dropout", default=0.5, type=float)
    argparser.add_argument("--gate-alpha", default=0.5, type=float)
    argparser.add_argument("--gate-hidden-ratio", default=0.5, type=float)
    args = argparser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    train_csv = os.path.join(base_dir, "data", "assist09", "splits", "assist09_train.csv")
    test_csv = os.path.join(base_dir, "data", "assist09", "splits", "assist09_test.csv")
    check_file(train_csv); check_file(test_csv)
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    log_dir = os.path.join(output_dir, f"KCD_vs_Channel_{current_time}")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "comparison.log")
    logger.set_filename(log_file)

    train_ds = Assist09KCDDataset(train_csv)
    test_ds = Assist09KCDDataset(test_csv, user_map=train_ds.user_map, item_map=train_ds.item_map,
                                 skill_map=train_ds.skill_map, user_oov_id=train_ds.user_oov_id)
    train_set = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_set = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    user_n = train_ds.student_n; item_n = train_ds.exer_n; knowledge_n = train_ds.knowledge_n
    log(f"user_n={user_n}, item_n={item_n}, knowledge_n={knowledge_n}")

    emb_dim = 128; rng = np.random.default_rng(2026)
    usr_emb = rng.normal(0, 0.01, size=(user_n, emb_dim)).astype(np.float32)
    itm_emb = rng.normal(0, 0.01, size=(item_n, emb_dim)).astype(np.float32)
    similar_students = make_neighbor_dict(user_n, topk=10)
    similar_exercises = make_neighbor_dict(item_n, topk=10)

    lr_list = [5e-4, 1e-3, 2e-3]
    model_names = ["KCD_Fluency", "KCD_Channel"]
    results = {n: {} for n in model_names}

    for lr in lr_list:
        for model_name in model_names:
            log(f"\n===== Training {model_name} with lr={lr} =====")
            bp = os.path.join(log_dir, f"{model_name}_lr{lr}_best.pt")

            if model_name == "KCD_Fluency":
                cdm = KCD_Fluency(knowledge_n=knowledge_n, exer_n=item_n, student_n=user_n,
                    log_name=log_file, best_model=bp, usr_emb=usr_emb, itm_emb=itm_emb,
                    similar_students=similar_students, similar_exercises=similar_exercises,
                    lambda_kcd=args.lambda_kcd, lambda_opp=args.lambda_opp,
                    lambda_acc=args.lambda_acc, lambda_sta=args.lambda_sta,
                    flu_latent_dim=args.flu_latent_dim, dropout=args.dropout)
            else:
                cdm = KCD_Channel(knowledge_n=knowledge_n, exer_n=item_n, student_n=user_n,
                    log_name=log_file, best_model=bp, usr_emb=usr_emb, itm_emb=itm_emb,
                    similar_students=similar_students, similar_exercises=similar_exercises,
                    lambda_kcd=args.lambda_kcd, lambda_opp=args.lambda_opp,
                    lambda_acc=args.lambda_acc, lambda_sta=args.lambda_sta,
                    flu_latent_dim=args.flu_latent_dim, dropout=args.dropout,
                    gate_alpha=args.gate_alpha, gate_hidden_ratio=args.gate_hidden_ratio)

            cdm.train(train_set, test_set, epoch=args.epoch, device=device, lr=lr)
            if os.path.exists(bp): cdm.load(bp, map_location=device)
            auc, acc, rmse, f1_val = cdm.eval(test_set, device=device)
            results[model_name][lr] = {"AUC": auc, "ACC": acc, "RMSE": rmse, "F1": f1_val}
            log(f"[{model_name}] lr={lr} -> AUC={auc:.6f}, ACC={acc:.6f}, RMSE={rmse:.6f}, F1={f1_val:.6f}")

    sep = "=" * 60
    log("\n" + sep + "\n MODEL COMPARISON SUMMARY" + "\n" + sep)
    log(f"{'LR':<12} {'Model':<20} {'AUC':<10} {'ACC':<10} {'F1':<10} {'RMSE':<10}")
    log("-" * 60)
    for mn in model_names:
        for lr in results[mn]:
            r = results[mn][lr]
            log(f"{lr:<12} {mn:<20} {r['AUC']:<10.6f} {r['ACC']:<10.6f} {r['F1']:<10.6f} {r['RMSE']:<10.6f}")
    log(sep)

    for mn in model_names:
        bl = max(results[mn], key=lambda l: results[mn][l]["AUC"])
        r = results[mn][bl]
        log(f"Best {mn}: lr={bl}, AUC={r['AUC']:.6f}, ACC={r['ACC']:.6f}, RMSE={r['RMSE']:.6f}, F1={r['F1']:.6f}")

    csv_path = os.path.join(log_dir, "kcd_comparison.csv")
    rows = [{"Model": mn, "LR": lr, **results[mn][lr]} for mn in model_names for lr in results[mn]]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log(f"CSV saved to {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(lr_list)); width = 0.35
    for i, metric in enumerate(["AUC", "RMSE"]):
        ax = axes[i]
        for j, mn in enumerate(model_names):
            vals = [results[mn][lr][metric] for lr in lr_list]
            ax.bar(x + j * width, vals, width, label=mn)
        ax.set_xlabel("Learning Rate"); ax.set_ylabel(metric)
        ax.set_title(f"{metric} Comparison")
        ax.set_xticks(x + width / 2); ax.set_xticklabels([str(lr) for lr in lr_list])
        ax.legend()
    chart_path = os.path.join(log_dir, "kcd_comparison.png")
    plt.tight_layout(); plt.savefig(chart_path, dpi=150); plt.close()
    log(f"Chart saved to {chart_path}")
    logger.close()
