# coding: utf-8
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from EduCDM import CDM


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class NetNCDMProxy(nn.Module):
    """
    Baseline:
      Original NCDM backbone + direct proxy fusion

    Inputs:
      - stu_id, item_id, q_vec
      - opportunity, acc_prior, stability

    Design:
      1) Keep original NCDM backbone unchanged
      2) Encode 3 handcrafted proxy features with a shallow MLP
      3) Add proxy correction directly in logit space
    """

    def __init__(self, knowledge_n, exer_n, student_n,
                 prednet_len1=512, prednet_len2=256,
                 dropout=0.5, proxy_hidden=32):
        super(NetNCDMProxy, self).__init__()

        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim

        # ---------- original NCDM backbone ----------
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)

        self.prednet_full1 = PosLinear(self.knowledge_dim, prednet_len1)
        self.drop_1 = nn.Dropout(p=dropout)
        self.prednet_full2 = PosLinear(prednet_len1, prednet_len2)
        self.drop_2 = nn.Dropout(p=dropout)
        self.prednet_full3 = PosLinear(prednet_len2, 1)

        # ---------- simple proxy branch ----------
        self.proxy_mlp = nn.Sequential(
            nn.Linear(3, proxy_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(proxy_hidden, 1)
        )

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    @staticmethod
    def _safe_log1p(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=0.0)
        return torch.log1p(x)

    @staticmethod
    def _to_col(t: torch.Tensor) -> torch.Tensor:
        return t.view(-1, 1) if t.dim() == 1 else t

    def forward(self, stu_id, input_exercise, input_knowledge_point,
                opportunity, acc_prior, stability,
                return_aux: bool = False):

        # safety clamp
        stu_id = torch.clamp(stu_id, 0, self.emb_num - 1)
        input_exercise = torch.clamp(input_exercise, 0, self.exer_n - 1)

        # ---------- original NCDM ----------
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))

        x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point.float()
        x = self.drop_1(torch.sigmoid(self.prednet_full1(x)))
        x = self.drop_2(torch.sigmoid(self.prednet_full2(x)))
        o_logit = self.prednet_full3(x)                # (B,1)
        o = torch.sigmoid(o_logit)                     # (B,1)

        # ---------- direct proxy fusion ----------
        opp = self._to_col(opportunity.float())
        acc = self._to_col(acc_prior.float())
        sta = self._to_col(stability.float())

        # same normalization as your fluency model
        f_opp = self._safe_log1p(opp)
        f_acc = torch.clamp(acc, 0.0, 1.0)
        f_sta = 1.0 - torch.clamp(4.0 * torch.clamp(sta, min=0.0), 0.0, 1.0)

        proxy_feat = torch.cat([f_opp, f_acc, f_sta], dim=1)   # (B,3)
        delta = self.proxy_mlp(proxy_feat)                     # (B,1)

        p_logit = o_logit + delta
        p = torch.sigmoid(p_logit)

        if not return_aux:
            return p.view(-1)

        return {
            "p": p.view(-1),
            "o": o.view(-1),
            "delta": delta.view(-1),
        }


class NCDMProxy(CDM):
    """
    Comparison baseline:
      Original NCDM + direct handcrafted proxy fusion
    """

    def __init__(self, knowledge_n, exer_n, student_n,
                 lr=0.002, dropout=0.5, proxy_hidden=32):
        super(NCDMProxy, self).__init__()
        self.lr = lr
        self.net = NetNCDMProxy(
            knowledge_n=knowledge_n,
            exer_n=exer_n,
            student_n=student_n,
            dropout=dropout,
            proxy_hidden=proxy_hidden
        )
        self.loss_y = nn.BCELoss()

    def train(self, train_data, test_data=None, epoch=10, device="cuda", lr=None, silence=False):
        self.net = self.net.to(device)
        self.net.train()
        optimizer = optim.Adam(self.net.parameters(), lr=(lr or self.lr))

        for epoch_i in range(epoch):
            losses = []

            for batch_data in tqdm(train_data, f"Epoch {epoch_i}", disable=silence):
                user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = batch_data

                user_id = user_id.to(device)
                item_id = item_id.to(device)
                knowledge_emb = knowledge_emb.to(device)
                y = y.to(device).float()
                opp = opp.to(device)
                acc_prior = acc_prior.to(device)
                stability = stability.to(device)

                pred = self.net(
                    user_id, item_id, knowledge_emb,
                    opp, acc_prior, stability,
                    return_aux=False
                )

                loss = self.loss_y(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            print(f"[Epoch {epoch_i}] loss={float(np.mean(losses)):.6f}")

            if test_data is not None:
                auc, acc = self.eval(test_data, device=device, silence=silence)
                print(f"[Epoch {epoch_i}] auc: {auc:.6f}, accuracy: {acc:.6f}")

    @torch.no_grad()
    def eval(self, test_data, device="cpu", silence=False):
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []

        for batch_data in tqdm(test_data, "Evaluating", disable=silence):
            user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = batch_data

            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)
            opp = opp.to(device)
            acc_prior = acc_prior.to(device)
            stability = stability.to(device)

            pred = self.net(
                user_id, item_id, knowledge_emb,
                opp, acc_prior, stability,
                return_aux=False
            )

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath, map_location=None):
        self.net.load_state_dict(torch.load(filepath, map_location=map_location))
        logging.info("load parameters from %s" % filepath)