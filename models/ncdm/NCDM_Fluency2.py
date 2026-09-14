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


class NetWithFluency(nn.Module):
    def __init__(self, knowledge_n, exer_n, student_n,
                 prednet_len1=512, prednet_len2=256,
                 dropout=0.5, fluency_lambda=0.5,
                 use_fluency=True, use_behavior=True, use_gate=True,
                 use_rt=True, use_opportunity=True):
        super(NetWithFluency, self).__init__()
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n

        self.use_fluency = use_fluency
        self.use_behavior = use_behavior
        self.use_gate = use_gate
        self.use_rt = use_rt
        self.use_opportunity = use_opportunity
        self.fluency_lambda = fluency_lambda

        self.student_emb = nn.Embedding(self.emb_num, self.knowledge_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)

        # Fluency latent (conditional)
        if use_fluency:
            self.student_fluency_emb = nn.Embedding(self.emb_num, self.knowledge_dim)

        # behavior -> gate (conditional)
        if use_fluency and use_behavior and use_gate:
            self.fluency_gate_net = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        # prednet
        self.prednet_full1 = PosLinear(self.knowledge_dim, prednet_len1)
        self.drop_1 = nn.Dropout(p=dropout)
        self.prednet_full2 = PosLinear(prednet_len1, prednet_len2)
        self.drop_2 = nn.Dropout(p=dropout)
        self.prednet_full3 = PosLinear(prednet_len2, 1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    @staticmethod
    def _safe_log1p(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=0.0)
        return torch.log1p(x)

    @staticmethod
    def _to_col(t: torch.Tensor) -> torch.Tensor:
        return t.view(-1, 1) if t.dim() == 1 else t

    def _build_behavior(self, opportunity, ms_first_response, attempt_count, hint_count):
        """Build behavior feature tensor with dimension masking per ablation flags."""
        opp = self._to_col(opportunity.float())
        rt = self._to_col(ms_first_response.float())
        att = self._to_col(attempt_count.float())
        hint = self._to_col(hint_count.float())

        att_extra = torch.clamp(att - 1.0, min=0.0)

        f_opp = self._safe_log1p(opp)
        f_rt = self._safe_log1p(rt)
        f_att = self._safe_log1p(att_extra)
        f_hint = self._safe_log1p(hint)

        parts = []
        # opportunity: mask or use
        parts.append(f_opp if self.use_opportunity else torch.zeros_like(f_opp))
        # response time: mask or use (inverted)
        parts.append(-f_rt if self.use_rt else torch.zeros_like(f_rt))
        # attempt & hint: always used when use_behavior=True
        parts.append(-f_att)
        parts.append(-f_hint)

        return torch.cat(parts, dim=1)  # (B, 4)

    def forward(self, stu_id, input_exercise, input_knowledge_point,
                opportunity, ms_first_response, attempt_count, hint_count):

        # Prevent CUDA embedding out-of-bounds
        stu_id = torch.clamp(stu_id, 0, self.emb_num - 1)
        input_exercise = torch.clamp(input_exercise, 0, self.exer_n - 1)

        # Mastery term (always computed, core NCDM)
        stat_emb = torch.sigmoid(self.student_emb(stu_id))         # (B, K)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))  # (B, K)
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # (B, 1)
        mastery_term = e_difficulty * (stat_emb - k_difficulty)   # (B, K)

        # Fluency term (conditional on ablation flags)
        if self.use_fluency:
            flu_emb = torch.sigmoid(self.student_fluency_emb(stu_id))  # (B, K)

            if self.use_behavior and self.use_gate:
                # Full model: behavior features -> MLP gate
                beh = self._build_behavior(opportunity, ms_first_response, attempt_count, hint_count)
                fluency_gate = torch.sigmoid(self.fluency_gate_net(beh))  # (B, 1)
            elif self.use_behavior and not self.use_gate:
                # Behavior features but no gate: simple average as gate
                beh = self._build_behavior(opportunity, ms_first_response, attempt_count, hint_count)
                fluency_gate = torch.sigmoid(beh.mean(dim=1, keepdim=True))  # (B, 1)
            else:
                # No behavior features: gate = 1 (fixed fluency contribution)
                fluency_gate = 1.0

            fluency_term = self.fluency_lambda * fluency_gate * flu_emb  # (B, K)
        else:
            fluency_term = 0.0

        input_x = (mastery_term + fluency_term) * input_knowledge_point  # (B, K)

        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))
        return output.view(-1)


class NCDMFluency(CDM):
    def __init__(self, knowledge_n, exer_n, student_n, lr=0.002,
                 fluency_lambda=0.5, dropout=0.5,
                 use_fluency=True, use_behavior=True, use_gate=True,
                 use_rt=True, use_opportunity=True):
        super(NCDMFluency, self).__init__()
        self.lr = lr
        self.net = NetWithFluency(
            knowledge_n, exer_n, student_n,
            dropout=dropout, fluency_lambda=fluency_lambda,
            use_fluency=use_fluency, use_behavior=use_behavior,
            use_gate=use_gate, use_rt=use_rt, use_opportunity=use_opportunity,
        )

    def train(self, train_data, test_data=None, epoch=10, device="cuda", lr=None, silence=False):
        self.net = self.net.to(device)
        self.net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=(lr or self.lr))

        for epoch_i in range(epoch):
            losses = []
            for batch_data in tqdm(train_data, f"Epoch {epoch_i}", disable=silence):
                user_id, item_id, knowledge_emb, y, opp, rt, attempt, hint = batch_data

                user_id = user_id.to(device)
                item_id = item_id.to(device)
                knowledge_emb = knowledge_emb.to(device)
                y = y.to(device).float()

                opp = opp.to(device)
                rt = rt.to(device)
                attempt = attempt.to(device)
                hint = hint.to(device)

                pred = self.net(user_id, item_id, knowledge_emb, opp, rt, attempt, hint)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            print(f"[Epoch {epoch_i}] average loss: {float(np.mean(losses)):.6f}")

            if test_data is not None:
                auc, acc = self.eval(test_data, device=device, silence=silence)
                print(f"[Epoch {epoch_i}] auc: {auc:.6f}, accuracy: {acc:.6f}")

    @torch.no_grad()
    def eval(self, test_data, device="cpu", silence=False):
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating", disable=silence):
            user_id, item_id, knowledge_emb, y, opp, rt, attempt, hint = batch_data

            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)
            opp = opp.to(device)
            rt = rt.to(device)
            attempt = attempt.to(device)
            hint = hint.to(device)

            pred = self.net(user_id, item_id, knowledge_emb, opp, rt, attempt, hint)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath, map_location=None):
        self.net.load_state_dict(torch.load(filepath, map_location=map_location))
        logging.info("load parameters from %s" % filepath)
