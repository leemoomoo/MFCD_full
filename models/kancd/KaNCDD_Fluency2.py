# coding: utf-8
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from EduCDM import CDM


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = torch.abs(self.weight)
        return F.linear(input, weight, self.bias)


class NetWithFluency(nn.Module):
    """
    KaNCDD + ASSIST09 Fluency version.

    ASSIST09 inputs:
        user_id
        problem_id
        skill_id
        correct
        opportunity
        ms_first_response
        attempt_count
        hint_count
    """

    def __init__(
        self,
        exer_n,
        student_n,
        knowledge_n,
        mf_type="gmf",
        dim=16,
        dropout=0.5,
        flu_latent_dim=16,
        gamma_init=1.0,
        fluency_lambda=0.5,
        rt_cap_ms=120000.0,
        relation_lambda=0.3,
    ):
        super(NetWithFluency, self).__init__()

        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.rt_cap_ms = float(rt_cap_ms)
        self.fluency_lambda = float(fluency_lambda)
        self.relation_lambda = float(relation_lambda)

        self.prednet_len1 = 256
        self.prednet_len2 = 128

        self.student_emb = nn.Embedding(student_n, dim)
        self.exercise_emb = nn.Embedding(exer_n, dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(knowledge_n, dim))
        self.e_discrimination = nn.Embedding(exer_n, 1)

        self.prednet_full1 = PosLinear(knowledge_n, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=dropout)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=dropout)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        if mf_type == "gmf":
            self.k_diff_full = nn.Linear(dim, 1)
            self.stat_full = nn.Linear(dim, 1)
        elif mf_type == "ncf1":
            self.k_diff_full = nn.Linear(2 * dim, 1)
            self.stat_full = nn.Linear(2 * dim, 1)
        elif mf_type == "ncf2":
            self.k_diff_full1 = nn.Linear(2 * dim, dim)
            self.k_diff_full2 = nn.Linear(dim, 1)
            self.stat_full1 = nn.Linear(2 * dim, dim)
            self.stat_full2 = nn.Linear(dim, 1)
        elif mf_type == "mf":
            pass
        else:
            raise ValueError(f"Unsupported mf_type: {mf_type}")

        self.beh_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, flu_latent_dim),
            nn.ReLU(),
        )

        self.student_flu_bias = nn.Embedding(student_n, flu_latent_dim)

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
        self.rt_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.att_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.hint_head = nn.Sequential(
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

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

    @staticmethod
    def _to_col(t: torch.Tensor) -> torch.Tensor:
        return t.view(-1, 1) if t.dim() == 1 else t

    @staticmethod
    def _safe_log1p(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=0.0)
        return torch.log1p(x)

    def _knowledge_relation(self):
        k = F.normalize(self.knowledge_emb, p=2, dim=1)
        rel = torch.sigmoid(torch.matmul(k, k.t()))
        rel = rel / torch.clamp(rel.sum(dim=1, keepdim=True), min=1e-6)
        return rel

    def _expand_knowledge_weight(self, input_knowledge_point):
        q = input_knowledge_point.float()
        rel = self._knowledge_relation()
        related_q = torch.matmul(q, rel)
        kc_weight = q + self.relation_lambda * related_q
        return torch.clamp(kc_weight, 0.0, 1.0)

    def _compute_stat_emb(self, stu_id):
        stu_emb = self.student_emb(stu_id)
        batch, dim = stu_emb.size()

        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.view(1, self.knowledge_n, dim).repeat(batch, 1, 1)

        if self.mf_type == "mf":
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1))
        elif self.mf_type == "gmf":
            stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == "ncf1":
            stat_emb = torch.sigmoid(
                self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))
            ).view(batch, -1)
        elif self.mf_type == "ncf2":
            stat_emb = torch.sigmoid(
                self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1))
            )
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)
        else:
            raise ValueError(f"Unsupported mf_type: {self.mf_type}")

        return stat_emb

    def _compute_k_difficulty(self, input_exercise):
        exer_emb = self.exercise_emb(input_exercise)
        batch, dim = exer_emb.size()

        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.view(1, self.knowledge_n, dim).repeat(batch, 1, 1)

        if self.mf_type == "mf":
            k_difficulty = torch.sigmoid((exer_emb * knowledge_emb).sum(dim=-1))
        elif self.mf_type == "gmf":
            k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == "ncf1":
            k_difficulty = torch.sigmoid(
                self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))
            ).view(batch, -1)
        elif self.mf_type == "ncf2":
            k_difficulty = torch.sigmoid(
                self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1))
            )
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
        else:
            raise ValueError(f"Unsupported mf_type: {self.mf_type}")

        return k_difficulty

    def forward(
        self,
        stu_id,
        input_exercise,
        input_knowledge_point,
        opportunity,
        ms_first_response,
        attempt_count,
        hint_count,
        return_aux=False,
    ):
        stu_id = torch.clamp(stu_id, 0, self.student_n - 1)
        input_exercise = torch.clamp(input_exercise, 0, self.exer_n - 1)

        stat_emb = self._compute_stat_emb(stu_id)
        k_difficulty = self._compute_k_difficulty(input_exercise)
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        kc_weight = self._expand_knowledge_weight(input_knowledge_point)

        input_x = e_discrimination * (stat_emb - k_difficulty) * kc_weight
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        o_logit = self.prednet_full3(input_x)
        o = torch.sigmoid(o_logit)

        opp = self._to_col(opportunity.float())
        rt = self._to_col(ms_first_response.float())
        att = self._to_col(attempt_count.float())
        hint = self._to_col(hint_count.float())

        rt = torch.clamp(rt, min=0.0, max=self.rt_cap_ms)
        att_extra = torch.clamp(att - 1.0, min=0.0)

        f_opp = self._safe_log1p(opp)
        f_rt = self._safe_log1p(rt) / np.log1p(self.rt_cap_ms)
        f_att = self._safe_log1p(att_extra) / np.log1p(10.0)
        f_hint = self._safe_log1p(hint) / np.log1p(10.0)

        f_rt = torch.clamp(f_rt, 0.0, 1.0)
        f_att = torch.clamp(f_att, 0.0, 1.0)
        f_hint = torch.clamp(f_hint, 0.0, 1.0)

        beh = torch.cat([f_opp, 1.0 - f_rt, 1.0 - f_att, 1.0 - f_hint], dim=1)

        z = self.beh_encoder(beh)
        z = z + torch.tanh(self.student_flu_bias(stu_id))
        flu = torch.sigmoid(self.flu_score_head(z))

        opp_hat = self.opp_head(z)
        rt_hat = torch.sigmoid(self.rt_head(z))
        att_hat = torch.sigmoid(self.att_head(z))
        hint_hat = torch.sigmoid(self.hint_head(z))

        gate_in = torch.cat(
            [z, e_discrimination, k_difficulty.mean(dim=1, keepdim=True)],
            dim=1,
        )
        gate = torch.sigmoid(self.fuse_gate(gate_in))

        gamma = F.softplus(self.gamma)
        delta = (flu - 0.5) * 4.0

        p_logit = o_logit + gamma * gate * delta
        p = torch.sigmoid(p_logit)

        if not return_aux:
            return p.view(-1)

        return {
            "p": p.view(-1),
            "o": o.view(-1),
            "flu": flu.view(-1),
            "gate": gate.view(-1),
            "opp_hat": opp_hat.view(-1),
            "rt_hat": rt_hat.view(-1),
            "att_hat": att_hat.view(-1),
            "hint_hat": hint_hat.view(-1),
            "opp_tgt": f_opp.view(-1),
            "rt_tgt": f_rt.view(-1),
            "att_tgt": f_att.view(-1),
            "hint_tgt": f_hint.view(-1),
        }


class KaNCDDFluency2(CDM):
    def __init__(
        self,
        exer_n,
        student_n,
        knowledge_n,
        dim=16,
        mf_type="gmf",
        lr=0.002,
        dropout=0.5,
        flu_latent_dim=16,
        fluency_lambda=0.5,
        relation_lambda=0.3,
        rt_cap_ms=120000.0,
        lambda_opp=0.1,
        lambda_rt=0.05,
        lambda_att=0.05,
        lambda_hint=0.05,
    ):
        super(KaNCDDFluency2, self).__init__()

        self.lr = lr
        self.lambda_opp = float(lambda_opp)
        self.lambda_rt = float(lambda_rt)
        self.lambda_att = float(lambda_att)
        self.lambda_hint = float(lambda_hint)

        self.net = NetWithFluency(
            exer_n=exer_n,
            student_n=student_n,
            knowledge_n=knowledge_n,
            mf_type=mf_type,
            dim=dim,
            dropout=dropout,
            flu_latent_dim=flu_latent_dim,
            fluency_lambda=fluency_lambda,
            relation_lambda=relation_lambda,
            rt_cap_ms=rt_cap_ms,
        )

        self.loss_y = nn.BCELoss()
        self.loss_opp = nn.MSELoss()
        self.loss_rt = nn.SmoothL1Loss()
        self.loss_att = nn.SmoothL1Loss()
        self.loss_hint = nn.SmoothL1Loss()

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=None, silence=False):
        self.net = self.net.to(device)
        self.net.train()

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

                out = self.net(
                    user_id,
                    item_id,
                    knowledge_emb,
                    opp,
                    rt,
                    attempt,
                    hint,
                    return_aux=True,
                )

                ly = self.loss_y(out["p"], y)
                lopp = self.loss_opp(out["opp_hat"], out["opp_tgt"])
                lrt = self.loss_rt(out["rt_hat"], out["rt_tgt"])
                latt = self.loss_att(out["att_hat"], out["att_tgt"])
                lhint = self.loss_hint(out["hint_hat"], out["hint_tgt"])

                loss = (
                    ly
                    + self.lambda_opp * lopp
                    + self.lambda_rt * lrt
                    + self.lambda_att * latt
                    + self.lambda_hint * lhint
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            print(f"[Epoch {epoch_i}] loss: {float(np.mean(losses)):.6f}")

            if test_data is not None:
                auc, acc = self.eval(test_data, device=device, silence=silence)
                print(f"[Epoch {epoch_i}] auc: {auc:.6f}, accuracy: {acc:.6f}")

    @torch.no_grad()
    def eval(self, test_data, device="cpu", silence=False):
        self.net = self.net.to(device)
        self.net.eval()

        y_true = []
        y_pred = []

        for batch_data in tqdm(test_data, "Evaluating", disable=silence):
            user_id, item_id, knowledge_emb, y, opp, rt, attempt, hint = batch_data

            pred = self.net(
                user_id.to(device),
                item_id.to(device),
                knowledge_emb.to(device),
                opp.to(device),
                rt.to(device),
                attempt.to(device),
                hint.to(device),
                return_aux=False,
            )

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.detach().cpu().tolist())

        if len(set(y_true)) < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(y_true, y_pred)

        acc = accuracy_score(y_true, np.array(y_pred) >= 0.5)
        return auc, acc

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath, map_location=None):
        self.net.load_state_dict(torch.load(filepath, map_location=map_location))
        logging.info("load parameters from %s" % filepath)