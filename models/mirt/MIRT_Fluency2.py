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


def irt2pl_logit(theta, a, b):
    """
    MIRT 2PL logit:
        logit = a^T theta - b
    """
    return torch.sum(a * theta, dim=-1) - b


class MIRTNetWithAssist09Fluency(nn.Module):
    """
    MIRT + ASSIST09 behavior fluency.

    Inputs:
        user
        item
        opportunity
        ms_first_response
        attempt_count
        hint_count
    """

    def __init__(
        self,
        user_num,
        item_num,
        latent_dim,
        a_range=None,
        dropout=0.5,
        flu_latent_dim=16,
        gamma_init=1.0,
        fluency_lambda=0.5,
        rt_cap_ms=120000.0,
        use_fluency=True,
    ):
        super(MIRTNetWithAssist09Fluency, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.a_range = a_range
        self.fluency_lambda = float(fluency_lambda)
        self.rt_cap_ms = float(rt_cap_ms)
        self.use_fluency = use_fluency

        self.theta = nn.Embedding(user_num, latent_dim)
        self.a = nn.Embedding(item_num, latent_dim)
        self.b = nn.Embedding(item_num, 1)

        self.beh_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, flu_latent_dim),
            nn.ReLU(),
        )

        self.student_flu_bias = nn.Embedding(user_num, flu_latent_dim)

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
            nn.Linear(flu_latent_dim + 3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    @staticmethod
    def _to_col(t):
        return t.view(-1, 1) if t.dim() == 1 else t

    @staticmethod
    def _safe_log1p(x):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=0.0)
        return torch.log1p(x)

    def _constrain_a(self, a):
        if self.a_range is not None:
            return self.a_range * torch.sigmoid(a)
        return F.softplus(a)

    def forward(
        self,
        user,
        item,
        opportunity,
        ms_first_response,
        attempt_count,
        hint_count,
        return_aux=False,
    ):
        user = torch.clamp(user, 0, self.user_num - 1)
        item = torch.clamp(item, 0, self.item_num - 1)

        theta = self.theta(user)
        a = self._constrain_a(self.a(item))
        b = self.b(item).view(-1)

        if torch.isnan(theta).any() or torch.isnan(a).any() or torch.isnan(b).any():
            raise ValueError("theta, a, or b contains NaN.")

        o_logit = irt2pl_logit(theta, a, b)
        o = torch.sigmoid(o_logit)
        if not self.use_fluency:
            return o.view(-1)

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

        beh = torch.cat(
            [
                f_opp,
                1.0 - f_rt,
                1.0 - f_att,
                1.0 - f_hint,
            ],
            dim=1,
        )

        z = self.beh_encoder(beh)
        z = z + torch.tanh(self.student_flu_bias(user))

        flu = torch.sigmoid(self.flu_score_head(z)).view(-1)

        opp_hat = self.opp_head(z).view(-1)
        rt_hat = torch.sigmoid(self.rt_head(z)).view(-1)
        att_hat = torch.sigmoid(self.att_head(z)).view(-1)
        hint_hat = torch.sigmoid(self.hint_head(z)).view(-1)

        opp_tgt = f_opp.view(-1)
        rt_tgt = f_rt.view(-1)
        att_tgt = f_att.view(-1)
        hint_tgt = f_hint.view(-1)

        a_mean = a.mean(dim=1, keepdim=True)
        a_norm = torch.norm(a, p=2, dim=1, keepdim=True)
        b_col = b.view(-1, 1)

        gate_in = torch.cat([z, a_mean, a_norm, b_col], dim=1)
        gate = torch.sigmoid(self.fuse_gate(gate_in)).view(-1)

        gamma = F.softplus(self.gamma)
        delta = (flu - 0.5) * 4.0

        p_logit = o_logit + gamma * gate * delta
        p = torch.sigmoid(p_logit)

        if not return_aux:
            return p

        return {
            "p": p,
            "o": o,
            "flu": flu,
            "gate": gate,
            "opp_hat": opp_hat,
            "rt_hat": rt_hat,
            "att_hat": att_hat,
            "hint_hat": hint_hat,
            "opp_tgt": opp_tgt,
            "rt_tgt": rt_tgt,
            "att_tgt": att_tgt,
            "hint_tgt": hint_tgt,
        }


class MIRTFluency(CDM):
    def __init__(
        self,
        user_num,
        item_num,
        latent_dim=16,
        a_range=1.0,
        lr=0.001,
        fluency_lambda=0.5,
        dropout=0.5,
        lambda_opp=0.1,
        lambda_rt=0.05,
        lambda_att=0.05,
        lambda_hint=0.05,
        flu_latent_dim=16,
        rt_cap_ms=120000.0,
        use_fluency=True,
    ):
        super(MIRTFluency, self).__init__()

        self.lr = lr
        self.lambda_opp = float(lambda_opp)
        self.lambda_rt = float(lambda_rt)
        self.lambda_att = float(lambda_att)
        self.lambda_hint = float(lambda_hint)

        self.irt_net = MIRTNetWithAssist09Fluency(
            user_num=user_num,
            item_num=item_num,
            latent_dim=latent_dim,
            a_range=a_range,
            dropout=dropout,
            flu_latent_dim=flu_latent_dim,
            fluency_lambda=fluency_lambda,
            rt_cap_ms=rt_cap_ms,
            use_fluency=use_fluency,
        )

        self.loss_y = nn.BCELoss()
        self.loss_opp = nn.MSELoss()
        self.loss_rt = nn.SmoothL1Loss()
        self.loss_att = nn.SmoothL1Loss()
        self.loss_hint = nn.SmoothL1Loss()

    def train(self, train_data, test_data=None, *, epoch=10, device="cpu", lr=None, silence=False):
        self.irt_net = self.irt_net.to(device)
        self.irt_net.train()

        optimizer = optim.Adam(self.irt_net.parameters(), lr=(lr or self.lr))

        for e in range(epoch):
            losses = []

            for batch_data in tqdm(train_data, f"Epoch {e}", disable=silence):
                user_id, item_id, response, opp, rt, attempt, hint = batch_data

                user_id = user_id.to(device)
                item_id = item_id.to(device)
                response = response.to(device).float()
                opp = opp.to(device)
                rt = rt.to(device)
                attempt = attempt.to(device)
                hint = hint.to(device)

                out = self.irt_net(
                    user_id,
                    item_id,
                    opp,
                    rt,
                    attempt,
                    hint,
                    return_aux=True,
                )

                ly = self.loss_y(out["p"], response)
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

            print(f"[Epoch {e}] loss: {float(np.mean(losses)):.6f}")

            if test_data is not None:
                auc, acc = self.eval(test_data, device=device, silence=silence)
                print(f"[Epoch {e}] auc: {auc:.6f}, accuracy: {acc:.6f}")

    @torch.no_grad()
    def eval(self, test_data, device="cpu", silence=False):
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()

        y_true = []
        y_pred = []

        for batch_data in tqdm(test_data, "evaluating", disable=silence):
            user_id, item_id, response, opp, rt, attempt, hint = batch_data

            pred = self.irt_net(
                user_id.to(device),
                item_id.to(device),
                opp.to(device),
                rt.to(device),
                attempt.to(device),
                hint.to(device),
                return_aux=False,
            )

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(response.detach().cpu().tolist())

        self.irt_net.train()

        if len(set(y_true)) < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(y_true, y_pred)

        acc = accuracy_score(y_true, np.array(y_pred) >= 0.5)
        return auc, acc

    def save(self, filepath):
        torch.save(self.irt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath, map_location=None):
        self.irt_net.load_state_dict(torch.load(filepath, map_location=map_location))
        logging.info("load parameters from %s" % filepath)