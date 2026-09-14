# coding: utf-8
# Neural DINA + ASSIST09 Fluency

import logging
import numpy as np
import torch
from EduCDM import CDM
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.autograd as autograd
import torch.nn.functional as F


class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        return STEFunction.apply(x)


class DINANetWithAssist09Fluency(nn.Module):
    """
    NeuralDINA + ASSIST09 Fluency.

    Inputs:
        user
        item
        knowledge
        opportunity
        ms_first_response
        attempt_count
        hint_count
    """

    def __init__(
        self,
        user_num,
        item_num,
        hidden_dim,
        max_slip=0.4,
        max_guess=0.4,
        dropout=0.5,
        flu_latent_dim=16,
        gamma_init=1.0,
        fluency_lambda=0.5,
        rt_cap_ms=120000.0,
        ste=False,
    ):
        super(DINANetWithAssist09Fluency, self).__init__()

        self._user_num = user_num
        self._item_num = item_num
        self.hidden_dim = hidden_dim

        self.step = 0
        self.max_step = 1000
        self.max_slip = max_slip
        self.max_guess = max_guess
        self.ste = ste

        self.fluency_lambda = float(fluency_lambda)
        self.rt_cap_ms = float(rt_cap_ms)

        self.guess = nn.Embedding(self._item_num, 1)
        self.slip = nn.Embedding(self._item_num, 1)
        self.theta = nn.Embedding(self._user_num, hidden_dim)

        if self.ste:
            self.sign = StraightThroughEstimator()

        self.beh_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, flu_latent_dim),
            nn.ReLU(),
        )

        self.student_flu_bias = nn.Embedding(self._user_num, flu_latent_dim)

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

    @staticmethod
    def _to_col(t):
        return t.view(-1, 1) if t.dim() == 1 else t

    @staticmethod
    def _safe_log1p(x):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=0.0)
        return torch.log1p(x)

    @staticmethod
    def _safe_logit(p, eps=1e-6):
        p = torch.clamp(p, eps, 1.0 - eps)
        return torch.log(p) - torch.log(1.0 - p)

    def _base_forward_prob(self, user, item, knowledge):
        if self.ste:
            theta = self.sign(self.theta(user))
            slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
            guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)

            mask_theta = (knowledge == 0).float() + (knowledge == 1).float() * theta
            n = torch.prod((mask_theta + 1) / 2, dim=-1)
            base_p = torch.pow(1 - slip, n) * torch.pow(guess, 1 - n)
            return base_p, slip, guess

        theta = self.theta(user)
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)

        if self.training:
            n = torch.sum(knowledge * (torch.sigmoid(theta) - 0.5), dim=1)
            t = max((np.sin(2 * np.pi * self.step / self.max_step) + 1) / 2 * 100, 1e-6)
            self.step = self.step + 1 if self.step < self.max_step else 0

            base_p = torch.sum(
                torch.stack([1 - slip, guess]).T
                * torch.softmax(torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1),
                dim=1,
            )
        else:
            n = torch.prod(knowledge * (theta >= 0).float() + (1 - knowledge), dim=1)
            base_p = (1 - slip) ** n * guess ** (1 - n)

        return base_p, slip, guess

    def forward(
        self,
        user,
        item,
        knowledge,
        opportunity,
        ms_first_response,
        attempt_count,
        hint_count,
        return_aux=False,
    ):
        user = torch.clamp(user, 0, self._user_num - 1)
        item = torch.clamp(item, 0, self._item_num - 1)

        base_p, slip, guess = self._base_forward_prob(user, item, knowledge)
        base_logit = self._safe_logit(base_p)

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

        gate_in = torch.cat(
            [
                z,
                guess.view(-1, 1),
                slip.view(-1, 1),
            ],
            dim=1,
        )

        gate = torch.sigmoid(self.fuse_gate(gate_in)).view(-1)
        gamma = F.softplus(self.gamma)
        delta = (flu - 0.5) * 4.0

        p_logit = base_logit + gamma * gate * delta
        p = torch.sigmoid(p_logit)

        if not return_aux:
            return p

        return {
            "p": p,
            "o": base_p,
            "flu": flu,
            "gate": gate,
            "opp_hat": opp_hat,
            "rt_hat": rt_hat,
            "att_hat": att_hat,
            "hint_hat": hint_hat,
            "opp_tgt": f_opp.view(-1),
            "rt_tgt": f_rt.view(-1),
            "att_tgt": f_att.view(-1),
            "hint_tgt": f_hint.view(-1),
        }


class DINAFluency2(CDM):
    def __init__(
        self,
        user_num,
        item_num,
        hidden_dim,
        ste=False,
        lr=0.001,
        max_slip=0.4,
        max_guess=0.4,
        dropout=0.5,
        fluency_lambda=0.5,
        lambda_opp=0.1,
        lambda_rt=0.05,
        lambda_att=0.05,
        lambda_hint=0.05,
        flu_latent_dim=16,
        gamma_init=1.0,
        rt_cap_ms=120000.0,
    ):
        super(DINAFluency2, self).__init__()

        self.lr = lr
        self.lambda_opp = float(lambda_opp)
        self.lambda_rt = float(lambda_rt)
        self.lambda_att = float(lambda_att)
        self.lambda_hint = float(lambda_hint)

        self.dina_net = DINANetWithAssist09Fluency(
            user_num=user_num,
            item_num=item_num,
            hidden_dim=hidden_dim,
            max_slip=max_slip,
            max_guess=max_guess,
            dropout=dropout,
            flu_latent_dim=flu_latent_dim,
            gamma_init=gamma_init,
            fluency_lambda=fluency_lambda,
            rt_cap_ms=rt_cap_ms,
            ste=ste,
        )

        self.loss_y = nn.BCELoss()
        self.loss_opp = nn.MSELoss()
        self.loss_rt = nn.SmoothL1Loss()
        self.loss_att = nn.SmoothL1Loss()
        self.loss_hint = nn.SmoothL1Loss()

    def train(self, train_data, test_data=None, *, epoch=10, device="cpu", lr=None, silence=False):
        self.dina_net = self.dina_net.to(device)
        self.dina_net.train()

        optimizer = torch.optim.Adam(self.dina_net.parameters(), lr=(lr or self.lr))

        for e in range(epoch):
            losses = []

            for batch_data in tqdm(train_data, f"Epoch {e}", disable=silence):
                user_id, item_id, knowledge, response, opp, rt, attempt, hint = batch_data

                user_id = user_id.to(device)
                item_id = item_id.to(device)
                knowledge = knowledge.to(device)
                response = response.to(device).float()

                opp = opp.to(device)
                rt = rt.to(device)
                attempt = attempt.to(device)
                hint = hint.to(device)

                out = self.dina_net(
                    user_id,
                    item_id,
                    knowledge,
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
                auc, accuracy = self.eval(test_data, device=device, silence=silence)
                print(f"[Epoch {e}] auc: {auc:.6f}, accuracy: {accuracy:.6f}")

    @torch.no_grad()
    def eval(self, test_data, device="cpu", silence=False):
        self.dina_net = self.dina_net.to(device)
        self.dina_net.eval()

        y_pred = []
        y_true = []

        for batch_data in tqdm(test_data, "evaluating", disable=silence):
            user_id, item_id, knowledge, response, opp, rt, attempt, hint = batch_data

            pred = self.dina_net(
                user_id.to(device),
                item_id.to(device),
                knowledge.to(device),
                opp.to(device),
                rt.to(device),
                attempt.to(device),
                hint.to(device),
                return_aux=False,
            )

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(response.detach().cpu().tolist())

        self.dina_net.train()

        if len(set(y_true)) < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(y_true, y_pred)

        acc = accuracy_score(y_true, np.array(y_pred) >= 0.5)
        return auc, acc

    def save(self, filepath):
        torch.save(self.dina_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath, map_location=None):
        self.dina_net.load_state_dict(torch.load(filepath, map_location=map_location))
        logging.info("load parameters from %s" % filepath)