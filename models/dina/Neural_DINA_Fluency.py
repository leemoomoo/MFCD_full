# coding: utf-8
# Neural DINA + Fluency

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
        x = STEFunction.apply(x)
        return x


class DINANetWithFluency(nn.Module):
    """
    NeuralDINA + Fluency Proxy

    Main task:
        - predict correctness p(correct)

    Fluency inputs:
        - opportunity
        - acc_prior
        - stability

    Fusion:
        p_logit = base_logit + gamma * gate * delta(flu)
    """

    def __init__(self,
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
                 ste=False):
        super(DINANetWithFluency, self).__init__()

        self._user_num = user_num
        self._item_num = item_num
        self.hidden_dim = hidden_dim

        self.step = 0
        self.max_step = 1000
        self.max_slip = max_slip
        self.max_guess = max_guess
        self.ste = ste

        self.fluency_lambda = float(fluency_lambda)   # compatibility only
        self.rt_cap_ms = float(rt_cap_ms)             # compatibility only

        # ---------- Neural DINA backbone ----------
        self.guess = nn.Embedding(self._item_num, 1)
        self.slip = nn.Embedding(self._item_num, 1)
        self.theta = nn.Embedding(self._user_num, hidden_dim)

        if self.ste:
            self.sign = StraightThroughEstimator()

        # ---------- Fluency encoder ----------
        self.beh_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, flu_latent_dim),
            nn.ReLU(),
        )

        # student-level fluency baseline
        self.student_flu_bias = nn.Embedding(self._user_num, flu_latent_dim)

        # scalar fluency in (0,1)
        self.flu_score_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # ---------- auxiliary heads ----------
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

        # ---------- gate ----------
        # use z + item slip/guess as item context
        self.fuse_gate = nn.Sequential(
            nn.Linear(flu_latent_dim + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        # init
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

    @staticmethod
    def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = torch.clamp(p, eps, 1.0 - eps)
        return torch.log(p) - torch.log(1.0 - p)

    def _base_forward_prob(self, user, item, knowledge):
        """
        Original NeuralDINA probability output.
        """
        if self.ste:
            theta = self.sign(self.theta(user))
            slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
            guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)
            mask_theta = (knowledge == 0).float() + (knowledge == 1).float() * theta
            n = torch.prod((mask_theta + 1) / 2, dim=-1)
            base_p = torch.pow(1 - slip, n) * torch.pow(guess, 1 - n)
            return base_p, slip, guess
        else:
            theta = self.theta(user)
            slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
            guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)

            if self.training:
                n = torch.sum(knowledge * (torch.sigmoid(theta) - 0.5), dim=1)
                t = max((np.sin(2 * np.pi * self.step / self.max_step) + 1) / 2 * 100, 1e-6)
                self.step = self.step + 1 if self.step < self.max_step else 0
                base_p = torch.sum(
                    torch.stack([1 - slip, guess]).T *
                    torch.softmax(torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1),
                    dim=1
                )
            else:
                n = torch.prod(knowledge * (theta >= 0).float() + (1 - knowledge), dim=1)
                base_p = (1 - slip) ** n * guess ** (1 - n)

            return base_p, slip, guess

    def forward(self,
                user,
                item,
                knowledge,
                opportunity,
                acc_prior,
                stability,
                return_aux: bool = False):

        # ===================== (1) base NeuralDINA =====================
        base_p, slip, guess = self._base_forward_prob(user, item, knowledge)
        base_logit = self._safe_logit(base_p)

        # ===================== (2) fluency proxy features =====================
        opp = self._to_col(opportunity.float())
        acc = self._to_col(acc_prior.float())
        sta = self._to_col(stability.float())

        f_opp = self._safe_log1p(opp)
        f_acc = torch.clamp(acc, 0.0, 1.0)
        # assume variance-like input -> convert to higher-is-better stability
        f_sta = 1.0 - torch.clamp(4.0 * torch.clamp(sta, min=0.0), 0.0, 1.0)

        beh = torch.cat([f_opp, f_acc, f_sta], dim=1)

        # ===================== (3) latent z + fluency score =====================
        z = self.beh_encoder(beh)
        z = z + torch.tanh(self.student_flu_bias(user))
        flu = torch.sigmoid(self.flu_score_head(z)).view(-1)

        # ===================== (4) auxiliary predictions =====================
        opp_tgt = f_opp.view(-1)
        acc_tgt = f_acc.view(-1)
        sta_tgt = f_sta.view(-1)

        opp_hat = self.opp_head(z).view(-1)
        acc_hat = torch.sigmoid(self.acc_head(z)).view(-1)
        sta_hat = torch.sigmoid(self.sta_head(z)).view(-1)

        # ===================== (5) gate + logit fusion =====================
        gate_in = torch.cat([
            z,
            guess.view(-1, 1),
            slip.view(-1, 1),
        ], dim=1)

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
            "acc_hat": acc_hat,
            "sta_hat": sta_hat,
            "opp_tgt": opp_tgt,
            "acc_tgt": acc_tgt,
            "sta_tgt": sta_tgt,
        }


class DINAFluency(CDM):
    def __init__(self,
                 user_num,
                 item_num,
                 hidden_dim,
                 ste=False,
                 lr=0.001,
                 max_slip=0.4,
                 max_guess=0.4,
                 dropout=0.5,
                 fluency_lambda=0.5,
                 lambda_rt=0.1,
                 lambda_hint=0.05,
                 lambda_att=0.05,
                 flu_latent_dim=16,
                 gamma_init=1.0,
                 rt_cap_ms=120000.0):
        super(DINAFluency, self).__init__()

        self.lr = lr
        self.lambda_rt = float(lambda_rt)       # opp aux
        self.lambda_hint = float(lambda_hint)   # acc aux
        self.lambda_att = float(lambda_att)     # sta aux

        self.dina_net = DINANetWithFluency(
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
        self.loss_acc = nn.SmoothL1Loss()
        self.loss_sta = nn.SmoothL1Loss()

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=None, silence=False):
        self.dina_net = self.dina_net.to(device)
        self.dina_net.train()

        optimizer = torch.optim.Adam(self.dina_net.parameters(), lr=(lr or self.lr))

        for e in range(epoch):
            losses = []
            ly_list, lopp_list, lacc_list, lsta_list = [], [], [], []

            for batch_data in tqdm(train_data, f"Epoch {e}", disable=silence):
                # expected:
                # user_id, item_id, knowledge, response, opp, acc_prior, stability
                user_id, item_id, knowledge, response, opp, acc_prior, stability = batch_data

                user_id = user_id.to(device)
                item_id = item_id.to(device)
                knowledge = knowledge.to(device)
                response = response.to(device).float()

                opp = opp.to(device)
                acc_prior = acc_prior.to(device)
                stability = stability.to(device)

                out = self.dina_net(
                    user_id,
                    item_id,
                    knowledge,
                    opp,
                    acc_prior,
                    stability,
                    return_aux=True,
                )

                ly = self.loss_y(out["p"], response)
                lopp = self.loss_opp(out["opp_hat"], out["opp_tgt"])
                lacc = self.loss_acc(out["acc_hat"], out["acc_tgt"])
                lsta = self.loss_sta(out["sta_hat"], out["sta_tgt"])

                loss = ly + self.lambda_rt * lopp + self.lambda_hint * lacc + self.lambda_att * lsta

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                ly_list.append(ly.item())
                lopp_list.append(lopp.item())
                lacc_list.append(lacc.item())
                lsta_list.append(lsta.item())

            print(
                f"[Epoch {e}] loss={float(np.mean(losses)):.6f} "
                f"(y={float(np.mean(ly_list)):.6f}, opp={float(np.mean(lopp_list)):.6f}, "
                f"acc={float(np.mean(lacc_list)):.6f}, sta={float(np.mean(lsta_list)):.6f})"
            )

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device, silence=silence)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (e, auc, accuracy))

    @torch.no_grad()
    def eval(self, test_data, device="cpu", silence=False):
        self.dina_net = self.dina_net.to(device)
        self.dina_net.eval()

        y_pred = []
        y_true = []

        for batch_data in tqdm(test_data, "evaluating", disable=silence):
            user_id, item_id, knowledge, response, opp, acc_prior, stability = batch_data

            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge = knowledge.to(device)

            opp = opp.to(device)
            acc_prior = acc_prior.to(device)
            stability = stability.to(device)

            pred = self.dina_net(
                user_id,
                item_id,
                knowledge,
                opp,
                acc_prior,
                stability,
                return_aux=False,
            )

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(response.detach().cpu().tolist())

        self.dina_net.train()
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.dina_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath, map_location=None):
        self.dina_net.load_state_dict(torch.load(filepath, map_location=map_location))
        logging.info("load parameters from %s" % filepath)