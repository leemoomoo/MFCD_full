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
    Multidimensional 2PL logit:
        logit = a^T theta - b
    theta: (B, D)
    a:     (B, D)
    b:     (B,)
    """
    return torch.sum(a * theta, dim=-1) - b


class MIRTNetWithFluency(nn.Module):
    """
    MIRT + Fluency Proxy version

    Main task:
        - predict correctness p(correct)

    Fluency inputs (dataset-agnostic proxies):
        - opportunity: historical practice exposure
        - acc_prior: historical correctness prior
        - stability: historical performance variance / instability proxy

    Fusion:
        p_logit = mirt_logit + gamma * gate * delta(fluency)
    """

    def __init__(self,
                 user_num,
                 item_num,
                 latent_dim,
                 a_range=None,
                 dropout=0.5,
                 flu_latent_dim=16,
                 gamma_init=1.0,
                 fluency_lambda=0.5,   # compatibility only
                 irf_kwargs=None):
        super(MIRTNetWithFluency, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.a_range = a_range
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.fluency_lambda = float(fluency_lambda)  # compatibility only

        # ---------- MIRT backbone ----------
        self.theta = nn.Embedding(self.user_num, latent_dim)   # student ability
        self.a = nn.Embedding(self.item_num, latent_dim)       # item discrimination
        self.b = nn.Embedding(self.item_num, 1)                # item difficulty

        # ---------- Fluency encoder ----------
        # inputs: opportunity, acc_prior, stability
        self.beh_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, flu_latent_dim),
            nn.ReLU(),
        )

        # student-level fluency baseline
        self.student_flu_bias = nn.Embedding(self.user_num, flu_latent_dim)

        # fluency scalar in (0,1)
        self.flu_score_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # ---------- auxiliary heads ----------
        # 1) log1p(opportunity)
        self.opp_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        # 2) acc_prior in [0,1]
        self.acc_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        # 3) stability score in [0,1], higher means more stable
        self.sta_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # ---------- fusion gate ----------
        # item context:
        #   mean(a), norm(a), b
        self.fuse_gate = nn.Sequential(
            nn.Linear(flu_latent_dim + 3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        # ---------- initialize ----------
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

    def _constrain_a(self, a: torch.Tensor) -> torch.Tensor:
        if self.a_range is not None:
            return self.a_range * torch.sigmoid(a)
        return F.softplus(a)

    def forward(self,
                user,
                item,
                opportunity,
                acc_prior,
                stability,
                return_aux: bool = False):

        # safety clamp
        user = torch.clamp(user, 0, self.user_num - 1)
        item = torch.clamp(item, 0, self.item_num - 1)

        # ===================== (1) MIRT backbone -> o_logit =====================
        theta = self.theta(user)                    # (B, D)
        a = self._constrain_a(self.a(item))        # (B, D), positive
        b = self.b(item).view(-1)                  # (B,)

        if torch.isnan(theta).any() or torch.isnan(a).any() or torch.isnan(b).any():
            raise ValueError("theta, a, or b contains NaN. Try a smaller a_range or lr.")

        o_logit = irt2pl_logit(theta, a, b)        # (B,)
        o = torch.sigmoid(o_logit)                 # (B,)

        # ===================== (2) proxy fluency features ======================
        opp = self._to_col(opportunity.float())
        acc = self._to_col(acc_prior.float())
        sta = self._to_col(stability.float())

        f_opp = self._safe_log1p(opp)                                  # exposure
        f_acc = torch.clamp(acc, 0.0, 1.0)                             # historical correctness prior
        # assume stability is variance-like in [0, 0.25] or nearby range
        # convert to stability score: higher is better
        f_sta = 1.0 - torch.clamp(4.0 * torch.clamp(sta, min=0.0), 0.0, 1.0)

        beh = torch.cat([f_opp, f_acc, f_sta], dim=1)                  # (B, 3)

        # ===================== (3) fluency latent z + score ====================
        z = self.beh_encoder(beh)                                      # (B, Dz)
        z = z + torch.tanh(self.student_flu_bias(user))                # student baseline
        flu = torch.sigmoid(self.flu_score_head(z)).view(-1)           # (B,)

        # ===================== (4) auxiliary predictions =======================
        opp_tgt = f_opp.view(-1)
        acc_tgt = f_acc.view(-1)
        sta_tgt = f_sta.view(-1)

        opp_hat = self.opp_head(z).view(-1)
        acc_hat = torch.sigmoid(self.acc_head(z)).view(-1)
        sta_hat = torch.sigmoid(self.sta_head(z)).view(-1)

        # ===================== (5) fusion in logit space =======================
        a_mean = a.mean(dim=1, keepdim=True)                           # (B,1)
        a_norm = torch.norm(a, p=2, dim=1, keepdim=True)              # (B,1)
        b_col = b.view(-1, 1)                                         # (B,1)

        gate_in = torch.cat([z, a_mean, a_norm, b_col], dim=1)        # (B, Dz+3)
        gate = torch.sigmoid(self.fuse_gate(gate_in)).view(-1)        # (B,)

        gamma = F.softplus(self.gamma)                                # > 0
        delta = (flu - 0.5) * 4.0                                     # centered to roughly [-2, 2]
        p_logit = o_logit + gamma * gate * delta
        p = torch.sigmoid(p_logit)                                    # (B,)

        if not return_aux:
            return p

        return {
            "p": p,
            "o": o,
            "flu": flu,
            "gate": gate,
            "opp_hat": opp_hat,
            "acc_hat": acc_hat,
            "sta_hat": sta_hat,
            "opp_tgt": opp_tgt,
            "acc_tgt": acc_tgt,
            "sta_tgt": sta_tgt,
        }


class MIRTFluency(CDM):
    def __init__(self,
                 user_num,
                 item_num,
                 latent_dim,
                 a_range=None,
                 lr=0.001,
                 fluency_lambda=0.5,
                 dropout=0.5,
                 # keep old naming style for compatibility
                                  flu_latent_dim=16):
        super(MIRTFluency, self).__init__()
        self.lr = lr

        # compatibility naming
        self.fluency_lambda = float(fluency_lambda)

        self.irt_net = MIRTNetWithFluency(
            user_num=user_num,
            item_num=item_num,
            latent_dim=latent_dim,
            a_range=a_range,
            dropout=dropout,
            flu_latent_dim=flu_latent_dim,
            fluency_lambda=fluency_lambda,
        )

        self.loss_y = nn.BCELoss()
        self.loss_opp = nn.MSELoss()
        self.loss_acc = nn.SmoothL1Loss()
        self.loss_sta = nn.SmoothL1Loss()

    def train(self, train_data, test_data=None, *, epoch: int,
              device="cpu", lr=None, silence=False):
        self.irt_net = self.irt_net.to(device)
        self.irt_net.train()

        optimizer = optim.Adam(self.irt_net.parameters(), lr=(lr or self.lr))

        for e in range(epoch):
            losses = []
            ly_list, lopp_list, lacc_list, lsta_list = [], [], [], []

            for batch_data in tqdm(train_data, f"Epoch {e}", disable=silence):
                # expected batch format:
                # user_id, item_id, response, opp, acc_prior, stability
                user_id, item_id, response, opp, acc_prior, stability = batch_data

                user_id = user_id.to(device)
                item_id = item_id.to(device)
                response = response.to(device).float()

                opp = opp.to(device)
                acc_prior = acc_prior.to(device)
                stability = stability.to(device)

                out = self.irt_net(
                    user_id, item_id,
                    opp, acc_prior, stability,
                    return_aux=True
                )

                # main loss
                ly = self.loss_y(out["p"], response)

                # auxiliary losses
                lopp = self.loss_opp(out["opp_hat"], out["opp_tgt"])
                lacc = self.loss_acc(out["acc_hat"], out["acc_tgt"])
                lsta = self.loss_sta(out["sta_hat"], out["sta_tgt"])

                loss = (1 - self.fluency_lambda) * ly + self.fluency_lambda * (lopp + lacc + lsta)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                ly_list.append(ly.item())
                lopp_list.append(lopp.item())
                lacc_list.append(lacc.item())
                lsta_list.append(lsta.item())

            print(f"[Epoch {e}] loss={float(np.mean(losses)):.6f} "
                  f"(y={float(np.mean(ly_list)):.6f}, opp={float(np.mean(lopp_list)):.6f}, "
                  f"acc={float(np.mean(lacc_list)):.6f}, sta={float(np.mean(lsta_list)):.6f})")

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device, silence=silence)
                print(f"[Epoch {e}] auc: {auc:.6f}, accuracy: {accuracy:.6f}")

    @torch.no_grad()
    def eval(self, test_data, device="cpu", silence=False):
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()

        y_true, y_pred = [], []

        for batch_data in tqdm(test_data, "evaluating", disable=silence):
            user_id, item_id, response, opp, acc_prior, stability = batch_data

            user_id = user_id.to(device)
            item_id = item_id.to(device)
            opp = opp.to(device)
            acc_prior = acc_prior.to(device)
            stability = stability.to(device)

            pred = self.irt_net(
                user_id, item_id,
                opp, acc_prior, stability,
                return_aux=False
            )

            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(response.detach().cpu().tolist())

        self.irt_net.train()
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.irt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath, map_location=None):
        self.irt_net.load_state_dict(torch.load(filepath, map_location=map_location))
        logging.info("load parameters from %s" % filepath)