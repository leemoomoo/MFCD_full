# coding: utf-8
"""
FairWISA-fluency: Fair Cognitive Diagnosis with Fluency Features
=================================================================
Combines FairWISA fairness framework (LearnableMatrix + Invariant Penalty)
with IRT backbone + fluency encoder.

Architecture:
  IRT backbone      --?latent traits + item parameters (pre-sigmoid logit)
  Fluency encoder   --?behavioral features --?fluency correction
  LearnableMatrix   --?automatic student grouping
  FairLoss          --?Invariant Penalty + BCE across groups

Two-phase training:
  Phase 1 (discover): freeze backbone, train LearnableMatrix
  Phase 2 (fairness): freeze groups, train model with fairness loss

Fluency inputs:
  - opportunity (exposure count)
  - acc_prior (historical accuracy)
  - stability (response consistency)
"""

import logging, math
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from EduCDM import CDM


# =====================================================================
# Invariant Penalty Loss (FairWISA Eq.5-6)
# =====================================================================
def _inv_penalty_loss(predict, labels):
    """Compute per-group inv(erse) penalty for fairness.
    loss0 = average predicted prob for negative class
    loss1 = average (1 - predicted prob) for positive class
    """
    ids_0 = labels == 0
    ids_1 = labels == 1
    if ids_0.sum() == 0 or ids_1.sum() == 0:
        return torch.tensor(0.0, device=predict.device), torch.tensor(0.0, device=predict.device)
    p0, p1 = predict[ids_0], predict[ids_1]
    return (p0 - 0.0).mean(), (1.0 - p1).mean()


# =====================================================================
# Learnable Grouping Matrix (FairWISA)
# =====================================================================
class LearnableMatrix(nn.Module):
    """Learnable student-to-group assignment matrix with Gumbel-Softmax."""

    def __init__(self, num_students: int, num_groups: int):
        super().__init__()
        self.num_students = num_students
        self.num_groups = num_groups
        self.matrix = nn.Parameter(torch.randn(num_students, num_groups))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, uid, hard: bool = False):
        probs = self.softmax(self.matrix[uid])
        if hard:
            return F.gumbel_softmax(probs, tau=0.1, hard=True)
        return probs

    def get_groups(self, uid):
        """Return hard group assignments: (B,) with values in {0, ..., k-1}."""
        probs = self.softmax(self.matrix)
        return torch.argmax(probs[uid], dim=1)


# =====================================================================
# IRT backbone with Fluency encoder (pre-sigmoid logit)
# =====================================================================
class IRTWithFluency(nn.Module):
    """
    IRT backbone (3PL: theta, a, b, c) + Fluency correction
    p = sigmoid( logit_irt + fluency_lambda * fluency_correction )

    Fluency correction:
      beh_encoder([opp, acc, sta]) --?z --?flu_score --?delta
      gamma * gate * delta  (with student bias + dynamic gate)
    """

    def __init__(self, user_num, item_num,
                 value_range=None, a_range=None,
                 # fluency params
                 dropout=0.5, flu_latent_dim=16, gamma_init=1.0,
                 fluency_lambda=0.5, rt_cap_ms=120000.0):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.fluency_lambda = float(fluency_lambda)
        self.rt_cap_ms = float(rt_cap_ms)
        self.value_range = value_range
        self.a_range = a_range

        # ---- IRT parameters ----
        self.theta = nn.Embedding(user_num, 1)
        self.a     = nn.Embedding(item_num, 1)
        self.b     = nn.Embedding(item_num, 1)
        self.c     = nn.Embedding(item_num, 1)

        # ---- Fluency encoder ----
        self.beh_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(64, flu_latent_dim), nn.ReLU(),
        )
        self.student_flu_bias = nn.Embedding(user_num, flu_latent_dim)

        self.flu_score_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

        # Auxiliary heads (for multi-task learning)
        self.opp_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16), nn.ReLU(), nn.Linear(16, 1),
        )
        self.acc_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16), nn.ReLU(), nn.Linear(16, 1),
        )
        self.sta_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16), nn.ReLU(), nn.Linear(16, 1),
        )

        # Dynamic gate (item-aware fluency weighting)
        self.fuse_gate = nn.Sequential(
            nn.Linear(flu_latent_dim + 1, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        # Init weights
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    @staticmethod
    def _safe_log1p(x):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=0.0)
        return torch.log1p(x)

    @classmethod
    def irf(cls, theta, a, b, c, D=1.702):
        """3PL IRF: p = c + (1-c) / (1 + exp(-D*a*(theta-b)))"""
        return c + (1.0 - c) * torch.sigmoid(D * a * (theta - b))

    def forward(self, user_id, item_id, opportunity=None, acc_prior=None, stability=None,
                return_aux=False):
        """
        Args:
            user_id: (B,) student indices
            item_id: (B,) exercise indices
            opportunity, acc_prior, stability: (B,) fluency features
            return_aux: if True, return dict with aux outputs
        Returns:
            p: (B,) predicted probability
            or dict with p, o, flu, gate, opp_hat, acc_hat, sta_hat, ...
        """
        # ---- IRT logit ----
        theta = torch.squeeze(self.theta(user_id), dim=-1)
        a_vec = torch.squeeze(self.a(item_id), dim=-1)
        b_vec = torch.squeeze(self.b(item_id), dim=-1)
        c_vec = torch.squeeze(self.c(item_id), dim=-1)
        c_vec = torch.sigmoid(c_vec)

        if self.value_range is not None:
            theta = self.value_range * (torch.sigmoid(theta) - 0.5)
            b_vec = self.value_range * (torch.sigmoid(b_vec) - 0.5)
        if self.a_range is not None:
            a_vec = self.a_range * torch.sigmoid(a_vec)
        else:
            a_vec = F.softplus(a_vec)

        # Pre-sigmoid logit from IRT
        logit_irt = self.irf(theta, a_vec, b_vec, c_vec)  # This IS the probability already

        # ---- Fluency features ----
        if opportunity is not None and acc_prior is not None and stability is not None:
            opp = opportunity.float().view(-1, 1)
            acc = acc_prior.float().view(-1, 1)
            sta = stability.float().view(-1, 1)

            f_opp = self._safe_log1p(opp)
            f_acc = torch.clamp(acc, 0.0, 1.0)
            f_sta = 1.0 - torch.clamp(4.0 * torch.clamp(sta, min=0.0), 0.0, 1.0)
            beh = torch.cat([f_opp, f_acc, f_sta], dim=1)

            # Fluency latent
            z = self.beh_encoder(beh)
            z = z + torch.tanh(self.student_flu_bias(user_id))
            flu = torch.sigmoid(self.flu_score_head(z))

            # Auxiliary predictions
            opp_hat = torch.sigmoid(self.opp_head(z))
            acc_hat = torch.sigmoid(self.acc_head(z))
            sta_hat = torch.sigmoid(self.sta_head(z))

            # Fusion gate
            gate_in = torch.cat([z, a_vec.unsqueeze(-1)], dim=1)
            gate = torch.sigmoid(self.fuse_gate(gate_in))

            gamma = F.softplus(self.gamma)
            delta = (flu - 0.5) * 4.0

            # Combine IRT prediction with fluency correction (logit space)
            # Convert IRT prob to logit, add fluency, convert back
            eps = 1e-7
            irt_logit = torch.log(logit_irt / (1 - logit_irt + eps) + eps)
            p_logit = irt_logit.view(-1) + self.fluency_lambda * gamma * gate.view(-1) * delta.view(-1)
            p = torch.sigmoid(p_logit)
        else:
            p = logit_irt
            opp_hat = acc_hat = sta_hat = None
            gate = None
            opp_tgt = acc_tgt = sta_tgt = None

        if not return_aux:
            return p.view(-1)

        return {
            "p": p.view(-1),
            "o": logit_irt.view(-1),
            "flu": flu.view(-1) if opportunity is not None else None,
            "gate": gate.view(-1) if gate is not None else None,
            "opp_hat": opp_hat.view(-1) if opp_hat is not None else None,
            "acc_hat": acc_hat.view(-1) if acc_hat is not None else None,
            "sta_hat": sta_hat.view(-1) if sta_hat is not None else None,
            "opp_tgt": f_opp.view(-1) if opportunity is not None else None,
            "acc_tgt": f_acc.view(-1) if acc_prior is not None else None,
            "sta_tgt": f_sta.view(-1) if stability is not None else None,
        }


# =====================================================================
# FairWISA-Fluency Model (CDM wrapper)
# =====================================================================
class FairWISAFluency(CDM):
    """
    Fair Cognitive Diagnosis: IRT + Fluency + Fairness.
    """

    def __init__(self, user_num, item_num,
                 num_groups=2, fairness_weight=10.0,
                 fluency_lambda=0.5, dropout=0.5,
                 flu_latent_dim=16, lr=0.002,
                 value_range=None, a_range=None):
        super().__init__()
        self.lr = lr
        self.num_groups = num_groups
        self.fairness_weight = float(fairness_weight)
        self.fluency_lambda = float(fluency_lambda)

        self.net = IRTWithFluency(
            user_num, item_num,
            value_range=value_range, a_range=a_range,
            dropout=dropout, flu_latent_dim=flu_latent_dim,
            fluency_lambda=fluency_lambda,
        )
        self.group_matrix = LearnableMatrix(user_num, num_groups)

        self.loss_y = nn.BCELoss()
        self.loss_opp = nn.MSELoss()
        self.loss_acc = nn.SmoothL1Loss()
        self.loss_sta = nn.SmoothL1Loss()

    def forward(self, user_id, item_id, opportunity=None, acc_prior=None, stability=None,
                return_aux=False):
        return self.net(user_id, item_id, opportunity, acc_prior, stability,
                        return_aux=return_aux)

    def unify_weights(self):
        self.net.fluency_lambda = self.fluency_lambda
        logging.info("Using fluency_lambda=%.4f (Eq.17 normalized)", self.fluency_lambda)

    def discover_groups(self, train_data, device="cuda", epoch=30, lr=0.1):
        """Phase 1: Train LearnableMatrix using frozen backbone."""
        self.net.to(device).eval()
        self.group_matrix.to(device).train()
        opt = optim.Adam(self.group_matrix.parameters(), lr=lr)

        for e in range(epoch):
            losses = []
            for batch in tqdm(train_data, f"  GE{e}"):
                uid, iid, kq, y, opp, ac, st = [x.to(device) for x in batch]
                p = self.net(uid, iid, opp, ac, st, return_aux=False)

                gp = self.group_matrix(uid, hard=True)
                i0, i1 = [], []
                for g in range(self.num_groups):
                    mask = gp[:, g] > 0.5
                    if mask.sum() < 2: continue
                    gp_, gy_ = p[mask], y[mask]
                    z0, z1 = gy_ == 0, gy_ == 1
                    if z0.sum() == 0 or z1.sum() == 0: continue
                    i0.append((gp_[z0] - 0.0).mean())
                    i1.append((1.0 - gp_[z1]).mean())
                il = torch.stack(i0).var() + torch.stack(i1).var() if len(i0) >= 2 else torch.tensor(0., device=device)
                l = -il; opt.zero_grad(); l.backward(); opt.step(); losses.append(l.item())
            print(f"  [GE{e}] v={np.mean(losses):.6f}")
        print("[Group Discovery] Done.")

    def train(self, train_data, test_data=None, *, epoch=10, device="cuda", lr=None, silence=False):
        """Phase 2: Train model with fairness constraint."""
        self.net.to(device).train()
        self.group_matrix.to(device).eval()
        opt = optim.Adam(self.net.parameters(), lr=(lr or self.lr))

        best_auc = 0.0
        for ep in range(epoch):
            self.net.train()
            le, li, la = [], [], []
            for batch in tqdm(train_data, f"Epoch {ep}", disable=silence):
                uid, iid, kq, y, opp, ac, st = [x.to(device) for x in batch]
                out = self.net(uid, iid, opp, ac, st, return_aux=True)
                p = out["p"]
                g = self.group_matrix.get_groups(uid)

                bl, i0l, i1l = [], [], []
                for gi in range(self.num_groups):
                    mk = g == gi; gp_, gy_ = p[mk], y[mk]
                    if gp_.numel() == 0: continue
                    bl.append(self.loss_y(gp_, gy_))
                    z0, z1 = gy_ == 0, gy_ == 1
                    if z0.sum() > 0 and z1.sum() > 0:
                        i0l.append((gp_[z0] - 0.).mean())
                        i1l.append((1. - gp_[z1]).mean())
                el = torch.stack(bl).sum()
                al = self.fluency_lambda * (self.loss_opp(out["opp_hat"], out["opp_tgt"])
                      + self.loss_acc(out["acc_hat"], out["acc_tgt"])
                      + self.loss_sta(out["sta_hat"], out["sta_tgt"]))
                il = torch.stack(i0l).var() + torch.stack(i1l).var() if len(i0l) >= 2 else torch.tensor(0., device=device)
                ((1 - self.fluency_lambda) * el + al + self.fairness_weight * il).backward(); opt.step()
                le.append(el.item()); li.append(il.item()); la.append(al.item())

            if test_data is not None:
                auc, acc = self.eval(test_data, device=device, silence=silence)
                print(f"  [E{ep}] e={np.mean(le):.4f} i={np.mean(li):.4f} a={np.mean(la):.4f}  auc={auc:.4f} acc={acc:.4f}")
                if auc > best_auc: best_auc = auc
            else:
                print(f"  [E{ep}] e={np.mean(le):.4f} i={np.mean(li):.4f} a={np.mean(la):.4f}")

    @torch.no_grad()
    def eval(self, test_data, device="cpu", silence=False):
        self.net.to(device).eval()
        yt, yp = [], []
        for batch in tqdm(test_data, "Eval", disable=silence):
            uid, iid, kq, y, opp, ac, st = [x.to(device) for x in batch]
            p = self.net(uid, iid, opp, ac, st, return_aux=False)
            yp.extend(p.cpu().tolist()); yt.extend(y.cpu().tolist())
        return roc_auc_score(yt, yp), accuracy_score(yt, np.array(yp) >= 0.5)

    def save(self, path):
        torch.save({"net": self.net.state_dict(), "group_matrix": self.group_matrix.state_dict()}, path)
        logging.info("save to %s", path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.net.load_state_dict(ckpt["net"])
        self.group_matrix.load_state_dict(ckpt["group_matrix"])
        logging.info("load from %s", path)


__all__ = ["FairWISAFluency", "LearnableMatrix"]
