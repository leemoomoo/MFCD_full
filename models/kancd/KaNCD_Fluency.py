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
    """
    KaNCD + Fluency Proxy version

    Main task:
        - predict correctness p(correct)

    Fluency inputs (dataset-agnostic proxies):
        - opportunity: historical practice exposure on related knowledge points
        - acc_prior: historical correctness prior on related knowledge points
        - stability: historical performance variance / instability proxy

    Fusion:
        p_logit = o_logit + gamma * gate * delta(fluency)
    """

    def __init__(self,
                 exer_n,
                 student_n,
                 knowledge_n,
                 mf_type,
                 dim,
                 dropout=0.5,
                 flu_latent_dim=16,
                 gamma_init=1.0,
                 fluency_lambda=0.5,
                 rt_cap_ms=120000.0):
        super(NetWithFluency, self).__init__()

        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128

        self.fluency_lambda = float(fluency_lambda)   # compatibility only
        self.rt_cap_ms = float(rt_cap_ms)             # compatibility only

        # ---------- KaNCD backbone ----------
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        self.e_discrimination = nn.Embedding(self.exer_n, 1)

        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=dropout)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=dropout)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)   # mastery logit

        if mf_type == 'gmf':
            self.k_diff_full = nn.Linear(self.emb_dim, 1)
            self.stat_full = nn.Linear(self.emb_dim, 1)
        elif mf_type == 'ncf1':
            self.k_diff_full = nn.Linear(2 * self.emb_dim, 1)
            self.stat_full = nn.Linear(2 * self.emb_dim, 1)
        elif mf_type == 'ncf2':
            self.k_diff_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.k_diff_full2 = nn.Linear(self.emb_dim, 1)
            self.stat_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.stat_full2 = nn.Linear(self.emb_dim, 1)

        # ---------- Fluency encoder ----------
        self.beh_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, flu_latent_dim),
            nn.ReLU(),
        )

        # student-level fluency baseline
        self.student_flu_bias = nn.Embedding(self.student_n, flu_latent_dim)

        # fluency scalar in (0,1)
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

        # ---------- fusion gate ----------
        # z + item context
        self.fuse_gate = nn.Sequential(
            nn.Linear(flu_latent_dim + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        # ---------- initialize ----------
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

    @staticmethod
    def _safe_log1p(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=0.0)
        return torch.log1p(x)

    @staticmethod
    def _to_col(t: torch.Tensor) -> torch.Tensor:
        return t.view(-1, 1) if t.dim() == 1 else t

    def _compute_stat_emb(self, stu_id):
        """
        Compute student knowledge proficiency: (B, K)
        """
        stu_emb = self.student_emb(stu_id)  # (B, D)
        batch, dim = stu_emb.size()

        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)

        if self.mf_type == 'mf':
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))
        elif self.mf_type == 'gmf':
            stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(
                self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))
            ).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(
                self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1))
            )
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)
        else:
            raise ValueError(f"Unsupported mf_type: {self.mf_type}")

        return stat_emb

    def _compute_k_difficulty(self, input_exercise):
        """
        Compute item knowledge difficulty: (B, K)
        """
        exer_emb = self.exercise_emb(input_exercise)  # (B, D)
        batch, dim = exer_emb.size()

        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)

        if self.mf_type == 'mf':
            k_difficulty = torch.sigmoid((exer_emb * knowledge_emb).sum(dim=-1, keepdim=False))
        elif self.mf_type == 'gmf':
            k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            k_difficulty = torch.sigmoid(
                self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))
            ).view(batch, -1)
        elif self.mf_type == 'ncf2':
            k_difficulty = torch.sigmoid(
                self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1))
            )
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
        else:
            raise ValueError(f"Unsupported mf_type: {self.mf_type}")

        return k_difficulty

    def forward(self,
                stu_id,
                input_exercise,
                input_knowledge_point,
                opportunity,
                acc_prior,
                stability,
                return_aux: bool = False):

        # safety clamp
        stu_id = torch.clamp(stu_id, 0, self.student_n - 1)
        input_exercise = torch.clamp(input_exercise, 0, self.exer_n - 1)

        # ===================== (1) KaNCD backbone -> o_logit =====================
        stat_emb = self._compute_stat_emb(stu_id)                         # (B, K)
        k_difficulty = self._compute_k_difficulty(input_exercise)        # (B, K)
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))  # (B, 1)

        input_x = e_discrimination * (stat_emb - k_difficulty) * input_knowledge_point.float()
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        o_logit = self.prednet_full3(input_x)                            # (B, 1)
        o = torch.sigmoid(o_logit)                                       # (B, 1)

        # ===================== (2) proxy fluency features ======================
        opp = self._to_col(opportunity.float())
        acc = self._to_col(acc_prior.float())
        sta = self._to_col(stability.float())

        f_opp = self._safe_log1p(opp)
        f_acc = torch.clamp(acc, 0.0, 1.0)
        # assume variance-like stability input; convert to higher-is-better score
        f_sta = 1.0 - torch.clamp(4.0 * torch.clamp(sta, min=0.0), 0.0, 1.0)

        beh = torch.cat([f_opp, f_acc, f_sta], dim=1)                    # (B, 3)

        # ===================== (3) fluency latent z + score ====================
        z = self.beh_encoder(beh)                                        # (B, Df)
        z = z + torch.tanh(self.student_flu_bias(stu_id))
        flu = torch.sigmoid(self.flu_score_head(z))                      # (B, 1)

        # ===================== (4) auxiliary predictions =======================
        opp_tgt = f_opp
        acc_tgt = f_acc
        sta_tgt = f_sta

        opp_hat = self.opp_head(z)
        acc_hat = torch.sigmoid(self.acc_head(z))
        sta_hat = torch.sigmoid(self.sta_head(z))

        # ===================== (5) fusion in logit space =======================
        gate_in = torch.cat(
            [
                z,
                e_discrimination,
                k_difficulty.mean(dim=1, keepdim=True),
            ],
            dim=1
        )
        gate = torch.sigmoid(self.fuse_gate(gate_in))                    # (B, 1)

        gamma = F.softplus(self.gamma)                                   # > 0
        delta = (flu - 0.5) * 4.0                                        # centered
        p_logit = o_logit + gamma * gate * delta
        p = torch.sigmoid(p_logit)                                       # (B, 1)

        if not return_aux:
            return p.view(-1)

        return {
            "p": p.view(-1),
            "o": o.view(-1),
            "flu": flu.view(-1),
            "gate": gate.view(-1),
            "opp_hat": opp_hat.view(-1),
            "acc_hat": acc_hat.view(-1),
            "sta_hat": sta_hat.view(-1),
            "opp_tgt": opp_tgt.view(-1),
            "acc_tgt": acc_tgt.view(-1),
            "sta_tgt": sta_tgt.view(-1),
        }


class KaNCDFluency(CDM):
    def __init__(self, **kwargs):
        super(KaNCDFluency, self).__init__()

        mf_type = kwargs['mf_type'] if 'mf_type' in kwargs else 'gmf'
        self.lr = kwargs['lr'] if 'lr' in kwargs else 0.002

        # keep old naming style for compatibility
        self.lambda_rt = float(kwargs['lambda_rt']) if 'lambda_rt' in kwargs else 0.1
        self.lambda_hint = float(kwargs['lambda_hint']) if 'lambda_hint' in kwargs else 0.05
        self.lambda_att = float(kwargs['lambda_att']) if 'lambda_att' in kwargs else 0.05

        self.net = NetWithFluency(
            exer_n=kwargs['exer_n'],
            student_n=kwargs['student_n'],
            knowledge_n=kwargs['knowledge_n'],
            mf_type=mf_type,
            dim=kwargs['dim'],
            dropout=kwargs['dropout'] if 'dropout' in kwargs else 0.5,
            flu_latent_dim=kwargs['flu_latent_dim'] if 'flu_latent_dim' in kwargs else 16,
            gamma_init=kwargs['gamma_init'] if 'gamma_init' in kwargs else 1.0,
            fluency_lambda=kwargs['fluency_lambda'] if 'fluency_lambda' in kwargs else 0.5,
            rt_cap_ms=kwargs['rt_cap_ms'] if 'rt_cap_ms' in kwargs else 120000.0,
        )

        self.loss_y = nn.BCELoss()
        self.loss_opp = nn.MSELoss()
        self.loss_acc = nn.SmoothL1Loss()
        self.loss_sta = nn.SmoothL1Loss()

    def train(self, train_set, valid_set=None, lr=None, device='cpu', epoch_n=15, silence=False):
        logging.info("training KaNCDFluency... (lr={})".format(lr if lr is not None else self.lr))
        self.net = self.net.to(device)
        self.net.train()

        optimizer = optim.Adam(self.net.parameters(), lr=(lr if lr is not None else self.lr))

        for epoch_i in range(epoch_n):
            self.net.train()
            losses = []
            ly_list, lopp_list, lacc_list, lsta_list = [], [], [], []

            for batch_data in tqdm(train_set, f"Epoch {epoch_i}", disable=silence):
                # expected batch format:
                # user_id, item_id, knowledge_emb, y, opp, acc_prior, stability
                user_info, item_info, knowledge_emb, y, opp, acc_prior, stability = batch_data

                user_info = user_info.to(device)
                item_info = item_info.to(device)
                knowledge_emb = knowledge_emb.to(device)
                y = y.to(device).float()

                opp = opp.to(device)
                acc_prior = acc_prior.to(device)
                stability = stability.to(device)

                out = self.net(
                    user_info,
                    item_info,
                    knowledge_emb,
                    opp,
                    acc_prior,
                    stability,
                    return_aux=True,
                )

                ly = self.loss_y(out["p"], y)
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
                f"[Epoch {epoch_i}] loss={float(np.mean(losses)):.6f} "
                f"(y={float(np.mean(ly_list)):.6f}, opp={float(np.mean(lopp_list)):.6f}, "
                f"acc={float(np.mean(lacc_list)):.6f}, sta={float(np.mean(lsta_list)):.6f})"
            )
            logging.info(
                f"[Epoch {epoch_i}] loss={float(np.mean(losses)):.6f} "
                f"(y={float(np.mean(ly_list)):.6f}, opp={float(np.mean(lopp_list)):.6f}, "
                f"acc={float(np.mean(lacc_list)):.6f}, sta={float(np.mean(lsta_list)):.6f})"
            )

            if valid_set is not None:
                auc, acc = self.eval(valid_set, device=device, silence=silence)
                print(f"[Epoch {epoch_i}] auc: {auc:.6f}, acc: {acc:.6f}")
                logging.info(f"[Epoch {epoch_i}] auc: {auc:.6f}, acc: {acc:.6f}")

        if valid_set is not None:
            return auc, acc
        return None, None

    @torch.no_grad()
    def eval(self, test_data, device="cpu", silence=False):
        logging.info('eval ... ')
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
                user_id,
                item_id,
                knowledge_emb,
                opp,
                acc_prior,
                stability,
                return_aux=False,
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