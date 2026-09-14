import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import sys


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class ACDMFluencyNet(nn.Module):
    def __init__(self, stu_n, exer_n, k_n, emb_dim,
                 flu_latent_dim=16, dropout=0.5, gamma_init=1.0):
        super(ACDMFluencyNet, self).__init__()
        self.stu_n, self.exer_n, self.k_n, self.emb_dim = stu_n, exer_n, k_n, emb_dim
        # ACDM QKV
        self.student_q = nn.Embedding(self.stu_n, self.emb_dim)
        self.exercise_k = nn.Embedding(self.exer_n, self.emb_dim)
        self.student_v = nn.Embedding(self.stu_n, self.emb_dim)
        self.exercise_v = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_v = nn.Embedding(self.k_n, self.emb_dim)
        self.register_buffer('k_index', torch.LongTensor(list(range(self.k_n))))
        self.prednet_full1 = PosLinear(2 * self.emb_dim, self.emb_dim)
        self.prednet_full2 = PosLinear(2 * self.emb_dim, self.emb_dim)
        self.prednet_full3 = PosLinear(self.emb_dim, 1)
        # Fluency branch
        self.beh_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(64, flu_latent_dim), nn.ReLU(),
        )
        self.student_flu_bias = nn.Embedding(self.stu_n, flu_latent_dim)
        self.flu_score_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16), nn.ReLU(), nn.Linear(16, 1),
        )
        # Multi-task auxiliary heads
        self.opp_head = nn.Sequential(nn.Linear(flu_latent_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        self.acc_head = nn.Sequential(nn.Linear(flu_latent_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        self.sta_head = nn.Sequential(nn.Linear(flu_latent_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        # Fusion gate
        self.fuse_gate = nn.Sequential(
            nn.Linear(flu_latent_dim + self.emb_dim, 32), nn.ReLU(), nn.Linear(32, 1),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    @staticmethod
    def _safe_log1p(x):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.log1p(torch.clamp(x, min=0.0))

    @staticmethod
    def _to_col(t):
        return t.view(-1, 1) if t.dim() == 1 else t

    def forward(self, stu_id, exer_id, kq, opportunity, acc_prior, stability, return_aux=False):
        device = stu_id.device
        self.k_index = self.k_index.to(device, non_blocking=True)
        # ACDM knowledge mastery
        stu_q = self.student_q(stu_id)
        exer_k = self.exercise_k(exer_id)
        stu_v = self.student_v(stu_id)
        exer_v = self.exercise_v(exer_id)
        k_v = self.knowledge_v(self.k_index)
        stu_v_exp = stu_v.unsqueeze(1).expand(-1, self.k_n, -1)
        exer_v_exp = exer_v.unsqueeze(1).expand(-1, self.k_n, -1)
        k_v_exp = k_v.unsqueeze(0).expand(stu_id.shape[0], -1, -1)
        preference = torch.sigmoid(self.prednet_full1(torch.cat((stu_v_exp, k_v_exp), dim=2)))
        diff = torch.sigmoid(self.prednet_full2(torch.cat((exer_v_exp, k_v_exp), dim=2)))
        advantage = preference - diff
        disc_1 = torch.sigmoid(stu_q * exer_k)
        input_x = advantage * disc_1.unsqueeze(1).expand(-1, self.k_n, -1)
        o_logits_concept = self.prednet_full3(input_x).squeeze(-1)
        sum_out = torch.sum(o_logits_concept * kq, dim=1)
        o_logit_mastery = sum_out / torch.sum(kq, dim=1).clamp(min=1.0)
        # Fluency
        f_opp = self._safe_log1p(self._to_col(opportunity.float()))
        f_acc = torch.clamp(self._to_col(acc_prior.float()), 0.0, 1.0)
        f_sta = 1.0 - torch.clamp(4.0 * torch.clamp(self._to_col(stability.float()), min=0.0), 0.0, 1.0)
        beh = torch.cat([f_opp, f_acc, f_sta], dim=1)
        z = self.beh_encoder(beh) + torch.tanh(self.student_flu_bias(stu_id))
        flu = torch.sigmoid(self.flu_score_head(z))
        # Fusion gate
        gate = torch.sigmoid(self.fuse_gate(torch.cat([z, exer_k], dim=1)))
        gamma = F.softplus(self.gamma)
        p_logit = o_logit_mastery + gamma * gate.squeeze(-1) * (flu.squeeze(-1) - 0.5) * 4.0
        p = torch.sigmoid(p_logit)
        if not return_aux:
            return p
        return {
            'p': p,
            'opp_hat': self.opp_head(z).squeeze(-1),
            'acc_hat': torch.sigmoid(self.acc_head(z)).squeeze(-1),
            'sta_hat': torch.sigmoid(self.sta_head(z)).squeeze(-1),
            'opp_tgt': f_opp.squeeze(-1),
            'acc_tgt': f_acc.squeeze(-1),
            'sta_tgt': f_sta.squeeze(-1),
        }


class ACDWithFluency:
    def __init__(self, student_n, exer_n, k_n, emb_dim,
                 flu_latent_dim=16, dropout=0.5, gamma_init=1.0,
                 aux_loss_weight=0.1):
        self.acd_net = ACDMFluencyNet(
            student_n, exer_n, k_n, emb_dim,
            flu_latent_dim=flu_latent_dim, dropout=dropout, gamma_init=gamma_init,
        )
        self.aux_loss_weight = aux_loss_weight
        self.reset_stats()

    def reset_stats(self):
        self.stu_opp = {}
        self.stu_acc_sum = {}
        self.stu_acc_var = {}
        self._in_eval = False

    def _compute_fluency_features(self, stu_ids, y_true=None):
        device = stu_ids.device
        batch_size = stu_ids.shape[0]
        opp = torch.zeros(batch_size, device=device)
        acc_prior_vals = torch.full((batch_size,), 0.5, device=device)
        sta = torch.zeros(batch_size, device=device)
        for i, sid in enumerate(stu_ids.cpu().tolist()):
            sid = int(sid)
            opp[i] = self.stu_opp.get(sid, 0)
            if sid in self.stu_acc_sum and self.stu_opp[sid] > 0:
                acc_prior_vals[i] = self.stu_acc_sum[sid] / self.stu_opp[sid]
            if sid in self.stu_acc_var:
                sta[i] = self.stu_acc_var[sid]
        return opp, acc_prior_vals, sta

    def _update_stats(self, stu_ids, y_true):
        for sid, yi in zip(stu_ids.cpu().tolist(), y_true.cpu().tolist()):
            sid = int(sid)
            prev_opp = self.stu_opp.get(sid, 0)
            prev_acc = self.stu_acc_sum.get(sid, 0.0)
            self.stu_opp[sid] = prev_opp + 1
            self.stu_acc_sum[sid] = prev_acc + float(yi)
            if prev_opp > 0:
                prev_avg = prev_acc / prev_opp
                diff = abs(float(yi) - prev_avg)
                old_var = self.stu_acc_var.get(sid, 0.0)
                self.stu_acc_var[sid] = 0.9 * old_var + 0.1 * diff

    def train(self, train_data, test_data=None, epoch=10, device='cpu', lr=0.002):
        self.acd_net = self.acd_net.to(device)
        self.acd_net.train()
        loss_fn = nn.BCELoss()
        aux_loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.acd_net.parameters(), lr=lr)
        best_epoch, best_f1, best_auc, best_acc, best_rmse = 0, 0.0, 0.0, 0.0, 1.0
        for epoch_i in range(epoch):
            self.reset_stats()
            epoch_losses = []
            for batch_data in tqdm(train_data, 'Epoch %s' % epoch_i, file=sys.stdout):
                batch = [x.to(device) for x in batch_data]
                user_id, item_id, kq, y = batch[0], batch[1], batch[2], batch[3]
                # Use pre-computed fluency features if available (7-tuple), else compute on-the-fly
                if len(batch) >= 7:
                    opp, acc_p, sta = batch[4], batch[5], batch[6]
                else:
                    opp, acc_p, sta = self._compute_fluency_features(user_id, y)
                out = self.acd_net(user_id, item_id, kq, opp, acc_p, sta, return_aux=True)
                loss = loss_fn(out['p'], y)
                aux_loss = (aux_loss_fn(out['opp_hat'], out['opp_tgt']) +
                            aux_loss_fn(out['acc_hat'], out['acc_tgt']) +
                            aux_loss_fn(out['sta_hat'], out['sta_tgt']))
                total = loss + self.aux_loss_weight * aux_loss
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
                epoch_losses.append(loss.mean().item())
                self._update_stats(user_id, y)
            print('[Epoch %d] average loss: %.6f' % (epoch_i, float(np.mean(epoch_losses))))
            if test_data is not None:
                auc, accuracy, rmse_val, f1 = self.eval(test_data, device=device)
                print('[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, f1: %.6f' % (epoch_i, auc, accuracy, rmse_val, f1))
                if best_f1 < f1:
                    best_epoch, best_f1, best_auc, best_acc, best_rmse = epoch_i, f1, auc, accuracy, rmse_val
                    self.save('params/arcd_fluency.params')
            print('BEST epoch<%d>, f1: %.6f, auc: %.6f, acc: %.6f, rmse: %.6f' % (best_epoch, best_f1, best_auc, best_acc, best_rmse))
        return best_epoch, best_f1, best_auc, best_acc, best_rmse

    def eval(self, test_data, device='cpu'):
        self.acd_net = self.acd_net.to(device)
        self.acd_net.eval()
        self.reset_stats()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_data in tqdm(test_data, 'Evaluating', file=sys.stdout):
                batch = [x.to(device) for x in batch_data]
                user_id, item_id, kq, y = batch[0], batch[1], batch[2], batch[3]
                if len(batch) >= 7:
                    opp, acc_p, sta = batch[4], batch[5], batch[6]
                else:
                    opp, acc_p, sta = self._compute_fluency_features(user_id, y)
                pred = self.acd_net(user_id, item_id, kq, opp, acc_p, sta)
                y_pred.extend(pred.detach().cpu().tolist())
                y_true.extend(y.tolist())
                self._update_stats(user_id, y)
        rmse_val = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
        y_pred_bin = np.array(y_pred) >= 0.5
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, y_pred_bin), rmse_val, f1_score(y_true, y_pred_bin)

    def save(self, filepath):
        torch.save(self.acd_net.state_dict(), filepath)
        logging.info('save parameters to %s' % filepath)

    def load(self, filepath):
        self.acd_net.load_state_dict(torch.load(filepath))
        logging.info('load parameters from %s' % filepath)