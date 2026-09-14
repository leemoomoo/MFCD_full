# coding: utf-8
# IRT + Fluency

import logging
import numpy as np
import torch
from EduCDM import CDM
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

def irt3pl(theta, a, b, c, F=torch):
    """
    3PL IRT:
    P = c + (1 - c) * sigmoid(a * (theta - b))
    """
    return c + (1 - c) * F.sigmoid(a * (theta - b))

class IRTNetFluency(nn.Module):

    def __init__(self, user_num, item_num, value_range, a_range, flu_dim=3, irf_kwargs=None):
        super(IRTNetFluency, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}

        # ===== IRT parameters =====
        self.theta = nn.Embedding(self.user_num, 1)
        self.a = nn.Embedding(self.item_num, 1)
        self.b = nn.Embedding(self.item_num, 1)
        self.c = nn.Embedding(self.item_num, 1)

        # ===== Fluency projection =====
        self.flu_proj = nn.Linear(flu_dim, 1)
        self.gamma = nn.Parameter(torch.tensor(1.0))

        self.value_range = value_range
        self.a_range = a_range

    def forward(self, user, item, opp, acc_prior, stability):

        theta = torch.squeeze(self.theta(user), dim=-1)

        # ===== fluency proxy =====
        flu = torch.stack([opp, acc_prior, stability], dim=1)
        flu = self.flu_proj(flu).squeeze(-1)

        theta = theta + self.gamma * flu

        a = torch.squeeze(self.a(item), dim=-1)
        b = torch.squeeze(self.b(item), dim=-1)
        c = torch.squeeze(self.c(item), dim=-1)

        c = torch.sigmoid(c)

        if self.value_range is not None:
            theta = self.value_range * (torch.sigmoid(theta) - 0.5)
            b = self.value_range * (torch.sigmoid(b) - 0.5)

        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)

        return self.irf(theta, a, b, c, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, c, **kwargs):
        return irt3pl(theta, a, b, c, F=torch, **kwargs)


class IRTFluency(CDM):

    def __init__(self, user_num, item_num, value_range=None, a_range=None):
        super(IRTFluency, self).__init__()

        self.net = IRTNetFluency(user_num, item_num, value_range, a_range)

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001):

        self.net = self.net.to(device)
        loss_function = nn.BCELoss()
        trainer = torch.optim.Adam(self.net.parameters(), lr)

        for e in range(epoch):

            losses = []

            for batch_data in tqdm(train_data, "Epoch %s" % e):

                user_id, item_id, response, opp, acc_prior, stability = batch_data

                user_id = user_id.to(device)
                item_id = item_id.to(device)
                response = response.to(device)

                opp = opp.to(device)
                acc_prior = acc_prior.to(device)
                stability = stability.to(device)

                pred = self.net(user_id, item_id, opp, acc_prior, stability)

                loss = loss_function(pred, response)

                trainer.zero_grad()
                loss.backward()
                trainer.step()

                losses.append(loss.mean().item())

            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))

            if test_data is not None:
                auc, acc = self.eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (e, auc, acc))

    def eval(self, test_data, device="cpu"):

        self.net = self.net.to(device)
        self.net.eval()

        y_pred = []
        y_true = []

        for batch_data in tqdm(test_data, "evaluating"):

            user_id, item_id, response, opp, acc_prior, stability = batch_data

            user_id = user_id.to(device)
            item_id = item_id.to(device)

            opp = opp.to(device)
            acc_prior = acc_prior.to(device)
            stability = stability.to(device)

            pred = self.net(user_id, item_id, opp, acc_prior, stability)

            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        self.net.train()

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)