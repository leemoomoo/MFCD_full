# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.knowledge_dim = args.n_skill
        self.exer_n = args.n_exercise
        self.emb_num = args.n_student
        self.affect_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256

        self.student_affect = nn.Embedding(self.emb_num, self.affect_dim)
        self.diff_emb = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.skill_emb = nn.Embedding(self.knowledge_dim, self.knowledge_dim)
        self.answer_emb = nn.Embedding(2, self.knowledge_dim)
        self.e_disc = nn.Embedding(self.exer_n, 1)

        self.prednet_affect = nn.Linear(self.knowledge_dim + self.affect_dim, 4)
        self.guess = nn.Linear(4, 1)
        self.slip = nn.Linear(4, 1)

        self.fc = nn.Linear(self.knowledge_dim, self.knowledge_dim)
        self.reLinear = nn.Linear(self.knowledge_dim, self.knowledge_dim)

        self.skill_state = nn.Embedding(self.emb_num, self.knowledge_dim)
        self.study_ability = nn.Embedding(self.emb_num, self.knowledge_dim)

        self.fc_state = nn.Linear(2 * self.knowledge_dim, 1)
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
            elif "norm" in name and "weight" in name:
                nn.init.constant_(param, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, stu_id, exer_id, qe, answers):
        stu_affect = torch.sigmoid(self.student_affect(stu_id))          # [B, K]
        skill_state = torch.sigmoid(self.skill_state(stu_id))            # [B, K]
        stu_ability = torch.sigmoid(self.study_ability(stu_id))          # [B, K]

        # ===== 稳定版 similarity matrix =====
        skill_state1 = skill_state.T                                     # [K, B]
        skill_state1 = F.normalize(skill_state1, p=2, dim=1, eps=1e-8)   # 防止除零
        similarity_martix = torch.mm(skill_state1, skill_state1.t())     # [K, K]

        skill_state2 = torch.matmul(skill_state, similarity_martix)      # [B, K]

        # exercise factor
        e_diff = torch.sigmoid(self.diff_emb(exer_id))                   # [B, K]
        e_disc = torch.sigmoid(self.e_disc(exer_id))                     # [B, 1]

        affect = torch.sigmoid(
            self.prednet_affect(torch.cat((stu_affect, e_diff), dim=-1))
        )                                                                # [B, 4]

        stu_ability_now = self.fc(torch.sigmoid(stu_ability + e_disc * e_diff))
        stu_ability = F.relu(self.dropout(stu_ability_now + self.reLinear(stu_ability)))

        g = torch.sigmoid(self.fc_state(torch.cat([skill_state2, stu_ability], -1)))  # [B, 1]
        state = g * skill_state2 + (1 - g) * stu_ability

        input_x = e_disc * (state - e_diff) * qe
        input_x = torch.sigmoid(self.prednet_full1(input_x))
        input_x = torch.sigmoid(self.prednet_full2(input_x))
        o = torch.sigmoid(self.prednet_full3(input_x))

        g = torch.sigmoid(self.guess(affect))
        s = torch.sigmoid(self.slip(affect))
        output = ((1 - s) * o) + (g * (1 - o))

        # 数值保护
        output = torch.nan_to_num(output, nan=0.5, posinf=1.0, neginf=0.0)
        output = torch.clamp(output, min=1e-6, max=1 - 1e-6)

        cl_loss = self.contrastive_loss(affect, answers, 0.05)
        cl_loss = torch.nan_to_num(cl_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return output, affect, cl_loss

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def contrastive_loss(self, affects, labels, tau=0.05):
        """
        稳定版 supervised contrastive loss
        affects: [B, D]
        labels : [B]
        """
        if affects.size(0) <= 1:
            return torch.tensor(0.0, device=affects.device)

        labels = labels.view(-1)
        features = F.normalize(affects, p=2, dim=1, eps=1e-8)

        sim_matrix = torch.matmul(features, features.T) / tau
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0]  # 数值稳定

        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask = torch.eye(labels.size(0), device=affects.device, dtype=torch.bool)

        pos_mask = labels_equal & (~mask)
        exp_sim = torch.exp(sim_matrix) * (~mask)

        denom = exp_sim.sum(dim=1, keepdim=True) + 1e-8
        log_prob = sim_matrix - torch.log(denom)

        pos_count = pos_mask.sum(dim=1).float()
        valid = pos_count > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=affects.device)

        mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / (pos_count + 1e-8)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


class SelfAttention(nn.Module):
    def __init__(self, dim, n_skill):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.fc = nn.Linear(dim, n_skill)

    def forward(self, q, v, k, device):
        query = self.query(q)
        key = self.key(v)
        value = self.value(v)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.dim, dtype=torch.float32, device=query.device)
        )
        attention_scores = torch.softmax(attention_scores, dim=-1)

        topk_scores, indices = torch.topk(attention_scores, k, dim=1)
        min_value = torch.full_like(attention_scores, fill_value=-1e10).to(device)
        min_value.scatter_(1, indices, topk_scores)
        attention_weights = torch.softmax(min_value, dim=-1)

        attended_values = torch.matmul(attention_weights, value)
        output = self.fc(attended_values)

        mask = torch.zeros_like(attention_scores, dtype=torch.float32)
        mask.scatter_(1, indices, 1)

        return output, mask, attention_weights


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, "weight"):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)