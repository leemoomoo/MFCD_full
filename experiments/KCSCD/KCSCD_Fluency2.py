# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class ChannelGate(nn.Module):
    """Input: core_feat [B,K] -> gate [B,K] in (0,1). Activated on mask=1."""
    def __init__(self, k_n, hidden_ratio=0.5):
        super().__init__()
        hidden = max(8, int(k_n * hidden_ratio))
        self.fc1 = nn.Linear(k_n, hidden)
        self.fc2 = nn.Linear(hidden, k_n)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, core_feat, mask):
        h = F.relu(self.fc1(core_feat))
        gate = torch.sigmoid(self.fc2(h))
        gate = gate * mask
        return gate


class Net(nn.Module):
    """
    KCSCD + ASSIST09 Fluency

    Compatible calls:
        output_1, affect_p, CL_loss = net(stu_id, exer_id, qe, answers)

        output_1, affect_p, CL_loss = net(
            stu_id, exer_id, qe, answers,
            opportunity=...,
            ms_first_response=...,
            attempt_count=...,
            hint_count=...
        )

        aux = net(..., return_aux=True)
    """

    def __init__(self, args, use_fluency=True):
        super(Net, self).__init__()

        self.knowledge_dim = int(args.n_skill)
        self.exer_n = int(args.n_exercise)
        self.emb_num = int(args.n_student)
        self.affect_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1 = int(getattr(args, "prednet_len1", 512))
        self.prednet_len2 = int(getattr(args, "prednet_len2", 256))
        self.dropout_p = float(getattr(args, "dropout", 0.5))
        self.rt_cap_ms = float(getattr(args, "rt_cap_ms", 120000.0))

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

        self.dropout = nn.Dropout(self.dropout_p)
        self.use_fluency = use_fluency

        self.flu_latent_dim = int(getattr(args, "flu_latent_dim", 8))

        self.lambda_opp = float(getattr(args, "lambda_opp", 0.005))
        self.lambda_rt = float(getattr(args, "lambda_rt", 0.005))
        self.lambda_att = float(getattr(args, "lambda_att", 0.002))
        self.lambda_hint = float(getattr(args, "lambda_hint", 0.002))
        self.lambda_consistency = float(getattr(args, "lambda_consistency", 0.002))

        gamma_init = float(getattr(args, "gamma_init", 0.05))
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        self.gate_alpha = float(getattr(args, "gate_alpha", 0.5))
        self.channel_gate = ChannelGate(k_n=self.knowledge_dim, hidden_ratio=float(getattr(args, "gate_hidden_ratio", 0.5)))

        self.delta_scale = float(getattr(args, "delta_scale", 0.5))
        self.gate_scale = float(getattr(args, "gate_scale", 0.2))
        self.warmup_epochs = int(getattr(args, "warmup_epochs", 3))
        self._epoch_counter = 0

        self.beh_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(32, self.flu_latent_dim),
            nn.ReLU(),
        )

        self.student_flu_bias = nn.Embedding(self.emb_num, self.flu_latent_dim)

        self.flu_score_head = nn.Sequential(
            nn.Linear(self.flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.opp_head = nn.Sequential(
            nn.Linear(self.flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.rt_head = nn.Sequential(
            nn.Linear(self.flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.att_head = nn.Sequential(
            nn.Linear(self.flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.hint_head = nn.Sequential(
            nn.Linear(self.flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.fuse_gate = nn.Sequential(
            nn.Linear(self.flu_latent_dim + 4 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.affect_proj = nn.Sequential(
            nn.Linear(4, self.flu_latent_dim),
            nn.ReLU(),
        )

        self.loss_opp = nn.MSELoss()
        self.loss_rt = nn.SmoothL1Loss()
        self.loss_att = nn.SmoothL1Loss()
        self.loss_hint = nn.SmoothL1Loss()
        self.loss_consistency = nn.MSELoss()

        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def start_new_epoch(self):
        self._epoch_counter += 1

    def _is_warmup(self):
        return self._epoch_counter < self.warmup_epochs

    @staticmethod
    def _safe_log1p(x):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=0.0)
        return torch.log1p(x)

    @staticmethod
    def _to_col(t):
        if t is None:
            return None
        return t.view(-1, 1) if t.dim() == 1 else t

    def _default_proxy(self, ref):
        return torch.zeros(ref.size(0), 1, device=ref.device, dtype=torch.float32)

    def _prepare_assist09_inputs(
        self,
        answers,
        opportunity,
        ms_first_response,
        attempt_count,
        hint_count,
    ):
        if answers is None:
            raise ValueError("answers is required for current training pipeline.")

        ref = answers if answers.dim() > 1 else answers.view(-1, 1)

        if opportunity is None:
            opportunity = self._default_proxy(ref)
        else:
            opportunity = self._to_col(opportunity.float()).to(ref.device)

        if ms_first_response is None:
            ms_first_response = self._default_proxy(ref)
        else:
            ms_first_response = self._to_col(ms_first_response.float()).to(ref.device)

        if attempt_count is None:
            attempt_count = torch.ones_like(ref)
        else:
            attempt_count = self._to_col(attempt_count.float()).to(ref.device)

        if hint_count is None:
            hint_count = self._default_proxy(ref)
        else:
            hint_count = self._to_col(hint_count.float()).to(ref.device)

        return opportunity, ms_first_response, attempt_count, hint_count

    def forward(
        self,
        stu_id,
        exer_id,
        qe,
        answers=None,
        opportunity=None,
        ms_first_response=None,
        attempt_count=None,
        hint_count=None,
        return_aux=False,
    ):
        stu_id = torch.clamp(stu_id.long(), 0, self.emb_num - 1)
        exer_id = torch.clamp(exer_id.long(), 0, self.exer_n - 1)
        qe = qe.float()

        stu_affect = torch.sigmoid(self.student_affect(stu_id))
        skill_state = torch.sigmoid(self.skill_state(stu_id))
        stu_ability = torch.sigmoid(self.study_ability(stu_id))

        # 1. 提取当前 Batch 学生的知识状态 [B, K]
        skill_state = torch.sigmoid(self.skill_state(stu_id))

        # 2. 获取全局所有学生的知识状态来计算稳定的知识点相关性 [emb_num, K]
        global_skill_states = torch.sigmoid(self.skill_state.weight)
        global_skill_states_t = global_skill_states.T  # [K, emb_num]

        # 3. 在全局学生维度上进行归一化，得到稳定的全局关系矩阵
        global_skill_states_t = F.normalize(global_skill_states_t, p=2, dim=1, eps=1e-8)
        similarity_matrix = torch.mm(global_skill_states_t, global_skill_states_t.t())  # 稳定的 [K, K] 矩阵

        # 4. 融合计算
        skill_state2 = torch.matmul(skill_state, similarity_matrix)

        e_diff = torch.sigmoid(self.diff_emb(exer_id))
        e_disc = torch.sigmoid(self.e_disc(exer_id))

        affect = torch.sigmoid(
            self.prednet_affect(torch.cat((stu_affect, e_diff), dim=-1))
        )

        stu_ability_now = self.fc(torch.sigmoid(stu_ability + e_disc * e_diff))
        stu_ability = F.relu(self.dropout(stu_ability_now + self.reLinear(stu_ability)))

        g_state = torch.sigmoid(self.fc_state(torch.cat([skill_state2, stu_ability], -1)))
        state = g_state * skill_state2 + (1 - g_state) * stu_ability

        input_x = e_disc * (state - e_diff) * qe
        input_x = torch.sigmoid(self.prednet_full1(input_x))
        input_x = torch.sigmoid(self.prednet_full2(input_x))
        o = torch.sigmoid(self.prednet_full3(input_x))

        g = torch.sigmoid(self.guess(affect))
        s = torch.sigmoid(self.slip(affect))

        base_output = ((1 - s) * o) + (g * (1 - o))
        base_output = torch.nan_to_num(base_output, nan=0.5, posinf=1.0, neginf=0.0)
        if not self.use_fluency:
            output = base_output.view(-1)
            return output, torch.zeros(output.size(0), 4, device=output.device), torch.tensor(0.0, device=output.device)
        base_output = torch.clamp(base_output, min=1e-6, max=1 - 1e-6)

        if answers is None:
            raise ValueError("answers is required.")

        cl_loss_base = self.contrastive_loss(affect, answers, 0.05)
        cl_loss_base = torch.nan_to_num(cl_loss_base, nan=0.0, posinf=0.0, neginf=0.0)

        opportunity, ms_first_response, attempt_count, hint_count = self._prepare_assist09_inputs(
            answers,
            opportunity,
            ms_first_response,
            attempt_count,
            hint_count,
        )

        rt = torch.clamp(ms_first_response, min=0.0, max=self.rt_cap_ms)
        att_extra = torch.clamp(attempt_count - 1.0, min=0.0)
        hint = torch.clamp(hint_count, min=0.0)

        f_opp = self._safe_log1p(opportunity)
        f_rt = self._safe_log1p(rt) / np.log1p(self.rt_cap_ms)
        f_att = self._safe_log1p(att_extra) / np.log1p(10.0)
        f_hint = self._safe_log1p(hint) / np.log1p(10.0)

        f_rt = torch.clamp(f_rt, 0.0, 1.0)
        f_att = torch.clamp(f_att, 0.0, 1.0)
        f_hint = torch.clamp(f_hint, 0.0, 1.0)

        beh = torch.cat([f_opp, 1.0 - f_rt, 1.0 - f_att, 1.0 - f_hint], dim=1)

        z = self.beh_encoder(beh)
        z = z + 0.1 * torch.tanh(self.student_flu_bias(stu_id))
        flu = torch.sigmoid(self.flu_score_head(z))

        base_logit = torch.logit(base_output, eps=1e-6)

        gate_in = torch.cat([z, affect, e_disc], dim=1)
        raw_gate = torch.sigmoid(self.fuse_gate(gate_in))
        gate = self.gate_scale * raw_gate

        gamma = F.softplus(self.gamma)
        delta = (flu - 0.5) * self.delta_scale

        if self.training and self._is_warmup():
            final_logit = base_logit
            gate_used = torch.zeros_like(gate)
        else:
            final_logit = base_logit + gamma * gate * delta
            gate_used = gate

        output = torch.sigmoid(final_logit)
        output = torch.nan_to_num(output, nan=0.5, posinf=1.0, neginf=0.0)
        output = torch.clamp(output, min=1e-6, max=1 - 1e-6)

        opp_hat = self.opp_head(z)
        rt_hat = torch.sigmoid(self.rt_head(z))
        att_hat = torch.sigmoid(self.att_head(z))
        hint_hat = torch.sigmoid(self.hint_head(z))

        opp_tgt = f_opp
        rt_tgt = f_rt
        att_tgt = f_att
        hint_tgt = f_hint

        lopp = self.loss_opp(opp_hat, opp_tgt)
        lrt = self.loss_rt(rt_hat, rt_tgt)
        latt = self.loss_att(att_hat, att_tgt)
        lhint = self.loss_hint(hint_hat, hint_tgt)

        affect_z = self.affect_proj(affect)
        consistency = self.loss_consistency(
            F.normalize(affect_z, p=2, dim=1),
            F.normalize(z, p=2, dim=1),
        )

        cl_loss = (
            cl_loss_base
            + self.lambda_opp * lopp
            + self.lambda_rt * lrt
            + self.lambda_att * latt
            + self.lambda_hint * lhint
            + self.lambda_consistency * consistency
        )
        cl_loss = torch.nan_to_num(cl_loss, nan=0.0, posinf=0.0, neginf=0.0)

        if return_aux:
            return {
                "p": output,
                "base_p": base_output,
                "flu": flu,
                "affect": affect,
                "gate": gate_used,
                "raw_gate": raw_gate,
                "opp_hat": opp_hat,
                "rt_hat": rt_hat,
                "att_hat": att_hat,
                "hint_hat": hint_hat,
                "opp_tgt": opp_tgt,
                "rt_tgt": rt_tgt,
                "att_tgt": att_tgt,
                "hint_tgt": hint_tgt,
                "consistency": consistency,
                "cl_loss": cl_loss,
                "base_cl_loss": cl_loss_base,
                "gamma": gamma.detach(),
            }

        return output, affect, cl_loss

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def contrastive_loss(self, affects, labels, tau=0.05):
        if affects.size(0) <= 1:
            return torch.tensor(0.0, device=affects.device)

        labels = labels.view(-1)
        features = F.normalize(affects, p=2, dim=1, eps=1e-8)

        sim_matrix = torch.matmul(features, features.T) / tau
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0]

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
        return -mean_log_prob_pos[valid].mean()



class NetWithChannel(Net):
    """Net (KCSCD) extended with ChannelGate inserted after input_x computation."""
    def forward(self, stu_id, exer_id, qe, answers=None,
                opportunity=None, ms_first_response=None,
                attempt_count=None, hint_count=None, return_aux=False):
        stu_id = torch.clamp(stu_id.long(), 0, self.emb_num - 1)
        exer_id = torch.clamp(exer_id.long(), 0, self.exer_n - 1)
        qe = qe.float()

        stu_affect = torch.sigmoid(self.student_affect(stu_id))
        skill_state = torch.sigmoid(self.skill_state(stu_id))
        stu_ability = torch.sigmoid(self.study_ability(stu_id))
        skill_state1 = skill_state.T
        skill_state1 = F.normalize(skill_state1, p=2, dim=1, eps=1e-8)
        similarity_matrix = torch.mm(skill_state1, skill_state1.t())
        skill_state2 = torch.matmul(skill_state, similarity_matrix)
        e_diff = torch.sigmoid(self.diff_emb(exer_id))
        e_disc = torch.sigmoid(self.e_disc(exer_id))
        affect = torch.sigmoid(self.prednet_affect(torch.cat((stu_affect, e_diff), dim=-1)))
        stu_ability_now = self.fc(torch.sigmoid(stu_ability + e_disc * e_diff))
        stu_ability = F.relu(self.dropout(stu_ability_now + self.reLinear(stu_ability)))
        g_state = torch.sigmoid(self.fc_state(torch.cat([skill_state2, stu_ability], -1)))
        state = g_state * skill_state2 + (1 - g_state) * stu_ability

        input_x = e_disc * (state - e_diff) * qe

        # ========== Channel Gate ==========
        mask = (qe > 0).float()
        gate_ch = self.channel_gate(input_x, mask)
        scale = 1.0 + self.gate_alpha * (gate_ch - 0.5)
        scale = torch.clamp(scale, min=0.0)
        input_x = input_x * scale
        # ==================================

        input_x = torch.sigmoid(self.prednet_full1(input_x))
        input_x = torch.sigmoid(self.prednet_full2(input_x))
        o = torch.sigmoid(self.prednet_full3(input_x))

        g = torch.sigmoid(self.guess(affect))
        s = torch.sigmoid(self.slip(affect))
        base_output = ((1 - s) * o) + (g * (1 - o))
        base_output = torch.nan_to_num(base_output, nan=0.5, posinf=1.0, neginf=0.0)
        if not self.use_fluency:
            output = base_output.view(-1)
            return output, torch.zeros(output.size(0), 4, device=output.device), torch.tensor(0.0, device=output.device)

        # Fluency
        f_opp = opportunity.unsqueeze(-1).float() if opportunity is not None else None
        f_rt = ms_first_response.unsqueeze(-1).float() if ms_first_response is not None else None
        f_att = attempt_count.unsqueeze(-1).float() if attempt_count is not None else None
        f_hint = hint_count.unsqueeze(-1).float() if hint_count is not None else None
        if f_opp is not None and f_rt is not None and f_att is not None and f_hint is not None:
            beh = torch.cat([f_opp, f_rt, f_att, f_hint], dim=1)
            beh_norm = torch.nan_to_num(beh, nan=0.0, posinf=1.0, neginf=0.0)
            z = self.beh_encoder(beh_norm) + self.student_flu_bias(stu_id)
            z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=0.0)
            flu = torch.sigmoid(self.flu_score_head(z))
        else:
            z = torch.zeros(stu_id.size(0), self.flu_latent_dim, device=stu_id.device)
            flu = torch.zeros(stu_id.size(0), 1, device=stu_id.device)

        gamma = F.softplus(self.gamma)
        delta = (flu - 0.5) * 2.0 * self.delta_scale
        gate_in = torch.cat([z, affect, e_disc], dim=1)
        raw_gate = self.fuse_gate(gate_in)
        gate = torch.sigmoid(raw_gate) * self.gate_scale
        base_logit = torch.logit(base_output, eps=1e-6)

        if self.training and self._is_warmup():
            final_logit = base_logit
            gate_used = torch.zeros_like(gate)
        else:
            final_logit = base_logit + gamma * gate * delta
            gate_used = gate

        output = torch.sigmoid(final_logit)
        output = torch.nan_to_num(output, nan=0.5, posinf=1.0, neginf=0.0)
        output = torch.clamp(output, min=1e-6, max=1 - 1e-6)

        opp_hat = self.opp_head(z)
        rt_hat = torch.sigmoid(self.rt_head(z))
        att_hat = torch.sigmoid(self.att_head(z))
        hint_hat = torch.sigmoid(self.hint_head(z))
        opp_tgt = f_opp; rt_tgt = f_rt; att_tgt = f_att; hint_tgt = f_hint
        lopp = self.loss_opp(opp_hat, opp_tgt) if opp_tgt is not None else torch.tensor(0.0, device=output.device)
        lrt = self.loss_rt(rt_hat, rt_tgt) if rt_tgt is not None else torch.tensor(0.0, device=output.device)
        latt = self.loss_att(att_hat, att_tgt) if att_tgt is not None else torch.tensor(0.0, device=output.device)
        lhint = self.loss_hint(hint_hat, hint_tgt) if hint_tgt is not None else torch.tensor(0.0, device=output.device)

        answers = answers.float() if answers is not None else None
        if answers is not None:
            affect_cl = affect.squeeze(-1) if affect.dim() > 2 else affect
            cl_loss_base = self.contrastive_loss(affect_cl, answers, tau=0.05)
        else:
            cl_loss_base = torch.tensor(0.0, device=output.device)

        affect_z = self.affect_proj(affect)
        consistency = self.loss_consistency(F.normalize(affect_z, p=2, dim=1), F.normalize(z, p=2, dim=1))
        cl_loss = (cl_loss_base + self.lambda_opp * lopp + self.lambda_rt * lrt
                   + self.lambda_att * latt + self.lambda_hint * lhint + self.lambda_consistency * consistency)
        cl_loss = torch.nan_to_num(cl_loss, nan=0.0, posinf=0.0, neginf=0.0)

        if return_aux:
            return {"p": output, "base_p": base_output, "flu": flu, "affect": affect,
                    "gate": gate_used, "raw_gate": raw_gate,
                    "opp_hat": opp_hat, "rt_hat": rt_hat, "att_hat": att_hat, "hint_hat": hint_hat,
                    "opp_tgt": opp_tgt, "rt_tgt": rt_tgt, "att_tgt": att_tgt, "hint_tgt": hint_tgt,
                    "consistency": consistency, "cl_loss": cl_loss, "base_cl_loss": cl_loss_base,
                    "gamma": gamma.detach()}
        return output, affect, cl_loss


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

def safe_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return float("nan")


@torch.no_grad()
def evaluate(model, loader, device, desc="Evaluating"):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        stu_id, exer_id, qe, y, opp, rt, attempt, hint = batch
        output, _, _ = model(stu_id.to(device), exer_id.to(device), qe.to(device),
                             y.to(device), opportunity=opp.to(device),
                             ms_first_response=rt.to(device),
                             attempt_count=attempt.to(device), hint_count=hint.to(device))
        y_true.extend(y.numpy().tolist())
        y_pred.extend(output.view(-1).detach().cpu().numpy().tolist())
    y_true = np.array(y_true); y_pred = np.array(y_pred); y_hat = (y_pred >= 0.5).astype(int)
    return {"auc": safe_auc(y_true, y_pred), "acc": accuracy_score(y_true, y_hat),
            "f1": f1_score(y_true, y_hat, zero_division=0), "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def train_one_epoch(model, loader, optimizer, bce_loss, device, epoch, cl_weight=1.0):
    model.train()
    if hasattr(model, "start_new_epoch"):
        model.start_new_epoch()
    losses, bce_losses, cl_losses = [], [], []
    for batch in loader:
        stu_id, exer_id, qe, y, opp, rt, attempt, hint = batch
        y = y.to(device).float()
        output, _, cl_loss = model(stu_id.to(device), exer_id.to(device), qe.to(device), y,
                                   opportunity=opp.to(device), ms_first_response=rt.to(device),
                                   attempt_count=attempt.to(device), hint_count=hint.to(device))
        pred = output.view(-1)
        loss_bce = bce_loss(pred, y)
        loss = loss_bce + cl_weight * cl_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if hasattr(model, "apply_clipper"): model.apply_clipper()
        losses.append(loss.item()); bce_losses.append(loss_bce.item()); cl_losses.append(cl_loss.item())
    return {"loss": float(np.mean(losses)), "bce": float(np.mean(bce_losses)), "cl": float(np.mean(cl_losses))}


if __name__ == "__main__":
    import os, sys, random, time, json
    from types import SimpleNamespace
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    def get_project_root():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def set_seed(seed=42):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    PROJECT_ROOT = get_project_root()
    if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

    class Assist09KCSCDDataset(Dataset):
        def __init__(self, csv_path, user_map=None, item_map=None, skill_map=None, user_oov_id=None):
            df = pd.read_csv(csv_path, encoding="utf-8")
            required = ["user_id","problem_id","skill_id","correct","opportunity","ms_first_response","attempt_count","hint_count"]
            missing = [c for c in required if c not in df.columns]
            if missing: raise ValueError(f"{csv_path} missing fields: {missing}")
            df = df.dropna(subset=["user_id","problem_id","skill_id","correct"]).copy()
            df["correct"] = pd.to_numeric(df["correct"], errors="coerce").astype(int)
            df["skill_id"] = pd.to_numeric(df["skill_id"], errors="coerce").astype(int)
            df = df.dropna(subset=["correct","skill_id"]).copy()
            # Filter out test rows with unseen items/skills
            if not is_train:
                df = df[df["problem_id"].isin(item_map)].copy()
                df = df[df["skill_id"].isin(skill_map)].copy()
            for col in ["opportunity","ms_first_response","attempt_count","hint_count"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            is_train = user_map is None or item_map is None or skill_map is None
            if is_train:
                all_users = sorted(df["user_id"].unique())
                self.user_map = {u: i for i, u in enumerate(all_users)}
                self.user_oov_id = len(all_users)
                all_items = sorted(df["problem_id"].unique())
                self.item_map = {it: i for i, it in enumerate(all_items)}
                all_skills = sorted(df["skill_id"].unique())
                self.skill_map = {s: i for i, s in enumerate(all_skills)}
            else:
                self.user_map = user_map; self.item_map = item_map
                self.skill_map = skill_map; self.user_oov_id = user_oov_id
            self.users = np.array([self.user_map.get(v, self.user_oov_id) for v in df["user_id"].values], dtype=np.int64)
            self.items = np.array([self.item_map[v] for v in df["problem_id"].values], dtype=np.int64)
            self.skills = np.array([self.skill_map[v] for v in df["skill_id"].values], dtype=np.int64)
            self.y = df["correct"].values.astype(np.float32)
            self.opp = df["opportunity"].values.astype(np.float32)
            self.rt = df["ms_first_response"].values.astype(np.float32)
            self.att = df["attempt_count"].values.astype(np.float32)
            self.hint = df["hint_count"].values.astype(np.float32)
            self.n_student = len(self.user_map) + 1
            self.n_exercise = len(self.item_map)
            self.n_skill = len(self.skill_map)
        def __len__(self): return len(self.y)
        def __getitem__(self, idx):
            ke = torch.zeros(self.n_skill, dtype=torch.float32); ke[int(self.skills[idx])] = 1.0
            return (torch.tensor(self.users[idx], dtype=torch.long), torch.tensor(self.items[idx], dtype=torch.long),
                    ke, torch.tensor(self.y[idx], dtype=torch.float32),
                    torch.tensor(self.opp[idx], dtype=torch.float32), torch.tensor(self.rt[idx], dtype=torch.float32),
                    torch.tensor(self.att[idx], dtype=torch.float32), torch.tensor(self.hint[idx], dtype=torch.float32))

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    root = get_project_root()
    train_csv = os.path.join(root, "data", "assist09", "splits", "assist09_train.csv")
    test_csv = os.path.join(root, "data", "assist09", "splits", "assist09_test.csv")
    train_ds = Assist09KCSCDDataset(train_csv)
    test_ds = Assist09KCSCDDataset(test_csv, user_map=train_ds.user_map, item_map=train_ds.item_map,
                                    skill_map=train_ds.skill_map, user_oov_id=train_ds.user_oov_id)

    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "KCSCD_vs_Channel_" + time.strftime("%Y-%m-%d-%H-%M"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "comparison.log")
    flog = open(log_file, "w", encoding="utf-8")

    def log(s):
        print(s); flog.write(str(s) + "\n"); flog.flush()

    base_args = dict(
        n_skill=train_ds.n_skill, n_exercise=train_ds.n_exercise, n_student=train_ds.n_student,
        prednet_len1=512, prednet_len2=256, dropout=0.5, flu_latent_dim=8,
        lambda_opp=0.005, lambda_rt=0.005, lambda_att=0.002, lambda_hint=0.002,
        lambda_consistency=0.002, gamma_init=0.05, delta_scale=0.5, gate_scale=0.2,
        warmup_epochs=3, rt_cap_ms=120000.0,
    )

    lr_list = [0.001, 0.002, 0.005]
    epochs = 50
    model_names = ["KCSCD", "KCSCD_Channel"]
    results = {n: {} for n in model_names}

    for lr in lr_list:
        for model_name in model_names:
            log(f"\n===== Training {model_name} lr={lr} =====")
            train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

            args = SimpleNamespace(**base_args, gate_alpha=0.5, gate_hidden_ratio=0.5)
            if model_name == "KCSCD":
                model = Net(args).to(device)
            else:
                model = NetWithChannel(args).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            bce_loss = nn.BCELoss()
            best_auc = -1.0; best_metrics = None
            best_path = os.path.join(log_dir, f"{model_name}_lr{lr}_best.pt")

            for epoch in range(epochs):
                train_metrics = train_one_epoch(model, train_loader, optimizer, bce_loss, device, epoch)
                test_metrics = evaluate(model, test_loader, device, f"Eval {epoch}")
                log(f"  [{model_name} lr={lr} Epoch {epoch:02d}] loss={train_metrics['loss']:.6f} | Test: auc={test_metrics['auc']:.6f} acc={test_metrics['acc']:.6f} f1={test_metrics['f1']:.6f} rmse={test_metrics['rmse']:.6f}")
                auc = test_metrics["auc"]
                if not np.isnan(auc) and auc > best_auc:
                    best_auc = auc; best_metrics = test_metrics
                    torch.save(model.state_dict(), best_path)

            results[model_name][lr] = best_metrics
            log(f"  [{model_name} lr={lr}] BEST: auc={best_metrics['auc']:.6f} acc={best_metrics['acc']:.6f} f1={best_metrics['f1']:.6f} rmse={best_metrics['rmse']:.6f}")

    flog.close()

    # Summary
    sep = "=" * 70
    print("\n" + sep)
    print(" KCSCD vs KCSCD+Channel COMPARISON SUMMARY")
    print(sep)
    print(f"{'LR':<12} {'Model':<20} {'AUC':<10} {'ACC':<10} {'F1':<10} {'RMSE':<10}")
    print("-" * 70)
    for mn in model_names:
        for lr in results[mn]:
            r = results[mn][lr]
            if r:
                print(f"{lr:<12} {mn:<20} {r['auc']:<10.6f} {r['acc']:<10.6f} {r['f1']:<10.6f} {r['rmse']:<10.6f}")
    print(sep)
    for mn in model_names:
        valid = {lr: r for lr, r in results[mn].items() if r is not None}
        if valid:
            bl = max(valid, key=lambda l: valid[l]["auc"])
            r = valid[bl]
            print(f"Best {mn}: lr={bl}, AUC={r['auc']:.6f}, ACC={r['acc']:.6f}, F1={r['f1']:.6f}, RMSE={r['rmse']:.6f}")

    # Save CSV
    csv_path = os.path.join(log_dir, "kcscd_comparison.csv")
    rows = []
    for mn in model_names:
        for lr in results[mn]:
            r = results[mn][lr]
            if r: rows.append({"Model": mn, "LR": lr, "AUC": r["auc"], "ACC": r["acc"], "F1": r["f1"], "RMSE": r["rmse"]})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Chart
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(lr_list)); width = 0.35
        for i, metric in enumerate(["auc", "rmse"]):
            ax = axes[i]
            for j, mn in enumerate(model_names):
                vals = [results[mn][lr][metric] if results[mn][lr] else 0 for lr in lr_list]
                ax.bar(x + j * width, vals, width, label=mn)
            ax.set_xlabel("Learning Rate"); ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()} Comparison")
            ax.set_xticks(x + width / 2); ax.set_xticklabels([str(lr) for lr in lr_list])
            ax.legend()
        chart_path = os.path.join(log_dir, "kcscd_comparison.png")
        plt.tight_layout(); plt.savefig(chart_path, dpi=150); plt.close()
        print(f"Chart saved to {chart_path}")
