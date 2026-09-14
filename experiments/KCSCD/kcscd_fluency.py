# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """
    True KCSCD_Fluency:
    - backbone is the original KCSCD model
    - add a conservative fluency branch on top of KCSCD output
    - keep interface compatible with existing train / compare scripts

    Compatible calls:
        output_1, affect_p, CL_loss = net(stu_id, exer_id, qe, answers)

        output_1, affect_p, CL_loss = net(
            stu_id, exer_id, qe, answers,
            opportunity=..., acc_prior=..., stability=...
        )

        aux = net(
            stu_id, exer_id, qe, answers,
            opportunity=..., acc_prior=..., stability=...,
            return_aux=True
        )
    """

    def __init__(self, args):
        super(Net, self).__init__()

        # -------------------------
        # original KCSCD config
        # -------------------------
        self.knowledge_dim = int(args.n_skill)
        self.exer_n = int(args.n_exercise)
        self.emb_num = int(args.n_student)
        self.affect_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1 = int(getattr(args, "prednet_len1", 512))
        self.prednet_len2 = int(getattr(args, "prednet_len2", 256))
        self.dropout_p = float(getattr(args, "dropout", 0.5))

        # -------------------------
        # original KCSCD backbone
        # -------------------------
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

        # -------------------------
        # fluency branch
        # -------------------------
        self.flu_latent_dim = int(getattr(args, "flu_latent_dim", 8))

        # conservative defaults
        self.lambda_rt = float(getattr(args, "lambda_rt", 0.005))      # opp loss
        self.lambda_hint = float(getattr(args, "lambda_hint", 0.005))  # acc loss
        self.lambda_att = float(getattr(args, "lambda_att", 0.002))    # sta loss
        self.lambda_consistency = float(getattr(args, "lambda_consistency", 0.002))

        gamma_init = float(getattr(args, "gamma_init", 0.05))
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

        self.delta_scale = float(getattr(args, "delta_scale", 0.5))
        self.gate_scale = float(getattr(args, "gate_scale", 0.2))
        self.warmup_epochs = int(getattr(args, "warmup_epochs", 3))
        self._epoch_counter = 0

        self.beh_encoder = nn.Sequential(
            nn.Linear(3, 32),
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
        self.acc_head = nn.Sequential(
            nn.Linear(self.flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.sta_head = nn.Sequential(
            nn.Linear(self.flu_latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # gate conditioned on fluency latent + original affect + item difficulty
        self.fuse_gate = nn.Sequential(
            nn.Linear(self.flu_latent_dim + 4 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # consistency: align fluency latent with KCSCD affect representation
        self.affect_proj = nn.Sequential(
            nn.Linear(4, self.flu_latent_dim),
            nn.ReLU(),
        )

        self.loss_opp = nn.MSELoss()
        self.loss_acc = nn.SmoothL1Loss()
        self.loss_sta = nn.SmoothL1Loss()
        self.loss_consistency = nn.MSELoss()

        # -------------------------
        # init
        # -------------------------
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
    def _safe_log1p(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=0.0)
        return torch.log1p(x)

    @staticmethod
    def _to_col(t: torch.Tensor) -> torch.Tensor:
        if t is None:
            return None
        return t.view(-1, 1) if t.dim() == 1 else t

    def _default_proxy(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros(ref.size(0), 1, device=ref.device, dtype=torch.float32)

    def _prepare_proxy_inputs(self, answers, opportunity, acc_prior, stability):
        if answers is None:
            raise ValueError("answers is required for compatibility in current training pipeline.")

        ref = answers if answers.dim() > 1 else answers.view(-1, 1)

        if opportunity is None:
            opportunity = self._default_proxy(ref)
        else:
            opportunity = self._to_col(opportunity.float()).to(ref.device)

        if acc_prior is None:
            acc_prior = self._default_proxy(ref) + 0.5
        else:
            acc_prior = self._to_col(acc_prior.float()).to(ref.device)

        if stability is None:
            stability = self._default_proxy(ref) + 0.25
        else:
            stability = self._to_col(stability.float()).to(ref.device)

        return opportunity, acc_prior, stability

    def forward(
        self,
        stu_id,
        exer_id,
        qe,
        answers=None,
        opportunity=None,
        acc_prior=None,
        stability=None,
        return_aux=False
    ):
        # safety
        stu_id = torch.clamp(stu_id.long(), 0, self.emb_num - 1)
        exer_id = torch.clamp(exer_id.long(), 0, self.exer_n - 1)
        qe = qe.float()

        # =====================================================
        # (1) original KCSCD backbone
        # =====================================================
        stu_affect = torch.sigmoid(self.student_affect(stu_id))          # [B, K]
        skill_state = torch.sigmoid(self.skill_state(stu_id))            # [B, K]
        stu_ability = torch.sigmoid(self.study_ability(stu_id))          # [B, K]

        # stable similarity matrix
        skill_state1 = skill_state.T                                     # [K, B]
        skill_state1 = F.normalize(skill_state1, p=2, dim=1, eps=1e-8)
        similarity_matrix = torch.mm(skill_state1, skill_state1.t())     # [K, K]

        skill_state2 = torch.matmul(skill_state, similarity_matrix)      # [B, K]

        # exercise factor
        e_diff = torch.sigmoid(self.diff_emb(exer_id))                   # [B, K]
        e_disc = torch.sigmoid(self.e_disc(exer_id))                     # [B, 1]

        affect = torch.sigmoid(
            self.prednet_affect(torch.cat((stu_affect, e_diff), dim=-1))
        )                                                                # [B, 4]

        stu_ability_now = self.fc(torch.sigmoid(stu_ability + e_disc * e_diff))
        stu_ability = F.relu(self.dropout(stu_ability_now + self.reLinear(stu_ability)))

        g_state = torch.sigmoid(self.fc_state(torch.cat([skill_state2, stu_ability], -1)))  # [B, 1]
        state = g_state * skill_state2 + (1 - g_state) * stu_ability

        input_x = e_disc * (state - e_diff) * qe
        input_x = torch.sigmoid(self.prednet_full1(input_x))
        input_x = torch.sigmoid(self.prednet_full2(input_x))
        o = torch.sigmoid(self.prednet_full3(input_x))                   # [B, 1]

        g = torch.sigmoid(self.guess(affect))
        s = torch.sigmoid(self.slip(affect))
        base_output = ((1 - s) * o) + (g * (1 - o))                     # [B, 1]
        base_output = torch.nan_to_num(base_output, nan=0.5, posinf=1.0, neginf=0.0)
        base_output = torch.clamp(base_output, min=1e-6, max=1 - 1e-6)

        # original KCSCD contrastive loss
        if answers is None:
            raise ValueError("answers is required.")
        cl_loss_base = self.contrastive_loss(affect, answers, 0.05)
        cl_loss_base = torch.nan_to_num(cl_loss_base, nan=0.0, posinf=0.0, neginf=0.0)

        # =====================================================
        # (2) fluency proxies
        # =====================================================
        opportunity, acc_prior, stability = self._prepare_proxy_inputs(
            answers, opportunity, acc_prior, stability
        )

        f_opp = self._safe_log1p(opportunity)
        f_acc = torch.clamp(acc_prior, 0.0, 1.0)
        f_sta = 1.0 - torch.clamp(4.0 * torch.clamp(stability, min=0.0), 0.0, 1.0)

        beh = torch.cat([f_opp, f_acc, f_sta], dim=1)                    # [B, 3]

        # =====================================================
        # (3) fluency latent
        # =====================================================
        z = self.beh_encoder(beh)                                         # [B, D]
        z = z + 0.1 * torch.tanh(self.student_flu_bias(stu_id))
        flu = torch.sigmoid(self.flu_score_head(z))                       # [B, 1]

        # =====================================================
        # (4) conservative residual fusion on KCSCD output
        # =====================================================
        base_logit = torch.logit(base_output, eps=1e-6)

        gate_in = torch.cat([z, affect, e_disc], dim=1)                   # [B, D+4+1]
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

        # =====================================================
        # (5) fluency auxiliary losses
        # =====================================================
        opp_hat = self.opp_head(z)
        acc_hat = torch.sigmoid(self.acc_head(z))
        sta_hat = torch.sigmoid(self.sta_head(z))

        opp_tgt = f_opp
        acc_tgt = f_acc
        sta_tgt = f_sta

        lopp = self.loss_opp(opp_hat, opp_tgt)
        lacc = self.loss_acc(acc_hat, acc_tgt)
        lsta = self.loss_sta(sta_hat, sta_tgt)

        affect_z = self.affect_proj(affect)
        consistency = self.loss_consistency(
            F.normalize(affect_z, p=2, dim=1),
            F.normalize(z, p=2, dim=1)
        )

        cl_loss = (
            cl_loss_base +
            self.lambda_rt * lopp +
            self.lambda_hint * lacc +
            self.lambda_att * lsta +
            self.lambda_consistency * consistency
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
                "acc_hat": acc_hat,
                "sta_hat": sta_hat,
                "opp_tgt": opp_tgt,
                "acc_tgt": acc_tgt,
                "sta_tgt": sta_tgt,
                "consistency": consistency,
                "cl_loss": cl_loss,
                "base_cl_loss": cl_loss_base,
                "gamma": gamma.detach(),
            }

        # keep original KCSCD style
        output_1 = output
        affect_p = affect
        CL_loss = cl_loss

        return output_1, affect_p, CL_loss

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def contrastive_loss(self, affects, labels, tau=0.05):
        """
        stable supervised contrastive loss
        affects: [B, D]
        labels : [B]
        """
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