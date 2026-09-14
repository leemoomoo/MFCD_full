
# coding: utf-8
"""Compare KaNCD (base) vs KaNCD+Fluency on ASSIST09."""
import os, sys, random, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
from tqdm import tqdm


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PROJECT_ROOT = get_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =========================
# Shared components
# =========================
class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = torch.abs(self.weight)
        return F.linear(input, weight, self.bias)


# =========================
# Model 1: KaNCD (base)
# =========================
class KaNCDNet(nn.Module):
    def __init__(self, exer_n, student_n, knowledge_n, mf_type="gmf", dim=16, dropout=0.5):
        super().__init__()
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_len1, self.prednet_len2 = 256, 128

        self.student_emb = nn.Embedding(student_n, dim)
        self.exercise_emb = nn.Embedding(exer_n, dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(knowledge_n, dim))
        self.e_discrimination = nn.Embedding(exer_n, 1)

        self.prednet_full1 = PosLinear(knowledge_n, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=dropout)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=dropout)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        if mf_type == "gmf":
            self.k_diff_full = nn.Linear(dim, 1)
            self.stat_full = nn.Linear(dim, 1)
        elif mf_type == "ncf1":
            self.k_diff_full = nn.Linear(2 * dim, 1)
            self.stat_full = nn.Linear(2 * dim, 1)
        elif mf_type == "ncf2":
            self.k_diff_full1 = nn.Linear(2 * dim, dim)
            self.k_diff_full2 = nn.Linear(dim, 1)
            self.stat_full1 = nn.Linear(2 * dim, dim)
            self.stat_full2 = nn.Linear(dim, 1)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

    def _compute_mastery(self, stu_id, exer_id):
        stu_emb = self.student_emb(stu_id)                           # [B, D]
        exer_emb = self.exercise_emb(exer_id)                        # [B, D]
        K, D = self.knowledge_n, self.emb_dim
        B = stu_emb.shape[0]
        kn_emb = self.knowledge_emb.unsqueeze(0).expand(B, K, D)     # [B, K, D]

        s = stu_emb.unsqueeze(1).expand(B, K, D)
        if self.mf_type == "gmf":
            stat_emb = torch.sigmoid(self.stat_full(s * kn_emb)).view(B, -1)
        elif self.mf_type == "ncf1":
            stat_emb = torch.sigmoid(self.stat_full(torch.cat([s, kn_emb], dim=-1))).view(B, -1)
        elif self.mf_type == "ncf2":
            h = torch.sigmoid(self.stat_full1(torch.cat([s, kn_emb], dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(h)).view(B, -1)

        e = exer_emb.unsqueeze(1).expand(B, K, D)
        if self.mf_type == "gmf":
            k_difficulty = torch.sigmoid(self.k_diff_full(e * kn_emb)).view(B, -1)
        elif self.mf_type == "ncf1":
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat([e, kn_emb], dim=-1))).view(B, -1)
        elif self.mf_type == "ncf2":
            h = torch.sigmoid(self.k_diff_full1(torch.cat([e, kn_emb], dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(h)).view(B, -1)

        e_disc = torch.sigmoid(self.e_discrimination(exer_id))        # [B, 1]
        return stat_emb, k_difficulty, e_disc

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        stu_id = torch.clamp(stu_id, 0, self.student_n - 1)
        input_exercise = torch.clamp(input_exercise, 0, self.exer_n - 1)
        stat_emb, k_difficulty, e_disc = self._compute_mastery(stu_id, input_exercise)
        input_x = e_disc * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))
        return output.view(-1)


# =========================
# Model 2: KaNCD + Fluency
# =========================
class KaNCDNetWithFluency(nn.Module):
    def __init__(self, exer_n, student_n, knowledge_n, mf_type="gmf", dim=16, dropout=0.5,
                 flu_latent_dim=16, gamma_init=0.05, rt_cap_ms=120000.0):
        super().__init__()
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.rt_cap_ms = float(rt_cap_ms)
        self.prednet_len1, self.prednet_len2 = 256, 128

        # --- Base KaNCD ---
        self.student_emb = nn.Embedding(student_n, dim)
        self.exercise_emb = nn.Embedding(exer_n, dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(knowledge_n, dim))
        self.e_discrimination = nn.Embedding(exer_n, 1)
        self.prednet_full1 = PosLinear(knowledge_n, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=dropout)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=dropout)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        if mf_type == "gmf":
            self.k_diff_full = nn.Linear(dim, 1)
            self.stat_full = nn.Linear(dim, 1)
        elif mf_type == "ncf1":
            self.k_diff_full = nn.Linear(2 * dim, 1)
            self.stat_full = nn.Linear(2 * dim, 1)
        elif mf_type == "ncf2":
            self.k_diff_full1 = nn.Linear(2 * dim, dim)
            self.k_diff_full2 = nn.Linear(dim, 1)
            self.stat_full1 = nn.Linear(2 * dim, dim)
            self.stat_full2 = nn.Linear(dim, 1)

        # --- Fluency ---
        self.beh_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(64, flu_latent_dim), nn.ReLU(),
        )
        self.student_flu_bias = nn.Embedding(student_n, flu_latent_dim)

        self.flu_score_head = nn.Sequential(
            nn.Linear(flu_latent_dim, 16), nn.ReLU(), nn.Linear(16, 1),
        )
        self.opp_head = nn.Sequential(nn.Linear(flu_latent_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        self.rt_head = nn.Sequential(nn.Linear(flu_latent_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        self.att_head = nn.Sequential(nn.Linear(flu_latent_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        self.hint_head = nn.Sequential(nn.Linear(flu_latent_dim, 16), nn.ReLU(), nn.Linear(16, 1))

        self.fuse_gate = nn.Sequential(
            nn.Linear(flu_latent_dim + 2, 32), nn.ReLU(), nn.Linear(32, 1),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

    def _compute_mastery(self, stu_id, exer_id):
        # Same as KaNCDNet._compute_mastery
        stu_emb = self.student_emb(stu_id)
        exer_emb = self.exercise_emb(exer_id)
        K, D = self.knowledge_n, self.emb_dim
        B = stu_emb.shape[0]
        kn_emb = self.knowledge_emb.unsqueeze(0).expand(B, K, D)
        s = stu_emb.unsqueeze(1).expand(B, K, D)
        if self.mf_type == "gmf":
            stat_emb = torch.sigmoid(self.stat_full(s * kn_emb)).view(B, -1)
        elif self.mf_type == "ncf1":
            stat_emb = torch.sigmoid(self.stat_full(torch.cat([s, kn_emb], dim=-1))).view(B, -1)
        elif self.mf_type == "ncf2":
            h = torch.sigmoid(self.stat_full1(torch.cat([s, kn_emb], dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(h)).view(B, -1)
        e = exer_emb.unsqueeze(1).expand(B, K, D)
        if self.mf_type == "gmf":
            k_difficulty = torch.sigmoid(self.k_diff_full(e * kn_emb)).view(B, -1)
        elif self.mf_type == "ncf1":
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat([e, kn_emb], dim=-1))).view(B, -1)
        elif self.mf_type == "ncf2":
            h = torch.sigmoid(self.k_diff_full1(torch.cat([e, kn_emb], dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(h)).view(B, -1)
        e_disc = torch.sigmoid(self.e_discrimination(exer_id))
        return stat_emb, k_difficulty, e_disc

    @staticmethod
    def _to_col(t: torch.Tensor) -> torch.Tensor:
        return t.view(-1, 1) if t.dim() == 1 else t

    @staticmethod
    def _safe_log1p(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=0.0)
        return torch.log1p(x)

    def forward(self, stu_id, exer_id, knowledge_emb,
                opportunity=None, ms_first_response=None,
                attempt_count=None, hint_count=None, return_aux=False):
        stu_id = torch.clamp(stu_id, 0, self.student_n - 1)
        exer_id = torch.clamp(exer_id, 0, self.exer_n - 1)

        # --- Base KaNCD ---
        stat_emb, k_difficulty, e_disc = self._compute_mastery(stu_id, exer_id)
        input_x = e_disc * (stat_emb - k_difficulty) * knowledge_emb.float()
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        o_logit = self.prednet_full3(input_x)
        o = torch.sigmoid(o_logit)

        # --- Fluency ---
        opp = self._to_col(opportunity.float())
        rt = self._to_col(ms_first_response.float())
        att = self._to_col(attempt_count.float())
        hnt = self._to_col(hint_count.float())

        f_opp = self._safe_log1p(opp)
        f_rt = self._safe_log1p(rt) / np.log1p(self.rt_cap_ms)
        f_att = self._safe_log1p(att)
        f_hint = self._safe_log1p(hnt)

        beh = torch.cat([f_opp, f_rt, f_att, f_hint], dim=1)
        z = self.beh_encoder(beh) + torch.tanh(self.student_flu_bias(stu_id))
        flu = torch.sigmoid(self.flu_score_head(z))

        # --- Gate & correction ---
        gate_in = torch.cat([z, e_disc, k_difficulty.mean(dim=1, keepdim=True)], dim=1)
        gate = torch.sigmoid(self.fuse_gate(gate_in))
        gamma = F.softplus(self.gamma)
        delta = (flu - 0.5) * 4.0
        p_logit = o_logit + gamma * gate * delta
        p = torch.sigmoid(p_logit)

        opp_hat = self.opp_head(z).view(-1)
        rt_hat = torch.sigmoid(self.rt_head(z)).view(-1)
        att_hat = torch.sigmoid(self.att_head(z)).view(-1)
        hint_hat = torch.sigmoid(self.hint_head(z)).view(-1)

        if not return_aux:
            return p.view(-1)

        return {
            "p": p.view(-1), "o": o.view(-1), "flu": flu.view(-1),
            "gate": gate.view(-1), "gamma": gamma.detach(),
            "opp_hat": opp_hat, "opp_tgt": f_opp.view(-1),
            "rt_hat": rt_hat, "rt_tgt": f_rt.view(-1),
            "att_hat": att_hat, "att_tgt": f_att.view(-1),
            "hint_hat": hint_hat, "hint_tgt": f_hint.view(-1),
        }


# =========================
# Dataset
# =========================
class Assist09FluencyDataset(Dataset):
    """Returns (user_id, item_id, knowledge_emb, y, opp, rt, attempt, hint)."""
    def __init__(self, csv_path, user_map=None, item_map=None, skill_map=None, user_oov_id=None):
        df = pd.read_csv(csv_path, encoding="utf-8")
        req = ["user_id","problem_id","skill_id","correct",
               "opportunity","ms_first_response","attempt_count","hint_count"]
        miss = [c for c in req if c not in df.columns]
        if miss: raise ValueError(f"Missing: {miss}")
        df = df.dropna(subset=req).copy()
        df["correct"] = pd.to_numeric(df["correct"], errors="coerce").astype(int)
        df["skill_id"] = pd.to_numeric(df["skill_id"], errors="coerce").astype(int)
        df = df.dropna(subset=["correct","skill_id"]).copy()
        for col in ["opportunity","ms_first_response","attempt_count","hint_count"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        is_train = user_map is None or item_map is None or skill_map is None
        def build_map(vals):
            uniq = pd.unique(vals)
            return {v:i for i,v in enumerate(uniq)}

        if is_train:
            user_map = build_map(df["user_id"].values)
            item_map = build_map(df["problem_id"].values)
            skill_map = build_map(df["skill_id"].values)

        self.user_map = user_map
        self.item_map = item_map
        self.skill_map = skill_map
        self.user_oov_id = len(user_map) if user_oov_id is None else user_oov_id

        if not is_train:
            before = len(df)
            df = df[df["problem_id"].isin(item_map)].copy()
            df = df[df["skill_id"].isin(skill_map)].copy()
            print(f"[{os.path.basename(csv_path)}] filtered: {before} -> {len(df)}")

        self.users = np.array([user_map.get(v, self.user_oov_id) for v in df["user_id"].values], dtype=np.int64)
        self.items = np.array([item_map[v] for v in df["problem_id"].values], dtype=np.int64)
        self.skills = np.array([skill_map[v] for v in df["skill_id"].values], dtype=np.int64)
        self.y = df["correct"].values.astype(np.float32)
        self.opp = df["opportunity"].values.astype(np.float32)
        self.rt = df["ms_first_response"].values.astype(np.float32)
        self.att = df["attempt_count"].values.astype(np.float32)
        self.hint = df["hint_count"].values.astype(np.float32)

        self.student_n = len(user_map) + 1
        self.exer_n = len(item_map)
        self.knowledge_n = len(skill_map)
        print(f"  samples={len(self.y)}, students={self.student_n}, items={self.exer_n}, skills={self.knowledge_n}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        ke = torch.zeros(self.knowledge_n, dtype=torch.float32)
        ke[int(self.skills[idx])] = 1.0
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            ke,
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.opp[idx], dtype=torch.float32),
            torch.tensor(self.rt[idx], dtype=torch.float32),
            torch.tensor(self.att[idx], dtype=torch.float32),
            torch.tensor(self.hint[idx], dtype=torch.float32),
        )


# =========================
# Training & evaluation
# =========================
def safe_auc(y_true, y_pred):
    try: return roc_auc_score(y_true, y_pred)
    except ValueError: return float("nan")

def train_one_model(name, model, train_loader, test_loader, epochs, lr, device,
                    is_fluency=False, lambda_opp=0.1, lambda_rt=0.05,
                    lambda_att=0.05, lambda_hint=0.05):
    loss_fn = nn.BCELoss()
    loss_opp = nn.MSELoss()
    loss_rt = nn.SmoothL1Loss()
    loss_att = nn.SmoothL1Loss()
    loss_hint = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_auc = -1.0
    best_metrics = None

    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"{name} E{epoch}", leave=False):
            uid, iid, kq, y, opp, rt, att, hnt = batch
            uid = uid.to(device); iid = iid.to(device); kq = kq.to(device)
            y = y.to(device).float()
            opp = opp.to(device); rt = rt.to(device); att = att.to(device); hnt = hnt.to(device)

            if is_fluency:
                out = model(uid, iid, kq, opp, rt, att, hnt, return_aux=True)
                ly = loss_fn(out["p"], y)
                lopp = loss_opp(out["opp_hat"], out["opp_tgt"])
                lrt = loss_rt(out["rt_hat"], out["rt_tgt"])
                latt = loss_att(out["att_hat"], out["att_tgt"])
                lhint = loss_hint(out["hint_hat"], out["hint_tgt"])
                loss = ly + lambda_opp*lopp + lambda_rt*lrt + lambda_att*latt + lambda_hint*lhint
            else:
                pred = model(uid, iid, kq)
                loss = loss_fn(pred, y)

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())

        # Evaluate
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                uid, iid, kq, y, opp, rt, att, hnt = batch
                if is_fluency:
                    pred = model(uid.to(device), iid.to(device), kq.to(device),
                                 opp.to(device), rt.to(device), att.to(device), hnt.to(device))
                else:
                    pred = model(uid.to(device), iid.to(device), kq.to(device))
                y_pred.extend(pred.view(-1).cpu().tolist())
                y_true.extend(y.tolist())

        y_true = np.array(y_true); y_pred = np.array(y_pred); y_hat = (y_pred >= 0.5).astype(int)
        auc = safe_auc(y_true, y_pred)
        acc = accuracy_score(y_true, y_hat)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        print(f"  [{name}] E{epoch:02d} loss={float(np.mean(losses)):.6f}  auc={auc:.6f}  acc={acc:.6f}  rmse={rmse:.6f}  f1={f1:.6f}")
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_metrics = {"auc": auc, "acc": acc, "rmse": rmse, "f1": f1}

    return best_metrics


# =========================
# Main
# =========================
def main():
    save_dir = os.path.join(PROJECT_ROOT, "outputs", "results", "kancd_vs_fluency")
    os.makedirs(save_dir, exist_ok=True)

    train_csv = os.path.join(PROJECT_ROOT, "data", "assist09", "splits", "assist09_train.csv")
    test_csv = os.path.join(PROJECT_ROOT, "data", "assist09", "splits", "assist09_test.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import time
    log_dir = os.path.join(save_dir, time.strftime("%Y-%m-%d-%H-%M"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "comparison.log")
    flog = open(log_file, "w", encoding="utf-8")
    def log(s): print(s); flog.write(str(s) + "\n"); flog.flush()

    def set_seed(s=42):
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)
    log(f"Device: {device}")

    train_ds = Assist09FluencyDataset(train_csv)
    test_ds = Assist09FluencyDataset(test_csv,
        user_map=train_ds.user_map, item_map=train_ds.item_map,
        skill_map=train_ds.skill_map, user_oov_id=train_ds.user_oov_id)

    lr_list = [0.001, 0.002, 0.005]
    epochs = 30
    configs = [
        ("KaNCD", lambda: KaNCDNet(
            exer_n=train_ds.exer_n, student_n=train_ds.student_n,
            knowledge_n=train_ds.knowledge_n, mf_type="gmf", dim=16, dropout=0.5), False),
        ("KaNCD+Fluency", lambda: KaNCDNetWithFluency(
            exer_n=train_ds.exer_n, student_n=train_ds.student_n,
            knowledge_n=train_ds.knowledge_n, mf_type="gmf", dim=16,
            dropout=0.5, flu_latent_dim=16, gamma_init=0.05), True),
    ]
    results = {}

    for lr in lr_list:
        for name, build_fn, is_flu in configs:
            set_seed(42)
            log(f"\n===== Training {name} lr={lr} =====")
            train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

            model = build_fn().to(device)
            metrics = train_one_model(name, model, train_loader, test_loader, epochs, lr, device, is_flu)
            results[(name, lr)] = metrics
            log(f"  [{name} lr={lr}] BEST: auc={metrics['auc']:.6f} acc={metrics['acc']:.6f} rmse={metrics['rmse']:.6f} f1={metrics['f1']:.6f}")
            del model; torch.cuda.empty_cache()

    flog.close()

    # Summary
    sep = "=" * 70
    print("\n" + sep)
    print(" KaNCD vs KaNCD+Fluency COMPARISON SUMMARY")
    print(sep)
    print(f"{'LR':<12} {'Model':<20} {'AUC':<10} {'ACC':<10} {'F1':<10} {'RMSE':<10}")
    print("-" * 70)
    for lr in lr_list:
        for name, _, _ in configs:
            r = results.get((name, lr))
            if r: print(f"{lr:<12} {name:<20} {r['auc']:<10.6f} {r['acc']:<10.6f} {r['f1']:<10.6f} {r['rmse']:<10.6f}")
    print(sep)
    for name, _, _ in configs:
        valid = {lr: results[(name, lr)] for lr in lr_list if (name, lr) in results and results[(name, lr)]}
        if valid:
            bl = max(valid, key=lambda l: valid[l]["auc"])
            r = valid[bl]
            print(f"Best {name}: lr={bl}, AUC={r['auc']:.6f}, ACC={r['acc']:.6f}, F1={r['f1']:.6f}, RMSE={r['rmse']:.6f}")

    csv_path = os.path.join(log_dir, "kancd_comparison.csv")
    rows = [{"Model": n, "LR": lr, **results[(n, lr)]} for lr in lr_list for n, _, _ in configs if (n, lr) in results and results[(n, lr)]]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(lr_list)); width = 0.35
        for i, metric in enumerate(["auc", "rmse"]):
            ax = axes[i]
            for j, (name, _, _) in enumerate(configs):
                vals = [results[(name, lr)][metric] if (name, lr) in results and results[(name, lr)] else 0 for lr in lr_list]
                ax.bar(x + j * width, vals, width, label=name)
            ax.set_xlabel("Learning Rate"); ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()} Comparison")
            ax.set_xticks(x + width / 2); ax.set_xticklabels([str(lr) for lr in lr_list])
            ax.legend()
        chart_path = os.path.join(log_dir, "kancd_comparison.png")
        plt.tight_layout(); plt.savefig(chart_path, dpi=150); plt.close()
        print(f"Chart saved to {chart_path}")
    except ImportError:
        pass

if __name__ == "__main__":
    main()
