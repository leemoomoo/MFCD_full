"""
Case Study Generator for MFCD Paper (Formulas 10, 11, 12).

Extracts and logs intermediate variables:
  - g_ij  (gate value, Eq.10)
  - s_flu (fluency score, Eq.9)
  - Δ_ij  (directional adjustment, Eq.12)
  - o_logit (mastery logit, Eq.1)
  - z_ij (latent fluency state, Eq.5)

Usage: D:\miniconda3\python.exe case_study_generate.py
"""

import os, sys, json, math
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ncdm.NCDM_Fluency import NCDMFluency
from experiments.ncdm.run_NCDM_Fluency import Assist09FluencyDataset


# Paths
TRAIN_CSV  = os.path.join(PROJECT_ROOT, "data", "assist09", "splits", "assist09_train.csv")
TEST_CSV   = os.path.join(PROJECT_ROOT, "data", "assist09", "splits", "assist09_test.csv")
CKPT_PATH  = os.path.join(PROJECT_ROOT, "checkpoints", "best_ncdm_fluency_assist09.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "case_study")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# 1. Load Dataset
train_ds = Assist09FluencyDataset(TRAIN_CSV)
test_ds  = Assist09FluencyDataset(
    TEST_CSV,
    user_map=train_ds.user_map,
    item_map=train_ds.item_map,
    skill_map=train_ds.skill_map,
    user_oov_id=train_ds.user_oov_id,
)

print(f"Train: {len(train_ds)}  Test: {len(test_ds)}")
print(f"Students={train_ds.student_n}, Exercises={train_ds.exer_n}, Knowledge={train_ds.knowledge_n}")

test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

# 2. Load Model
model = NCDMFluency(
    knowledge_n=train_ds.knowledge_n,
    exer_n=train_ds.exer_n,
    student_n=train_ds.student_n,
    lr=0.002,
    fluency_lambda=0.5,
    dropout=0.5,
)

state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model.net.load_state_dict(state)
model.net = model.net.to(DEVICE)
model.net.eval()
print(f"Checkpoint loaded: {CKPT_PATH}")


# 3. Debug forward: extract every intermediate variable
def extract_intermediates(batch):
    user_id, item_id, knowledge_emb, y, opp, acc_prior, stability = [t.to(DEVICE) for t in batch]

    with torch.no_grad():
        out = model.net(
            user_id, item_id, knowledge_emb,
            opp, acc_prior, stability,
            return_aux=True,
        )

    # Mastery backbone (Eq.1 �?o_logit)
    stat_emb = torch.sigmoid(model.net.student_emb(user_id))
    k_diff   = torch.sigmoid(model.net.k_difficulty(item_id))
    e_diff   = torch.sigmoid(model.net.e_difficulty(item_id))

    x = e_diff * (stat_emb - k_diff) * knowledge_emb.float()
    x = model.net.drop_1(torch.sigmoid(model.net.prednet_full1(x)))
    x = model.net.drop_2(torch.sigmoid(model.net.prednet_full2(x)))
    o_logit = model.net.prednet_full3(x)                  # (B,1) raw logit

    # Raw behavioral features (Eq.2-4)
    opp_col = model.net._to_col(opp.float())
    acc_col = model.net._to_col(acc_prior.float())
    sta_col = model.net._to_col(stability.float())

    f_opp = model.net._safe_log1p(opp_col)
    f_acc = torch.clamp(acc_col, 0.0, 1.0)
    f_sta = 1.0 - torch.clamp(4.0 * torch.clamp(sta_col, min=0.0), 0.0, 1.0)
    beh_raw = torch.cat([f_opp, f_acc, f_sta], dim=1)    # (B, 3)

    # Fluency latent (Eq.5: z_ij)
    z = model.net.beh_encoder(beh_raw)
    z = z + torch.tanh(model.net.student_flu_bias(user_id))

    # Fluency score (Eq.9: s_flu)
    flu = torch.sigmoid(model.net.flu_score_head(z))

    # Fusion gate (Eq.10: g_ij)
    f_exe = torch.cat([e_diff, k_diff.mean(dim=1, keepdim=True)], dim=1)
    gate_in = torch.cat([z, f_exe], dim=1)
    gate = torch.sigmoid(model.net.fuse_gate(gate_in))

    # Delta (def. in Eq.12: Δ_ij)
    delta = (flu - 0.5) * 4.0

    # Gamma (trainable scaling)
    gamma = torch.nn.functional.softplus(model.net.gamma)

    # Final prediction (Eq.12: ŷ)
    p_logit = o_logit + model.net.fluency_lambda * gamma * gate * delta

    return {
        "o_logit":   o_logit.detach().cpu(),
        "core_prob": torch.sigmoid(o_logit).detach().cpu(),
        "s_flu":     flu.detach().cpu(),
        "z_ij":      z.detach().cpu(),
        "g_ij":      gate.detach().cpu(),
        "delta_ij":  delta.detach().cpu(),
        "gamma":     gamma.detach().cpu(),
        "y_pred":    out["p"].detach().cpu(),
        "y_true":    y,
        "f_exe":     f_exe.detach().cpu(),
        "beh_raw":   beh_raw.detach().cpu(),
    }


# 4. Run inference on test set
records = []
print("Running inference on test set...")
for i, batch in enumerate(test_loader):
    if i % 5000 == 0:
        print(f"  {i}/{len(test_loader)}")
    try:
        r = extract_intermediates(batch)
        records.append({k: v.item() if v.numel() == 1 else v.numpy().tolist()
                        for k, v in r.items()})
    except Exception as e:
        print(f"  Error at idx {i}: {e}")
        continue

print(f"Collected {len(records)} records.")

import pandas as pd
df = pd.DataFrame(records)


# 5. Find interesting case-study candidates
print("\n--- Gate (g_ij) distribution ---")
print(df["g_ij"].describe())
print("\n--- Delta (Δ_ij) distribution ---")
print(df["delta_ij"].describe())

df["pred_change"] = (df["y_pred"] - df["core_prob"]).abs()
df["adj_sign"]    = np.sign(df["y_pred"] - df["core_prob"])

# Top candidates: high gate (>0.7) + negative delta (<-0.5) + prediction drops
candidates = df[
    (df["g_ij"] > 0.7) &
    (df["delta_ij"] < -0.5) &
    (df["adj_sign"] < 0)
].sort_values("pred_change", ascending=False)

print(f"\nHigh-gate + negative-correction candidates: {len(candidates)}")

if len(candidates) >= 4:
    showcase = candidates.head(8)
else:
    candidates2 = df[df["g_ij"] > 0.5].sort_values("pred_change", ascending=False)
    showcase = candidates2.head(8)

print("\n--- Showcase pairs (intermediate variable trace) ---")
for idx, row in showcase.iterrows():
    print(
        f"  idx={int(idx):>6d}  "
        f"o_logit={row['o_logit']:+.4f}  "
        f"σ(o)={row['core_prob']:.4f}  "
        f"s_flu={row['s_flu']:.4f}  "
        f"g_ij={row['g_ij']:.4f}  "
        f"Δ={row['delta_ij']:+.4f}  "
        f"ŷ={row['y_pred']:.4f}  "
        f"y={int(row['y_true'])}  "
        f"beh=[{','.join(f'{v:.3f}' for v in row['beh_raw'][:3])}]"
    )


# 6. Generate LaTeX table
latex_rows = []
for idx, row in showcase.iterrows():
    latex_rows.append(
        f"        {int(idx)} & "
        f"${row['o_logit']:.3f}$ & "
        f"${row['core_prob']:.3f}$ & "
        f"${row['s_flu']:.3f}$ & "
        f"${row['g_ij']:.3f}$ & "
        f"${row['delta_ij']:.3f}$ & "
        f"${row['y_pred']:.3f}$ & "
        f"${int(row['y_true'])}$ \\\\"
    )

latex_table = r"""\begin{table}[t]
\centering
\caption{Case study of \mymethod: intermediate variable trace for selected student-exercise pairs.}
\label{tab:case_study}
\resizebox{\columnwidth}{!}{
\begin{tabular}{@{}lccccccc@{}}
\toprule
\textbf{$i$} &
\textbf{$o_{i,j}^{\text{logit}}$} &
\textbf{$\sigma(o_{i,j}^{\text{logit}})$} &
\textbf{$s_{i,j}^{\text{flu}}$} &
\textbf{$g_{i,j}$} &
\textbf{$\Delta_{i,j}$} &
\textbf{$\hat{y}_{i,j}$} &
\textbf{$y_{i,j}$} \\
\midrule
""" + "\n".join(latex_rows) + """
\\
\bottomrule
\end{tabular}
}
\end{table}"""

tex_path = os.path.join(OUTPUT_DIR, "case_study_table.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex_table)
print(f"\nLaTeX table: {tex_path}")

df_path = os.path.join(OUTPUT_DIR, "case_study_records.csv")
df.to_csv(df_path, index=False)
print(f"Full records: {df_path}")
print("\nDone.")

