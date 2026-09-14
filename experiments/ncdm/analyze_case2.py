import os, torch, sys, numpy as np, pandas as pd
ROOT = r"D:\12487\NCDM PLUS2"
sys.path.insert(0, ROOT)
from models.ncdm.NCDM_Fluency import NCDMFluency
from experiments.ncdm.run_NCDM_Fluency import Assist09FluencyDataset

train_ds = Assist09FluencyDataset(os.path.join(ROOT, "data", "assist09", "splits", "assist09_train.csv"))
m = NCDMFluency(knowledge_n=train_ds.knowledge_n, exer_n=train_ds.exer_n, student_n=train_ds.student_n)
m.net.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "best_ncdm_fluency_assist09.pt"),
    map_location="cpu", weights_only=False))
print(f"Gamma raw param: {m.net.gamma.item():.6f}")
print(f"Gamma (softplus): {torch.nn.functional.softplus(m.net.gamma).item():.6f}")
print(f"fluency_lambda: {m.net.fluency_lambda}")
print(f"Gate MLP: {m.net.fuse_gate}")

df = pd.read_csv(r"D:\12487\NCDM PLUS2\outputs\case_study\all_records.csv")
df["pc"] = (df["y_pred"] - df["core_prob"]).abs()
# Find diverse examples
# 1. Extreme penalty: s_flu very low -> delta negative large -> y_pred much lower than core
# 2. Extreme boost: s_flu very high -> delta positive large -> y_pred much higher than core  
# 3. Near miss: y_pred != y (model wrong)
# 4. Fluency vs mastery conflict: high core_prob with low s_flu
# 5. Consistent: both match well

# Deciles
df["s_flu_bin"] = pd.qcut(df["s_flu"], 10, labels=False, duplicates="drop")
df["delta_bin"] = pd.qcut(df["delta_ij"], 10, labels=False, duplicates="drop")
df["err"] = ((df["y_pred"] >= 0.5).astype(int) != df["y_true"]).astype(int)

samples = []
# High fluency boost
s = df[df["s_flu"] > 0.95].sort_values("pc", ascending=False).head(2)
samples.extend(s.to_dict("records"))
# Low fluency penalty  
s = df[df["s_flu"] < 0.05].sort_values("pc", ascending=False).head(2)
samples.extend(s.to_dict("records"))
# Wrong predictions
s = df[df["err"] == 1].sort_values("pc", ascending=False).head(2)
samples.extend(s.to_dict("records"))
# Mild corrections
s = df[(df["s_flu"] > 0.3) & (df["s_flu"] < 0.7)].sort_values("pc", ascending=False).head(2)
samples.extend(s.to_dict("records"))

print("\n" + "=" * 100)
hdr = f"  {'type':>15s} | o_logit | sigma(o)|  s_flu |  g_ij  |  delta | y_pred |  y  |" 
print(hdr)
print("=" * 100)
for r in samples:
    t = "high_flu" if r["s_flu"] > 0.95 else ("low_flu" if r["s_flu"] < 0.05 else ("wrong" if r["err"]==1 else "mid_flu"))
    print(f"  {t:>15s} | {r['o_logit']:+7.4f} | {r['core_prob']:7.4f} | {r['s_flu']:6.4f} | {r['g_ij']:6.4f} | {r['delta_ij']:+7.4f} | {r['y_pred']:7.4f} | {int(r['y_true']):3d}")

print("\n--- Read the trace for one example ---")
r = samples[0]
print(f"o_logit = {r['o_logit']:.4f}")
print(f"sigma(o_logit) = {r['core_prob']:.4f}  (core prediction, Eq.1)")
print(f"s_flu = {r['s_flu']:.4f}  (fluency score, Eq.9)")
print(f"g_ij = {r['g_ij']:.4f}  (gate, Eq.10)")
print(f"Delta = (s_flu - 0.5) * 4.0 = ({r['s_flu']:.4f} - 0.5) * 4.0 = {r['delta_ij']:.4f}")
g = torch.nn.functional.softplus(m.net.gamma).item()
print(f"gamma = {g:.4f}  (trainable scaling)")
print(f"fluency_lambda = {m.net.fluency_lambda}")
adj = m.net.fluency_lambda * g * r["g_ij"] * r["delta_ij"]
p_logit = r["o_logit"] + adj
import math as mth
p_sig = 1/(1+mth.exp(-p_logit))
print(f"y_pred = sigmoid(o_logit + lambda * gamma * g_ij * Delta)")
print(f"       = sigmoid({r['o_logit']:.4f} + {m.net.fluency_lambda} * {g:.4f} * {r['g_ij']:.4f} * ({r['delta_ij']:.4f}))")
print(f"       = sigmoid({p_logit:.4f})")
print(f"       = {p_sig:.4f}")
print(f"  (matches y_pred = {r['y_pred']:.4f})")
