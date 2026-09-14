# coding: utf-8
"Generate visualizations from pre-extracted equation trace CSV."

import sys, os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = r"D:\12487\NCDM PLUS2\outputs\case_study"
os.makedirs(OUT_DIR, exist_ok=True)

csv_path = os.path.join(OUT_DIR, "case_study_equation_trace.csv")
if not os.path.exists(csv_path):
    print(f"ERROR: {csv_path} not found")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} pairs from {csv_path}")
print(df.to_string(index=False))

# --- Generate paper table ---
print("\n=== Paper Table ===")
table_cols = ["Pair", "KP", "Beh(opp/acc/sta)", "s_flu(Eq.11)", "g_ij(Eq.10)", 
              "Delta_ij", "Core o(logit)", "Refined y_hat(Eq.12)", "y_true"]
paper_rows = []
for _, r in df.iterrows():
    raw_sid = int(r["student_raw_id"])
    raw_iid = int(r["exercise_raw_id"])
    kc = str(r["knowledge_concepts"]).strip("[]").replace(", ", ",")
    opp = f"{r['opportunity']:.1f}"
    acc = f"{r['acc_prior']:.3f}"
    sta = f"{r['stability']:.3f}"
    paper_rows.append({
        "Pair": f"u_{raw_sid}, e_{raw_iid}",
        "KP": f"KP#{kc}",
        "Beh(opp/acc/sta)": f"{opp}/{acc}/{sta}",
        "s_flu(Eq.11)": f"{r['s_flu (Eq.11)']:.4f}",
        "g_ij(Eq.10)": f"{r['g_ij (Eq.10)']:.4f}",
        "Delta_ij": f"{r['delta_ij']:.4f}",
        "Core o(logit)": f"{r['o_mastery']:.4f}",
        "Refined y_hat(Eq.12)": f"{r['p_refined (Eq.12)']:.4f}",
        "y_true": f"{int(r['y_true'])}",
    })

pt = pd.DataFrame(paper_rows, columns=table_cols)
pt.to_csv(os.path.join(OUT_DIR, "case_study_paper_table.csv"), index=False)
print(pt.to_string(index=False))

# --- Bar chart ---
labels = [f"Stu#{int(r['student_raw_id'])}\nEx#{int(r['exercise_raw_id'])}" for _, r in df.iterrows()]
gates = df["g_ij (Eq.10)"].tolist()
cores = df["o_mastery"].tolist()
refined = df["p_refined (Eq.12)"].tolist()

x = np.arange(len(labels))
w = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

ax1.bar(x, gates, color="#2C5F7C", edgecolor="white", linewidth=0.8, alpha=0.9)
for i, v in enumerate(gates):
    ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax1.set_title("Dynamic Gate {i,j}$ (Eq.10)", fontweight="bold", fontsize=13)
ax1.set_ylabel("Gate Value", fontsize=12)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylim(0, 1.05); ax1.grid(axis="y", alpha=0.3, linestyle="--")

ax2.bar(x - w/2, cores, w, label="Core $\\sigma(o^{logit})$",
        color="#2C5F7C", edgecolor="white", linewidth=0.8, alpha=0.9)
ax2.bar(x + w/2, refined, w, label="Refined $\\hat{y}_{ij}$ (Eq.12)",
        color="#C44E52", edgecolor="white", linewidth=0.8, alpha=0.9)
for i in range(len(x)):
    ax2.annotate(f"$\\Delta$={refined[i]-cores[i]:+.3f}",
                xy=(i, max(cores[i], refined[i]) + 0.03),
                ha="center", va="bottom", fontsize=9, fontstyle="italic", color="#333")
ax2.set_title("Core vs Refined Prediction (Eq.12)", fontweight="bold", fontsize=13)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("Predicted Score", fontsize=12)
ax2.legend(loc="lower left", fontsize=10, framealpha=0.9)
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.set_ylim(0, 1.1)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figure_bar_chart_gate_trace.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "figure_bar_chart_gate_trace.pdf"), bbox_inches="tight")
plt.close()
print(f"\nFigure saved: figure_bar_chart_gate_trace.png/pdf")

# --- Formula trace walkthrough ---
print("\n" + "="*85)
print("  FORMULA TRACE: numerical walkthrough of MFCD")
print("="*85)
for _, r in df.iterrows():
    s = int(r["student_raw_id"]); e = int(r["exercise_raw_id"])
    kc = r["knowledge_concepts"]
    print(f"\n--- Stu #{s}, Ex #{e} (KP{kc}) ---")
    print(f"  Beh: opp={r['opportunity']:.3f} acc={r['acc_prior']:.3f} sta={r['stability']:.3f}")
    ol = r["o_logit (Eq.1)"]
    print(f"  Eq.1:  o_logit={ol:.4f} -> o={r['o_mastery']:.4f}")
    sf = r["s_flu (Eq.11)"]
    dt = r["delta_ij"]
    print(f"  Eq.11: s_flu={sf:.4f}  delta=({sf:.4f}-0.5)*4={dt:.4f}")
    gj = r["g_ij (Eq.10)"]
    print(f"  Eq.10: g_ij={gj:.4f}")
    gm = r["gamma"]; lb = r["fluency_lambda"]; cr = r["correction_term"]
    print(f"  Eq.12: gamma={gm:.4f} lambda={lb}")
    print(f"         correction = gamma*lambda*g*delta = {cr:.4f}")
    final_p = r["p_refined (Eq.12)"]
    print(f"         p_logit = {ol:.4f} + ({cr:.4f}) = {ol+cr:.4f}")
    print(f"         p = sigmoid(p_logit) = {final_p:.4f}")
    print(f"         y_true = {int(r['y_true'])}")

print(f"\nAll outputs in: {OUT_DIR}")
print("Done.")
