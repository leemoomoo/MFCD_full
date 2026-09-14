import pandas as pd, numpy as np
import os
df = pd.read_csv(r"D:\12487\NCDM PLUS2\outputs\case_study\all_records.csv")
print(f"Total records: {len(df)}")
# Distribution summary
print("\n--- Gate (g_ij) distribution ---")
print(df["g_ij"].describe())
print("\n--- Fluency (s_flu) distribution ---")
print(df["s_flu"].describe())
print("\n--- Delta distribution ---")
print(df["delta_ij"].describe())

# Diverse showcase: pick interesting examples
# 1. High gate + large neg delta (penalty) 
# 2. High gate + large pos delta (boost)
# 3. Low gate (mastery dominates)
# 4. Mid gate various
df["pc"] = (df["y_pred"] - df["core_prob"]).abs()
df["as"] = np.sign(df["y_pred"] - df["core_prob"])

# Deduplicate by rounding to 3 dec
df["key"] = df[["o_logit","s_flu","g_ij","delta_ij"]].round(3).astype(str).agg("_".join, axis=1)
du = df.drop_duplicates(subset="key")

print(f"\nUnique states (rounded): {len(du)}")

# Categories
high_neg = du[(du["g_ij"] > 0.8) & (du["delta_ij"] < -1.0)]
high_pos = du[(du["g_ij"] > 0.8) & (du["delta_ij"] > 1.0)]
low_gate = du[du["g_ij"] < 0.3]
mid_few  = du[(du["g_ij"] > 0.4) & (du["g_ij"] < 0.7)]

picks = []
picks.extend(high_neg.sort_values("pc", ascending=False).head(3).to_dict("records"))
picks.extend(high_pos.sort_values("pc", ascending=False).head(3).to_dict("records"))
picks.extend(low_gate.sort_values("pc", ascending=False).head(1).to_dict("records"))
picks.extend(mid_few.sort_values("pc", ascending=False).head(1).to_dict("records"))

print("\n" + "=" * 112)
hdr = f"  {'type':>8s} | {'o_logit':>8s} | {'sigma(o)':>8s} | {'s_flu':>6s} | {'g_ij':>6s} | {'delta':>7s} | {'y_pred':>7s} | {'y':>3s} | beh[EI EA RC]"
print(hdr)
print("=" * 112)
for r in picks:
    cat = "penalty" if r["delta_ij"] < -0.5 and r["g_ij"] > 0.7 else ("boost" if r["delta_ij"] > 0.5 and r["g_ij"] > 0.7 else ("low_gate" if r["g_ij"] < 0.3 else "mid_gate"))
    print(f"  {cat:>8s} | {r['o_logit']:+8.4f} | {r['core_prob']:8.4f} | {r['s_flu']:6.4f} | {r['g_ij']:6.4f} | {r['delta_ij']:+7.4f} | {r['y_pred']:7.4f} | {int(r['y_true']):3d} | [{r['beh_fp']:.3f} {r['beh_fa']:.3f} {r['beh_fs']:.3f}]")

# Save picks
pd.DataFrame(picks).to_csv(r"D:\12487\NCDM PLUS2\outputs\case_study\diverse_showcase.csv", index=False)
print(f"\nDiverse showcase saved.")
print("Done.")
