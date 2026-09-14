import os
path = r"D:\12487\NCDM PLUS2\paper\FCD_corrected.tex"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

def fmt_row(ds_names, datasets_upper, datasets_lower, field="baseline"):
    vals = []
    for dn in ds_names:
        d = datasets_upper[dn] if dn in datasets_upper else datasets_lower[dn]
        base_idx = 0 if field == "baseline" else 4
        for j in range(4):
            vals.append(f"{d[base_idx+j]:.2f}")
    return " & ".join(vals)

def fmt_delta_row(ds_names, datasets_upper, datasets_lower):
    vals = []
    for dn in ds_names:
        d = datasets_upper[dn] if dn in datasets_upper else datasets_lower[dn]
        for j in range(4):
            d_val = d[4+j] - d[j]
            improved = d_val < 0 if j == 3 else d_val > 0
            cmd = "redbf" if improved else "green"
            vals.append(f"\\{cmd}{{{d_val:.2f}}}")
    return " & ".join(vals)

du = {
    "assist09": (71.75,68.57,76.29,45.54,71.70,68.67,76.59,45.51),
    "assist17": (87.95,85.43,91.29,31.97,88.20,85.53,91.34,31.86),
    "junyi":    (80.02,79.65,87.19,37.99,79.72,79.73,87.31,38.08),
}
dl = {
    "math1":  (79.99,71.45,69.23,43.44,81.16,72.56,71.67,42.05),
    "math2":  (78.02,70.43,71.95,44.19,79.10,71.81,74.60,43.37),
    "frcsub": (86.39,78.17,78.01,39.10,87.63,79.85,80.85,37.91),
}

upper_baseline = fmt_row(["assist09","assist17","junyi"], du, dl, "baseline")
upper_fluency  = fmt_row(["assist09","assist17","junyi"], du, dl, "fluency")
upper_delta    = fmt_delta_row(["assist09","assist17","junyi"], du, dl)

lower_baseline = fmt_row(["math1","math2","frcsub"], du, dl, "baseline")
lower_fluency  = fmt_row(["math1","math2","frcsub"], du, dl, "fluency")
lower_delta    = fmt_delta_row(["math1","math2","frcsub"], du, dl)

# Build replacement strings
old_upper = "\\multirow{3}{*}{FairWISA} & baseline & 71.92 & 68.75 & 76.41 & 45.46 & --- & --- & --- & --- & --- & --- & --- & --- \\\\\n                      & \\mymethod\\ & 72.79 & 69.60 & 77.55 & 45.02 & --- & --- & --- & --- & --- & --- & --- & --- \\\\\n                      & $\\Delta$ & \\redbf{0.87} & \\redbf{0.85} & \\redbf{1.14} & \\redbf{-0.44} & --- & --- & --- & --- & --- & --- & --- & --- \\\\"
new_upper = f"\\multirow{{3}}{{*}}{{FairWISA}} & baseline & {upper_baseline} \\\\\n                      & \\mymethod\\ & {upper_fluency} \\\\\n                      & $\\Delta$ & {upper_delta} \\\\"

old_lower = "\\multirow{3}{*}{FairWISA} & baseline & 71.92 & 68.75 & 76.41 & 45.46 & --- & --- & --- & --- & --- & --- & --- & --- \\\\\n                      & \\mymethod\\ & 72.79 & 69.60 & 77.55 & 45.02 & --- & --- & --- & --- & --- & --- & --- & --- \\\\\n                      & $\\Delta$ & \\redbf{0.87} & \\redbf{0.85} & \\redbf{1.14} & \\redbf{-0.44} & --- & --- & --- & --- & --- & --- & --- & --- \\\\"
new_lower = f"\\multirow{{3}}{{*}}{{FairWISA}} & baseline & {lower_baseline} \\\\\n                      & \\mymethod\\ & {lower_fluency} \\\\\n                      & $\\Delta$ & {lower_delta} \\\\"

content = content.replace(old_upper, new_upper)
content = content.replace(old_lower, new_lower)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Table updated successfully")